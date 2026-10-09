"""Local S3 HTTP transport and disposable PostgreSQL; neither is a cloud service."""

import hashlib
import json
import shutil
import subprocess
import tempfile
import threading
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from xml.sax.saxutils import escape

from tests.api_database import Database


@contextmanager
def database():
    with tempfile.TemporaryDirectory(prefix="zils-storage-db-") as directory:
        root = Path(directory)
        data, socket = root / "data", root / "socket"
        socket.mkdir()
        subprocess.run(
            ["initdb", "-D", str(data), "-A", "trust", "--no-locale"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "pg_ctl",
                "-D",
                str(data),
                "-l",
                str(root / "log"),
                "-o",
                f"-k {socket} -c listen_addresses=''",
                "-w",
                "start",
            ],
            check=True,
            capture_output=True,
        )
        command = ["psql", "-X", "-v", "ON_ERROR_STOP=1", "-h", str(socket), "-d", "postgres"]
        try:
            for source in (
                "tests/sql/queue-bootstrap.sql",
                "supabase/migrations/202610090001_spaces_storage.sql",
            ):
                proc = subprocess.run([*command, "-f", source], capture_output=True, text=True)
                if proc.returncode:
                    raise RuntimeError(proc.stderr)
            yield Database(command)
        finally:
            subprocess.run(
                ["pg_ctl", "-D", str(data), "-m", "immediate", "-w", "stop"], capture_output=True
            )


def md5(raw):
    return hashlib.md5(raw, usedforsecurity=False).hexdigest()


class S3:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.uploads, self.objects = {}, {}
        self.lose_completion = False
        self.lose_creation = False
        self.deny_lists = False
        self.lock = threading.RLock()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status=200, xml="", headers=None):
                raw = xml.encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Content-Type", "application/xml")
                for key, value in (headers or {}).items():
                    self.send_header(key, str(value))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(raw)

            def fail(self, code, status=404):
                self.reply(
                    status, f"<Error><Code>{code}</Code><Message>fixture failure</Message></Error>"
                )

            def dispatch(self):
                parsed = urlsplit(self.path)
                query = parse_qs(parsed.query, keep_blank_values=True)
                key = unquote(parsed.path).split("/", 2)[-1]
                uid = query.get("uploadId", [None])[0]
                upload = fixture.uploads.get(uid)
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if self.command == "POST" and "uploads" in query:
                    uid = str(uuid.uuid4())
                    fixture.uploads[uid] = {
                        "key": key,
                        "generation": self.headers.get("x-amz-meta-zils-generation"),
                        "parts": {},
                        "created": datetime.now(timezone.utc).isoformat(),
                    }
                    if fixture.lose_creation:
                        return self.fail("InternalError", 500)
                    return self.reply(
                        xml=f"<InitiateMultipartUploadResult><Bucket>test-storage</Bucket><Key>{escape(key)}</Key><UploadId>{uid}</UploadId></InitiateMultipartUploadResult>"
                    )
                if self.command == "GET" and "uploads" in query:
                    content = "".join(
                        f"<Upload><Key>{escape(v['key'])}</Key><UploadId>{k}</UploadId><Initiated>{v['created']}</Initiated></Upload>"
                        for k, v in fixture.uploads.items()
                    )
                    return self.reply(
                        xml=f"<ListMultipartUploadsResult><IsTruncated>false</IsTruncated>{content}</ListMultipartUploadsResult>"
                    )
                if self.command == "GET" and query.get("list-type") == ["2"]:
                    content = "".join(
                        f"<Contents><Key>{escape(k)}</Key><LastModified>{v['created']}</LastModified><Size>{v['file'].stat().st_size}</Size></Contents>"
                        for k, v in fixture.objects.items()
                    )
                    return self.reply(
                        xml=f"<ListBucketResult><IsTruncated>false</IsTruncated>{content}</ListBucketResult>"
                    )
                if uid:
                    if not upload or upload["key"] != key:
                        return self.fail("NoSuchUpload")
                    if self.command == "PUT":
                        number = int(query["partNumber"][0])
                        target = fixture.root / f"{uid}-{number}"
                        target.write_bytes(raw)
                        upload["parts"][number] = {
                            "file": target,
                            "etag": '"' + md5(raw) + '"',
                            "size": len(raw),
                        }
                        return self.reply(headers={"ETag": upload["parts"][number]["etag"]})
                    if self.command == "GET":
                        if fixture.deny_lists:
                            return self.fail("AccessDenied", 403)
                        parts = "".join(
                            f"<Part><PartNumber>{n}</PartNumber><ETag>{escape(p['etag'])}</ETag><Size>{p['size']}</Size></Part>"
                            for n, p in upload["parts"].items()
                        )
                        return self.reply(
                            xml=f"<ListPartsResult><IsTruncated>false</IsTruncated>{parts}</ListPartsResult>"
                        )
                    if self.command == "DELETE":
                        del fixture.uploads[uid]
                        return self.reply(204)
                    if self.command == "POST":
                        parts = ET.fromstring(raw)
                        requested = [
                            (int(e.findtext("{*}PartNumber")), e.findtext("{*}ETag")) for e in parts
                        ]
                        if len(requested) != 1 or requested[0][0] not in upload["parts"]:
                            return self.fail("InvalidPart", 400)
                        part = upload["parts"][requested[0][0]]
                        if part["etag"] != requested[0][1]:
                            return self.fail("InvalidPart", 400)
                        target = fixture.root / str(uuid.uuid4())
                        shutil.copyfile(part["file"], target)
                        etag = '"' + md5(bytes.fromhex(part["etag"].strip('"'))) + '-1"'
                        fixture.objects[key] = {
                            "file": target,
                            "etag": etag,
                            "generation": upload["generation"],
                            "created": datetime.now(timezone.utc).isoformat(),
                        }
                        del fixture.uploads[uid]
                        if fixture.lose_completion:
                            fixture.lose_completion = False
                            return self.fail("InternalError", 500)
                        return self.reply(
                            xml=f"<CompleteMultipartUploadResult><ETag>{escape(etag)}</ETag></CompleteMultipartUploadResult>"
                        )
                if self.command in ("HEAD", "GET"):
                    obj = fixture.objects.get(key)
                    if not obj:
                        return self.fail("NoSuchKey")
                    self.send_response(200)
                    self.send_header("Content-Length", str(obj["file"].stat().st_size))
                    self.send_header("ETag", obj["etag"])
                    self.send_header("x-amz-meta-zils-generation", obj["generation"])
                    self.end_headers()
                    if self.command == "GET":
                        with obj["file"].open("rb") as source:
                            shutil.copyfileobj(source, self.wfile)
                    return
                if self.command == "DELETE":
                    fixture.objects.pop(key, None)
                    return self.reply(204)
                self.fail("InvalidRequest", 400)

            def handle_request(self):
                with fixture.lock:
                    self.dispatch()

            do_GET = do_HEAD = do_PUT = do_POST = do_DELETE = handle_request

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()


class LostReplyDB:
    """Drop one RPC reply after the real SQL transaction commits."""

    def __init__(self, db, action):
        self.db, self.action = db, action

    def rpc(self, name, args):
        from zils.cloud import APIError

        result = self.db.rpc(name, args)
        if args.get("p_action") == self.action:
            self.action = None
            raise APIError(503, "Lost database reply")
        return result

    def rows(self, *args):
        return self.db.rows(*args)

    def request(self, method, route, data=None, headers=None):
        if method == "DELETE" and route.startswith("/rest/v1/zils_storage_retired?"):
            generation = parse_qs(urlsplit(route).query)["generation"][0].removeprefix("eq.")
            uuid.UUID(generation)
            return self.db.sql(f"delete from zils_storage_retired where generation='{generation}'")
        raise AssertionError(json.dumps([method, route]))
