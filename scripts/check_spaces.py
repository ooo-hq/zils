"""Verify private Spaces behavior using unique, disposable catalog identities."""

import argparse
import json
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

from scripts.migrate_storage import write_report
from zils import settings
from zils.cloud import APIError, Supabase, upload
from zils.spaces import SpacesStorage, configured_client
from zils.storage import LIMITS, ObjectCatalog, fingerprint

GATES = (
    "private_read",
    "multipart_immutable",
    "roundtrip_bytes",
    "roundtrip_sha256",
    "catalog_recovery",
    "cors",
)


def passed(report):
    return all(report.get(key) is True for key in GATES)


class LostCommitReply:
    """Inject only reply loss after a real database commit, never fabricate the result."""

    def __init__(self, db):
        self.db, self.lost = db, False

    def rpc(self, name, values):
        result = self.db.rpc(name, values)
        if name == "zils_storage_object" and values.get("p_action") == "commit" and not self.lost:
            self.lost = True
            raise APIError(503, "Injected lost commit reply.")
        return result


def cors_allows(response, origin, method):
    headers = response.headers
    methods = {
        part.strip().upper() for part in headers.get("Access-Control-Allow-Methods", "").split(",")
    }
    allowed = {
        part.strip().lower() for part in headers.get("Access-Control-Allow-Headers", "").split(",")
    }
    return (
        response.status_code in (200, 204)
        and headers.get("Access-Control-Allow-Origin") == origin
        and method in methods
        and "content-type" in allowed
        and headers.get("Access-Control-Allow-Credentials", "").lower() != "true"
    )


def check(source, store, origin):
    parsed = urlsplit(origin)
    if (
        parsed.scheme != "https"
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or "*" in origin
    ):
        raise ValueError("Use an exact HTTPS frontend origin")
    origin = origin.rstrip("/")
    report = {key: False for key in GATES}
    report["version"] = "zils-spaces-preflight/v1"
    namespace = "fez-training-models"
    run_id = str(uuid.uuid4())
    paths = [f"{run_id}/preflight/replay", f"{run_id}/preflight/checkpoint"]
    try:
        size, sha256 = fingerprint(source)
        if not 0 < size <= LIMITS[namespace]:
            raise ValueError("Checkpoint exceeds the storage limit")
        report.update(source_bytes=size, source_sha256=sha256)
        with tempfile.TemporaryDirectory(prefix="zils-spaces-check-") as directory:
            root = Path(directory)
            small = root / "small"
            small.write_bytes(b"zils-private-multipart-preflight")
            grant = store.signed(namespace, paths[0], upload=True)
            headers = {
                "Origin": origin,
                "Access-Control-Request-Method": "PUT",
                "Access-Control-Request-Headers": "content-type",
            }
            cors = True
            for method in ("PUT", "GET", "HEAD"):
                with requests.options(
                    grant["url"],
                    headers=headers | {"Access-Control-Request-Method": method},
                    timeout=(10, 30),
                    allow_redirects=False,
                ) as response:
                    cors = cors_allows(response, origin, method) and cors
            wrong = "https://zils-storage-preflight.invalid"
            with requests.options(
                grant["url"],
                headers=headers | {"Origin": wrong},
                timeout=(10, 30),
                allow_redirects=False,
            ) as response:
                report["cors"] = cors and response.headers.get(
                    "Access-Control-Allow-Origin"
                ) not in (wrong, "*")
            upload(grant["url"], small, grant["headers"], grant["method"])
            pending = store.catalog.get(namespace, paths[0])
            proxy = LostCommitReply(store.catalog.db)
            recovery = SpacesStorage(ObjectCatalog(proxy), store.client, store.bucket)
            try:
                recovery.exists(namespace, paths[0])
            except APIError as error:
                if error.status != 503 or not proxy.lost:
                    raise
            report["catalog_recovery"] = proxy.lost and store.exists(namespace, paths[0])
            # A consumed ID may return its original completion idempotently.
            committed = store.catalog.get(namespace, paths[0])
            original_etag = store._head(committed)["ETag"]
            consumed = False
            try:
                repeated = store._call(
                    "complete_multipart_upload",
                    **store._location(pending),
                    UploadId=pending["upload_id"],
                    MultipartUpload={"Parts": [{"PartNumber": 1, "ETag": committed["part_etag"]}]},
                )
                consumed = repeated.get("ETag", "").strip('"') == original_etag.strip('"')
            except APIError as error:
                consumed = error.status == 404
            with requests.put(
                grant["url"],
                data=b"replay overwrite",
                headers=grant["headers"],
                timeout=(10, 30),
                allow_redirects=False,
            ) as response:
                replay_denied = response.status_code in (400, 403, 404)
            replay = root / "replay"
            store.download(namespace, paths[0], replay, 1024)
            report["multipart_immutable"] = (
                consumed and replay_denied and fingerprint(replay) == fingerprint(small)
            )
            url = urlsplit(store.signed(namespace, paths[0])["url"])
            anonymous = urlunsplit((url.scheme, url.netloc, url.path, "", ""))
            with requests.get(
                anonymous, stream=True, timeout=(10, 30), allow_redirects=False
            ) as response:
                report["private_read"] = response.status_code in (403, 404)
            store.upload(namespace, paths[1], source)
            destination = root / "checkpoint"
            store.download(namespace, paths[1], destination, LIMITS[namespace])
            count, digest = fingerprint(destination)
            report.update(downloaded_bytes=count, downloaded_sha256=digest)
            report["roundtrip_bytes"] = count == size
            report["roundtrip_sha256"] = digest == sha256
    except Exception as error:
        report["error_code"] = type(error).__name__
    finally:
        try:
            store.remove(namespace, paths)
            report["cleanup_scheduled"] = True
        except Exception as error:
            report["cleanup_scheduled"] = False
            report["cleanup_error_code"] = type(error).__name__
    report["passed"] = passed(report) and report.get("cleanup_scheduled") is True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    client, bucket = configured_client()
    try:
        result = check(
            args.source,
            SpacesStorage(ObjectCatalog(Supabase()), client, bucket),
            settings.required("ZILS_WEB_ORIGIN"),
        )
    finally:
        client.close()
    write_report(args.report, result)
    print(json.dumps({key: result[key] for key in (*GATES, "passed")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
