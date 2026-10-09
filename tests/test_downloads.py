"""Wall-clock transfer budgets cover blocked headers and slow body streams."""

import copy
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from unittest.mock import patch

from tests import test_image_server
from tests.test_queue import server
from zils.cloud import APIError, download
from zils.decisions import DecisionError
from zils.image_server import ImageRuntime


def handler(raw, *, slow_headers=False):
    class Handler(BaseHTTPRequestHandler):
        first = True
        connected = threading.Event()

        def do_GET(self):
            slow, type(self).first = type(self).first, False
            self.connected.set()
            try:
                if slow and slow_headers:
                    for byte in b"HTTP/1.0 200 OK\r\n":
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(0.08)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                for chunk in [bytes([b]) for b in raw] if slow else [raw]:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    if slow:
                        time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    return Handler


class DownloadDeadlineTest(unittest.TestCase):
    def test_slow_headers_and_body_obey_wall_clock_deadline(self):
        for headers in (False, True):
            with (
                self.subTest(headers=headers),
                server(handler(b"x" * 40, slow_headers=headers)) as origin,
                tempfile.TemporaryDirectory() as tmp,
            ):
                target = Path(tmp) / "download"
                started = time.monotonic()
                with self.assertRaises(APIError) as failure:
                    download(origin + "/slow", target, 1024, max_seconds=0.4)
                self.assertEqual(failure.exception.status, 503)
                self.assertLess(time.monotonic() - started, 0.9)
                self.assertFalse(target.exists())

    def test_timed_out_download_releases_image_execution_for_next_request(self):
        fixture = test_image_server.ImageRuntimeTest()
        fixture.setUp()
        try:
            raw = fixture.image.data
            with server(handler(raw)) as origin:
                envelope = copy.deepcopy(fixture.envelope)
                envelope["image"]["url"] = envelope["image"]["url"].replace(fixture.origin, origin)
                engine = test_image_server.Engine()
                runtime = ImageRuntime(
                    engine,
                    {"image-release": {"fingerprint": "a" * 64, "temperature": 1.0}},
                    origin,
                    token="secret",
                    timeout=0.8,
                )
                try:
                    with patch(
                        "zils.image_server.download",
                        side_effect=lambda url, path, limit, **kw: download(
                            url, path, limit, max_seconds=0.4
                        ),
                    ):
                        with self.assertRaises(DecisionError):
                            runtime.dispatch("POST", "/v1/systemone", "secret", envelope, "slow")
                        result = runtime.dispatch(
                            "POST", "/v1/systemone", "secret", envelope, "next"
                        )
                        self.assertEqual(result[0], 200)
                        self.assertEqual(engine.executions, 1)
                finally:
                    runtime.close()
        finally:
            fixture.doCleanups()
