"""Exercise real loopback HTTP framing and error serialization."""

import importlib
import importlib.util
import socket
import threading
import unittest
from contextlib import contextmanager

import requests


@contextmanager
def server(module, dispatch, origin=None, **options):
    service = module.Server(("127.0.0.1", 0), module.make_handler(dispatch, origin, **options))
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        yield service.server_port
    finally:
        service.shutdown()
        service.server_close()
        thread.join(2)


class TransportTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("fez.decision_http"), "HTTP transport is missing"
        )
        return importlib.import_module("fez.decision_http")

    def test_json_auth_and_request_ids(self):
        m = self.module()

        def dispatch(method, path, bearer, body, request_id):
            return 200, {"token_present": bearer == "secret", "body": body}

        with server(m, dispatch) as port:
            r = requests.post(
                f"http://127.0.0.1:{port}/v1/systemone",
                json={"a": 1},
                headers={"Authorization": "Bearer secret"},
            )
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.json()["token_present"])
            self.assertEqual(r.headers["Cache-Control"], "no-store")
            self.assertEqual(len(r.headers["X-Request-ID"]), 36)
            self.assertEqual(requests.post(f"http://127.0.0.1:{port}/x", json={}).status_code, 401)

    def test_ambiguous_framing_and_bad_json(self):
        m = self.module()
        with server(m, lambda *args: (200, {})) as port:
            for headers, body in [
                ("Content-Length: 2\r\nContent-Length: 2", "{}"),
                ("Content-Length: 2\r\nTransfer-Encoding: chunked", "{}"),
                ("Content-Length: 13", '{"x":1,"x":2}'),
            ]:
                with (
                    self.subTest(headers=headers),
                    socket.create_connection(("127.0.0.1", port)) as s,
                ):
                    raw = f"POST /x HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer x\r\nContent-Type: application/json\r\n{headers}\r\n\r\n{body}"
                    s.sendall(raw.encode())
                    s.shutdown(socket.SHUT_WR)
                    response = b""
                    while chunk := s.recv(65536):
                        response += chunk
                    self.assertIn(b" 400 ", response.split(b"\r\n")[0])

    def test_cors_and_errors_exclude_private_exception(self):
        m = self.module()
        from fez.decisions import DecisionError

        def dispatch(*args):
            raise DecisionError(422, "invalid_request", "Invalid field", ["body", "state"])

        with server(m, dispatch, "https://app.example") as port:
            url = f"http://127.0.0.1:{port}/x"
            preflight = requests.options(url, headers={"Origin": "https://app.example"})
            self.assertEqual(preflight.status_code, 204)
            self.assertEqual(preflight.headers["Content-Length"], "0")
            self.assertEqual(
                requests.options(url, headers={"Origin": "https://evil.example"}).status_code, 403
            )
            r = requests.post(
                url, json={}, headers={"Authorization": "Bearer x", "Origin": "https://app.example"}
            )
            self.assertEqual(r.status_code, 422)
            self.assertEqual(r.json()["detail"][0]["loc"], ["body", "state"])
            self.assertEqual(r.headers["Access-Control-Allow-Origin"], "https://app.example")

        def crash(*args):
            raise RuntimeError("private secret")

        with server(m, crash) as port:
            r = requests.get(f"http://127.0.0.1:{port}/x", headers={"Authorization": "Bearer x"})
            self.assertEqual(r.status_code, 500)
            self.assertNotIn("private secret", r.text)
