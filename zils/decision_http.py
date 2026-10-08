"""Bounded development HTTP transport shared by Zils gateway and private runtime."""

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .decisions import MAX_BODY, MAX_DEPTH, DecisionError, decode_body


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler):
        super().__init__(address, handler)
        self.slots = threading.BoundedSemaphore(32)

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\nRetry-After: 1\r\nConnection: close\r\n\r\n"
                )
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


def make_handler(
    dispatch,
    origin=None,
    *,
    body_limit=MAX_BODY,
    max_depth=MAX_DEPTH,
    allowed_methods=("GET", "POST"),
):
    if not allowed_methods or set(allowed_methods) - {"GET", "POST", "DELETE"}:
        raise ValueError("Unsupported HTTP method configuration")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, *args):
            pass

        def send(self, status, body, request_id, retry_after=None):
            data = (
                b""
                if status == 204
                else json.dumps(body, allow_nan=False, ensure_ascii=False).encode("utf-8")
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Request-ID", request_id)
            self.send_header("Connection", "close")
            self.send_header("Vary", "Origin")
            if origin and self.headers.get("Origin") == origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header(
                    "Access-Control-Allow-Methods", ", ".join((*allowed_methods, "OPTIONS"))
                )
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Expose-Headers", "X-Request-ID, Retry-After")
            if retry_after is not None:
                self.send_header("Retry-After", str(max(1, int(retry_after))))
            self.end_headers()
            if status != 204 and self.command != "HEAD":
                self.wfile.write(data)
            self.close_connection = True

        def handle_request(self):
            request_id = str(uuid.uuid4())
            try:
                if self.headers.get("Origin") is not None and (
                    not origin or self.headers.get("Origin") != origin
                ):
                    raise DecisionError(403, "origin_not_allowed", "Origin is not allowed.")
                if self.command == "OPTIONS":
                    if not origin or self.headers.get("Origin") != origin:
                        raise DecisionError(403, "origin_not_allowed", "Origin is not allowed.")
                    self.send(204, {}, request_id)
                    return
                if self.command not in allowed_methods:
                    raise DecisionError(405, "method_not_allowed", "Method is not supported.")
                auth = self.headers.get_all("Authorization", [])
                if (
                    len(auth) != 1
                    or not auth[0].startswith("Bearer ")
                    or not 1 <= len(auth[0][7:]) <= 8192
                ):
                    raise DecisionError(
                        401, "invalid_credentials", "A valid bearer credential is required."
                    )
                if (
                    self.headers.get("Transfer-Encoding") is not None
                    or self.headers.get("Content-Encoding") is not None
                ):
                    raise DecisionError(
                        400, "invalid_framing", "Encoded request bodies are not supported."
                    )
                lengths = self.headers.get_all("Content-Length", [])
                if (
                    len(lengths) > 1
                    or (lengths and not lengths[0].isascii())
                    or (lengths and not lengths[0].isdigit())
                ):
                    raise DecisionError(400, "invalid_framing", "Invalid content length.")
                length = int(lengths[0]) if lengths else 0
                if length > body_limit:
                    raise DecisionError(413, "body_too_large", "Request exceeds the body limit.")
                body = {}
                if self.command == "POST":
                    if not lengths:
                        raise DecisionError(400, "invalid_framing", "Content-Length is required.")
                    types = self.headers.get_all("Content-Type", [])
                    if (
                        len(types) != 1
                        or types[0].split(";")[0].strip().lower() != "application/json"
                    ):
                        raise DecisionError(415, "unsupported_media_type", "Use application/json.")
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise DecisionError(400, "invalid_framing", "Incomplete request body.")
                    body = decode_body(raw, limit=body_limit, max_depth=max_depth)
                elif length:
                    raise DecisionError(
                        400, "invalid_framing", "GET requests must not have a body."
                    )
                status, result = dispatch(self.command, self.path, auth[0][7:], body, request_id)
                self.send(status, result, request_id)
            except DecisionError as error:
                result = {
                    "error": {"code": error.code, "message": str(error)},
                    "request_id": request_id,
                }
                if error.field is not None:
                    result["detail"] = [{"loc": error.field, "msg": str(error), "type": error.code}]
                self.send(error.status, result, request_id, error.retry_after)
            except (TimeoutError, ConnectionError):
                self.send(
                    408,
                    {
                        "error": {"code": "request_timeout", "message": "Request interrupted."},
                        "request_id": request_id,
                    },
                    request_id,
                )
            except Exception:
                self.send(
                    500,
                    {
                        "error": {"code": "internal_error", "message": "Unexpected service error."},
                        "request_id": request_id,
                    },
                    request_id,
                )

        def safe_handle(self):
            try:
                self.handle_request()
            except (OSError, ValueError):
                self.close_connection = True

        do_GET = safe_handle
        do_POST = safe_handle
        do_OPTIONS = safe_handle
        do_DELETE = safe_handle
        do_PUT = safe_handle
        do_PATCH = safe_handle
        do_HEAD = safe_handle

    return Handler
