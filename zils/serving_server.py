"""HTTP endpoint a miner runs: signed manifest on GET, hotkey-authenticated signed inference on POST."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import decisions, models, serving


class Served:
    """One loaded checkpoint: model, weights_sha256, max_input_tokens, count_tokens(), predict()."""

    def __init__(self, checkpoint, device):
        from . import checkpoint_hash, d3

        self.decision = d3.DecisionModel(checkpoint, device)
        self.model = self.decision.meta["model"]
        self.weights_sha256 = checkpoint_hash(checkpoint)
        self.max_input_tokens = models.spec(self.model).get("max_input_tokens", 2048)
        self._cache = {}

    def _encode(self, state, question):
        key = json.dumps([state, question], sort_keys=True)
        if key not in self._cache:
            if len(self._cache) >= 64:  # a rejected request leaves entries behind; bound them
                self._cache.clear()
            self._cache[key] = self.decision.encoder.inputs(state, question)
        return self._cache[key]

    def count_tokens(self, state, question):
        return len(self._encode(state, question)[0])

    def predict(self, state, question):
        ids, keys = self._encode(state, question)
        self._cache.pop(json.dumps([state, question], sort_keys=True))  # runs under the inference lock
        probs = self.decision.probabilities(ids, keys, question, self.decision.meta["temperature"])
        return probs, len(ids)


def load_served(checkpoint, device="cuda"):
    checkpoint = Path(checkpoint)
    meta = models.metadata(checkpoint)
    if models.artifact_kind(checkpoint) == "adapter" or meta is None or meta["model"] not in models.d3_family():
        raise ValueError("serving needs a public d3-family checkpoint, not an adapter")
    return Served(checkpoint, device)


class Refusal(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


def make_server(key, endpoint, served, allowed, bind, slots=8, clock=time.time):
    by_model = {s.model: s for s in served}
    manifest = [
        {"model": s.model, "weights_sha256": s.weights_sha256, "max_input_tokens": s.max_input_tokens}
        for s in served
    ]
    seen, seen_lock, infer_lock = {}, threading.Lock(), threading.Lock()

    def fresh(caller, nonce, now):
        with seen_lock:
            for k in [k for k, t in seen.items() if now - t > 2 * serving.FRESH_SECONDS]:
                del seen[k]
            if (caller, nonce) in seen:
                return False
            seen[caller, nonce] = now
            return True

    def answer(raw):
        try:
            msg = decisions.decode_body(raw)
        except decisions.DecisionError as error:
            raise Refusal(error.status, error.code, str(error)) from None
        payload = msg.get("payload") if isinstance(msg, dict) else None
        caller = payload.get("caller") if isinstance(payload, dict) else None
        now = int(clock())
        if not isinstance(caller, str):
            raise Refusal(401, "unauthorized", "Invalid signed request.")
        try:
            payload = serving.verify(msg, "request", hotkey=caller, now=now)
        except ValueError:
            raise Refusal(401, "unauthorized", "Invalid signed request.") from None
        if payload["audience"] != key.ss58_address:
            raise Refusal(401, "unauthorized", "Request is addressed to another miner.")
        if not allowed(caller):
            raise Refusal(403, "forbidden", "Caller is not permitted.")
        if not isinstance(payload["nonce"], str) or not fresh(caller, payload["nonce"], now):
            raise Refusal(401, "unauthorized", "Nonce already used.")
        model = by_model.get(payload["model"])
        if model is None:
            raise Refusal(404, "model_not_found", "Model is not served here.")
        if payload["weights_sha256"] != model.weights_sha256:
            raise Refusal(409, "weights_mismatch", "Weights do not match.")
        body = payload["body"]
        if isinstance(body, dict) and ("images" in body or "videos" in body):
            raise Refusal(422, "unsupported_media", "This endpoint serves text only.")
        try:
            decisions.validate_request(body)
        except decisions.DecisionError as error:
            raise Refusal(error.status, error.code, str(error)) from None
        questions = body["questions"]
        with infer_lock:  # one request on the GPU at a time
            if any(model.count_tokens(body["state"], q) > model.max_input_tokens for q in questions.values()):
                raise Refusal(413, "input_too_long", "Input exceeds the served token limit.")
            predictions = {}
            for qid, q in questions.items():
                probs, tokens = model.predict(body["state"], q)
                predictions[qid] = {"probabilities": probs, "input_tokens": tokens}
        digest = serving.request_digest(payload)
        return serving.sign(key, "answer", serving.answer_payload(digest, model.model, model.weights_sha256, predictions))

    class Handler(BaseHTTPRequestHandler):
        timeout = 10  # per-connection socket timeout: a stalled client cannot hold a slot

        def log_message(self, *args):
            pass

        def send_json(self, status, value, headers=()):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for name, text in headers:
                self.send_header(name, text)
            self.end_headers()
            self.wfile.write(data)

        def refuse(self, status, code, message):
            self.send_json(status, {"error": {"code": code, "message": message}})

        def do_GET(self):
            if self.path != "/v1/serving":
                return self.refuse(404, "not_found", "Not found.")
            body = serving.manifest_body(key.ss58_address, endpoint, manifest, now=int(clock()))
            self.send_json(200, serving.sign(key, "manifest", body))

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self.refuse(404, "not_found", "Not found.")
            length = self.headers.get("Content-Length", "")
            if not length.isdigit():
                self.close_connection = True
                return self.refuse(411, "length_required", "Content-Length is required.")
            if int(length) > decisions.MAX_BODY:
                self.close_connection = True
                return self.refuse(413, "body_too_large", "Request exceeds the body limit.")
            try:
                raw = self.rfile.read(int(length))
                if len(raw) != int(length):
                    raise ConnectionError()
            except OSError:  # timeout or reset: drop the client
                self.close_connection = True
                return
            try:
                result = answer(raw)
            except Refusal as error:
                return self.refuse(error.status, error.code, error.message)
            self.send_json(200, result)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        gate = threading.BoundedSemaphore(slots)

        def process_request(self, request, client_address):
            if not self.gate.acquire(blocking=False):
                try:
                    request.sendall(
                        b"HTTP/1.0 503 Service Unavailable\r\nRetry-After: 1\r\nContent-Length: 0\r\n\r\n"
                    )
                except OSError:
                    pass
                self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except BaseException:
                self.gate.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self.gate.release()

    return Server(bind, Handler)
