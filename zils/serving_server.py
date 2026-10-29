"""HTTP endpoint a miner runs: signed manifest on GET, hotkey-authenticated signed inference on POST.

This coordinator holds the hotkey and runs HTTP, auth, nonces, signing and deadlines in the miner's
interpreter. Inference runs in a child process under the pinned d3 interpreter (zils/serving_worker.py),
spoken to over JSON lines; `start_worker` starts it and returns one `Served` per checkpoint."""

import itertools
import json
import queue
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import ROOT, decisions, serving

MAX_QUESTIONS = 32
PER_IP = 2  # concurrent connections one source address may hold


def log(text):
    """Operator log on stderr: exception classes and fixed text only, never request content."""
    print(f"zils serve: {text}", file=sys.stderr, flush=True)


class WorkerError(Exception):
    """The worker refused one request: kind is "too_long", "invalid", "runtime" or (from a shared
    adapter runtime, zils/serving_runtime.py) "busy"."""

    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


class Worker:
    """The child process. Callers serialize requests (the server's inference lock); any timeout, broken
    pipe or garbled reply kills the child, and every later call raises OSError."""

    def __init__(self, process, timeout):
        self.process, self.timeout = process, timeout
        self.lines, self.ids, self.exited = queue.Queue(), itertools.count(), threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)
        self.process.wait()
        self.exited.set()

    def stop(self):
        self.process.kill()

    def close(self):
        """Stop the child and release its pipes."""
        self.stop()
        self.exited.wait()
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                pipe.close()
            except OSError:  # a request left unflushed in a broken pipe
                pass

    def receive(self, timeout=None):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.stop()
            raise OSError("serving worker timed out") from None
        if line is None:
            self.lines.put(None)  # stays dead for the next caller
            raise OSError("serving worker exited")
        try:
            return json.loads(line)
        except ValueError:
            self.stop()
            raise OSError("serving worker wrote a malformed line") from None

    def call(self, model, state, questions):
        rid = next(self.ids)
        try:
            self.process.stdin.write(
                json.dumps({"id": rid, "model": model, "questions": questions, "state": state})
                + "\n"
            )
            self.process.stdin.flush()
        except (OSError, ValueError):  # broken pipe, or stdin already closed
            raise OSError("serving worker exited") from None
        reply = self.receive(self.timeout)
        if not isinstance(reply, dict) or reply.get("id") != rid:
            self.stop()
            raise OSError("serving worker answered out of order")
        return reply


class Served:
    """One checkpoint the worker loaded: model, weights_sha256, max_input_tokens, predict()."""

    def __init__(self, worker, model, weights_sha256, max_input_tokens):
        self.worker, self.model = worker, model
        self.weights_sha256, self.max_input_tokens = weights_sha256, max_input_tokens

    def predict(self, state, questions):
        """{qid: {"probabilities", "input_tokens"}}; WorkerError on a refusal, OSError if the worker is gone."""
        reply = self.worker.call(self.model, state, questions)
        if "error" in reply:
            raise WorkerError(
                reply["error"].get("kind") if isinstance(reply["error"], dict) else None
            )
        return reply["predictions"]


def start_worker(checkpoints, python, args=("-m", "zils.serving_worker"), timeout=120.0):
    """Start the worker under `python` (the d3 interpreter), wait while it loads, return its `Served`.
    `timeout` bounds each request's inference, not the load."""
    flags = [flag for path in checkpoints for flag in ("--checkpoint", str(Path(path).resolve()))]
    process = subprocess.Popen(
        [str(python), *args, *flags],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    worker = Worker(process, timeout)
    try:
        ready = worker.receive()
        if ready.get("ready") is not True or not ready["served"]:
            raise OSError("serving worker did not report ready with a checkpoint")
        return [Served(worker, **item) for item in ready["served"]]
    except (OSError, AttributeError, KeyError, TypeError):
        worker.close()
        raise ValueError(
            f"serving worker failed to start (exit {process.wait()}); see its error above"
        ) from None


class Refusal(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


# A worker refusal's HTTP answer; "runtime" and a lost worker are 503 (the node, not the caller).
# "busy" is a shared runtime's full queue: callers read 503 with any code but "unavailable" as busy.
REFUSALS = {
    "too_long": (413, "input_too_long", "Input exceeds the served token limit."),
    "invalid": (422, "invalid_request", "Request could not be encoded."),
    "busy": (503, "busy", "Inference is busy; retry shortly."),
}
UNAVAILABLE = (503, "unavailable", "Inference is unavailable; retry later.")


def make_server(
    key,
    endpoint,
    served,
    allowed,
    bind,
    slots=8,
    clock=time.time,
    deadline=15.0,
    header_deadline=3.0,
):
    """`deadline` bounds reading the whole request, `header_deadline` the request line and headers."""
    by_model = {s.model: s for s in served}
    if len(by_model) != len(served):
        raise ValueError("two served checkpoints have the same model")
    manifest = [
        {
            "model": s.model,
            "weights_sha256": s.weights_sha256,
            "max_input_tokens": s.max_input_tokens,
        }
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
        if not (
            isinstance(payload["model"], str)
            and isinstance(payload["weights_sha256"], str)
            and isinstance(payload["body"], dict)
        ):
            raise Refusal(400, "invalid_request", "Request fields have the wrong type.")
        model = by_model.get(payload["model"])
        if model is None:
            raise Refusal(404, "model_not_found", "Model is not served here.")
        if payload["weights_sha256"] != model.weights_sha256:
            raise Refusal(409, "weights_mismatch", "Weights do not match.")
        body = payload["body"]
        if body.get("model") != payload["model"]:
            raise Refusal(400, "model_mismatch", "Body model differs from the signed model.")
        if "images" in body or "videos" in body:
            raise Refusal(422, "unsupported_media", "This endpoint serves text only.")
        try:
            decisions.validate_request(body)
        except decisions.DecisionError as error:
            raise Refusal(error.status, error.code, str(error)) from None
        questions = body["questions"]
        if len(questions) > MAX_QUESTIONS:
            raise Refusal(
                413, "too_many_questions", "Request has more than %d questions." % MAX_QUESTIONS
            )
        with infer_lock:  # one request on the GPU at a time
            try:
                predictions = model.predict(body["state"], questions)
            except WorkerError as failure:
                if failure.kind in REFUSALS:
                    raise Refusal(*REFUSALS[failure.kind]) from None
                log(f"inference failed (worker reported {failure.kind})")
                raise Refusal(*UNAVAILABLE) from None
            except OSError as failure:
                log(f"inference failed ({type(failure).__name__}: {failure})")
                raise Refusal(*UNAVAILABLE) from None
        digest = serving.request_digest(payload)
        signed = serving.answer_payload(
            key.ss58_address, digest, model.model, model.weights_sha256, predictions
        )
        return serving.sign(key, "answer", signed)

    class Handler(BaseHTTPRequestHandler):
        timeout = 10  # per-read timeout; `deadline` below bounds the total

        def setup(self):
            super().setup()
            self.sent = False
            self.watchdog = threading.Timer(
                deadline, self.expire
            )  # total read deadline, trickle-proof
            self.headers_due = threading.Timer(
                header_deadline, self.expire
            )  # idle connections free fast
            for timer in (self.watchdog, self.headers_due):
                timer.daemon = True
                timer.start()

        def parse_request(self):
            ok = super().parse_request()  # reads the headers; cancel only once they are in
            self.headers_due.cancel()
            return ok

        def expire(self):
            try:
                self.request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        def finish(self):
            self.watchdog.cancel()
            self.headers_due.cancel()
            try:
                super().finish()
            except OSError:
                pass

        def log_message(self, *args):
            pass

        def send_json(self, status, value, headers=()):
            data = json.dumps(value).encode()
            self.sent = True
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for name, text in headers:
                self.send_header(name, text)
            self.end_headers()
            self.wfile.write(data)

        def refuse(self, status, code, message):
            self.send_json(status, {"error": {"code": code, "message": message}})

        def handle_one_request(self):
            try:
                super().handle_one_request()
            except OSError:  # client vanished mid-write
                self.close_connection = True
            except Exception as error:
                self.close_connection = True
                log(f"internal error ({type(error).__name__})")
                if not self.sent:
                    try:
                        self.refuse(500, "internal_error", "Internal error.")
                    except OSError:
                        pass

        def do_GET(self):
            if self.path != "/v1/serving":
                return self.refuse(404, "not_found", "Not found.")
            body = serving.manifest_body(key.ss58_address, endpoint, manifest, now=int(clock()))
            self.send_json(200, serving.sign(key, "manifest", body))

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self.refuse(404, "not_found", "Not found.")
            length = self.headers.get("Content-Length", "")
            if not (length.isascii() and length.isdigit()):
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
            self.watchdog.cancel()  # fully read: slow inference must not be killed
            try:
                result = answer(raw)
            except Refusal as error:
                return self.refuse(error.status, error.code, error.message)
            self.send_json(200, result)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        gate = threading.BoundedSemaphore(slots)
        peers, peers_lock = {}, threading.Lock()  # open connections per source address
        nonces = seen

        def handle_error(self, request, client_address):
            pass

        def take(self, address):
            with self.peers_lock:
                if self.peers.get(address, 0) >= PER_IP or not self.gate.acquire(blocking=False):
                    return False
                self.peers[address] = self.peers.get(address, 0) + 1
                return True

        def give_back(self, address):
            with self.peers_lock:
                self.peers[address] -= 1
                if not self.peers[address]:
                    del self.peers[address]
                self.gate.release()

        def process_request(self, request, client_address):
            if not self.take(client_address[0]):
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
                self.give_back(client_address[0])
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self.give_back(client_address[0])

    return Server(bind, Handler)
