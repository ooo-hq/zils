"""`zils serve --runtime`'s backend against a fake adapter runtime on loopback; no GPU, no platform."""

import contextlib
import hashlib
import io
import json
import os
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from unittest.mock import patch

from zils import checkpoint_hash, models, protocol, serving_runtime, serving_server

MODEL = models.D3
BASE = f"{MODEL}-base"
TOKEN = "SECRET-RUNTIME-TOKEN-" + "x" * 24
QUESTION = {"type": "noul", "instructions": None, "criteria": None}


def fingerprint(model=MODEL):
    identity = models.profile_identity(model)
    return hashlib.sha256(
        (identity["profile_sha256"] + identity["runtime_sha256"]).encode()
    ).hexdigest()


def health(model=MODEL, **changes):
    """What the adapter runtime's GET /health returns with --serve-base."""
    value = {
        "profile": models.spec(model),
        **models.profile_identity(model),
        "ready": True,
        "models": [
            {"release_id": "zils-adapter-claims", "fingerprint": "c" * 64},
            {"release_id": f"{model}-base", "fingerprint": fingerprint(model)},
        ],
    }
    value.update(changes)
    return value


def answer(questions, **changes):
    value = {
        "release_id": BASE,
        "fingerprint": fingerprint(),
        "limits": {"max_pass_tokens": 8192},
        "predictions": {
            qid: {"probabilities": {"true": 0.25, "false": 0.75}, "input_tokens": 10}
            for qid in questions
        },
    }
    value.update(changes)
    return value


class FakeRuntime:
    """A loopback HTTP server shaped like zils_platform.adapter_server: bearer auth, JSON errors."""

    def __init__(self):
        self.health = health()
        self.reply = None  # (status, body) for the next POSTs; None answers well
        self.requests = []
        runtime = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, body):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def authorized(self):
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    self.send(401, {"error": {"code": "invalid_credentials", "message": "no"}})
                    return False
                return True

            def do_GET(self):
                runtime.requests.append(("GET", self.path, None, dict(self.headers)))
                if self.authorized():
                    self.send(200, runtime.health)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                runtime.requests.append(("POST", self.path, body, dict(self.headers)))
                if not self.authorized():
                    return
                if runtime.reply is not None:
                    return self.send(*runtime.reply)
                self.send(200, answer(body["request"]["questions"]))

        self.handler = Handler


def checkpoint(root, name="reference", kind="base", temperature=1.0):
    path = Path(root) / name
    path.mkdir()
    models.write_metadata(path, kind=kind, model=MODEL, temperature=temperature)
    return path


class RuntimeServedTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.enterContext(patch.dict(os.environ, {models.candidate_base_env(MODEL): ""}))
        self.reference = checkpoint(self.root)
        self.err = self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.runtime = FakeRuntime()
        self.url = self.enterContext(protocol.local_server(self.runtime.handler))

    def connect(self, url=None, token=TOKEN, path=None):
        return serving_runtime.connect(url or self.url, token, path or self.reference, timeout=5.0)

    def posts(self):
        return [r for r in self.runtime.requests if r[0] == "POST"]

    def test_connect_describes_the_reference_and_checks_health(self):
        [served] = self.connect()
        self.assertEqual(
            (served.model, served.weights_sha256, served.max_input_tokens),
            (MODEL, checkpoint_hash(self.reference), 8192),
        )
        [(method, path, _, headers)] = self.runtime.requests
        self.assertEqual(
            (method, path, headers["Authorization"]), ("GET", "/health", f"Bearer {TOKEN}")
        )

    def test_predict_asks_only_the_base_on_the_bulk_lane_and_passes_questions_unchanged(self):
        [served] = self.connect()
        questions = {
            "q1": QUESTION,
            "q2": {"type": "choice", "instructions": "Pick one", "options": {"a": "A", "b": "B"}},
        }
        predictions = served.predict("a state", questions)
        self.assertEqual(predictions, answer(questions)["predictions"])
        [(_, path, body, headers)] = self.posts()
        self.assertEqual(path, "/v1/systemone")
        self.assertEqual(
            body,
            {
                "request": {"model": BASE, "state": "a state", "questions": questions},
                "lane": "bulk",
            },
        )
        self.assertEqual(headers["Authorization"], f"Bearer {TOKEN}")

    def test_predict_never_requests_another_model(self):
        [served] = self.connect()
        served.model = "zils-adapter-claims"  # even a mutated object keeps its pinned release id
        served.predict("s", {"q1": QUESTION})
        self.assertEqual([body["request"]["model"] for _, _, body, _ in self.posts()], [BASE])

    def test_runtime_errors_map_to_worker_error_kinds(self):
        [served] = self.connect()
        for status, code, kind in (
            (413, "context_limit", "too_long"),
            (413, "body_too_large", "too_long"),
            (422, "invalid_request", "invalid"),
            (422, "adapter_input", "invalid"),
            (400, "invalid_framing", "invalid"),
            (529, "overloaded", "busy"),
            (503, "runtime_unavailable", "runtime"),
            (504, "model_timeout", "runtime"),
            (500, "internal_error", "runtime"),
            # The runtime's own refusals of the miner's credential or of the base: the node, not the caller.
            (401, "invalid_credentials", "runtime"),
            (404, "model_not_found", "runtime"),
        ):
            with self.subTest(status=status, code=code):
                self.runtime.reply = (status, {"error": {"code": code, "message": "m"}})
                with self.assertRaises(serving_server.WorkerError) as caught:
                    served.predict("s", {"q1": QUESTION})
                self.assertEqual(caught.exception.kind, kind)

    def test_bad_replies_are_runtime_errors(self):
        [served] = self.connect()
        questions = {"q1": QUESTION, "q2": QUESTION}
        good = answer(questions)
        for reply in (
            b"not json",
            [],
            answer({"q1": QUESTION}),  # a missing question id
            answer({**questions, "q3": QUESTION}),  # an extra question id
            answer(questions, release_id="zils-adapter-claims"),
            answer(questions, fingerprint="0" * 64),
            {k: v for k, v in good.items() if k != "predictions"},
            answer(questions, predictions={"q1": [], "q2": []}),
            answer(
                questions,
                predictions={
                    qid: {"probabilities": {}, "input_tokens": "ten"} for qid in questions
                },
            ),
        ):
            with self.subTest(reply=reply):
                self.runtime.reply = (200, reply)
                with self.assertRaises(serving_server.WorkerError) as caught:
                    served.predict("s", questions)
                self.assertEqual(caught.exception.kind, "runtime")

    def test_connection_refused_is_a_runtime_error_not_a_crash(self):
        [served] = self.connect()
        with socket.socket() as probe:  # a loopback port nothing listens on
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        served.url = f"http://127.0.0.1:{port}"
        with self.assertRaises(serving_server.WorkerError) as caught:
            served.predict("s", {"q1": QUESTION})
        self.assertEqual(caught.exception.kind, "runtime")

    def test_identity_mismatch_refuses_to_start(self):
        other = "f" * 64
        bad_base = [{"release_id": BASE, "fingerprint": other}]
        for changes in (
            {"profile_sha256": other},
            {"runtime_sha256": other},
            {"models": bad_base},
            {"models": [{"release_id": "zils-adapter-claims", "fingerprint": "c" * 64}]},
            {"ready": False},
        ):
            with self.subTest(changes=changes):
                self.runtime.health = health(**changes)
                with self.assertRaises(ValueError):
                    self.connect()
        self.assertEqual(self.posts(), [])

    def test_unreachable_or_unauthorized_runtime_refuses_to_start(self):
        with self.assertRaises(ValueError) as caught:
            self.connect(token="wrong-token-" + "y" * 24)
        self.assertIn("401", str(caught.exception))
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with self.assertRaises(ValueError):
            self.connect(url=f"http://127.0.0.1:{port}")

    def test_reference_temperature_other_than_one_refuses_to_start(self):
        warm = checkpoint(self.root, "warm", temperature=1.3)
        with self.assertRaises(ValueError) as caught:
            self.connect(path=warm)
        self.assertIn("temperature", str(caught.exception))
        self.assertEqual(self.runtime.requests, [])

    def test_only_a_base_checkpoint_can_be_served_from_the_runtime(self):
        merged = checkpoint(self.root, "merged", kind="candidate")
        with self.assertRaises(ValueError) as caught:
            self.connect(path=merged)
        self.assertIn("base", str(caught.exception))
        adapter = checkpoint(self.root, "adapter", kind="adapter")
        with self.assertRaises(ValueError):
            self.connect(path=adapter)
        self.assertEqual(self.runtime.requests, [])

    def test_non_loopback_urls_are_refused(self):
        port = self.url.rsplit(":", 1)[1]
        for url in (
            f"http://10.0.0.5:{port}",
            f"http://8.8.8.8:{port}",
            f"https://127.0.0.1:{port}",
            f"http://127.0.0.1.example.com:{port}",
            f"http://user@127.0.0.1:{port}",
            f"http://127.0.0.1:{port}/v1",
            "http://127.0.0.1",
            "http://127.0.0.1:0",
            "http://127.0.0.1:70000",
            f"http://[::1]:{port}",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.connect(url=url)
        self.assertEqual(self.runtime.requests, [])

    def test_localhost_is_accepted_and_dialled_as_127_0_0_1(self):
        port = self.url.rsplit(":", 1)[1]
        [served] = self.connect(url=f"http://localhost:{port}")
        self.assertEqual(served.url, f"http://127.0.0.1:{port}")

    def test_an_empty_token_is_refused(self):
        with self.assertRaises(ValueError):
            self.connect(token="")

    def test_the_token_never_appears_in_errors_or_stderr(self):
        err, texts = io.StringIO(), []
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(err):
            [served] = self.connect()
            for status in (401, 404, 413, 422, 500, 503, 504, 529):
                self.runtime.reply = (status, {"error": {"code": "x", "message": "m"}})
                with self.assertRaises(serving_server.WorkerError) as caught:
                    served.predict("s", {"q1": QUESTION})
                texts.append(repr(caught.exception) + str(caught.exception))
            for changes in ({"ready": False}, {"profile_sha256": "f" * 64}):
                self.runtime.health = health(**changes)
                with self.assertRaises(ValueError) as caught:
                    self.connect()
                texts.append(str(caught.exception))
            with self.assertRaises(ValueError) as caught:
                self.connect(url="http://10.0.0.5:1")
            texts.append(str(caught.exception))
            self.runtime.health = health()
            self.runtime.reply = None
            stop = threading.Event()
            thread = served.watch(stop, interval=0.01)
            stop.wait(0.2)
            stop.set()
            thread.join(5)
        texts.append(repr(served))
        self.assertNotIn(TOKEN, err.getvalue() + "".join(texts))


class RuntimeWatchTest(unittest.TestCase):
    """The background /health check: pauses are logged, an identity change stops serving."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.enterContext(patch.dict(os.environ, {models.candidate_base_env(MODEL): ""}))
        self.runtime = FakeRuntime()
        self.url = self.enterContext(protocol.local_server(self.runtime.handler))
        [self.served] = serving_runtime.connect(self.url, TOKEN, checkpoint(tmp.name), timeout=5.0)

    def watch(self, seconds=0.3):
        stop, err = threading.Event(), io.StringIO()
        with contextlib.redirect_stderr(err):
            thread = self.served.watch(stop, interval=0.01)
            stopped = stop.wait(seconds)
            stop.set()  # ends the thread before the fake runtime shuts down
            thread.join(5)
        return stopped, err.getvalue()

    def test_a_healthy_runtime_keeps_serving(self):
        stopped, log = self.watch()
        self.assertFalse(stopped)
        self.assertEqual(log, "")

    def test_a_paused_runtime_is_logged_and_keeps_serving(self):
        self.runtime.health = health(ready=False)
        stopped, log = self.watch()
        self.assertFalse(stopped)
        self.assertIn("runtime is not ready", log)

    def test_an_unreachable_runtime_is_logged_and_keeps_serving(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.served.url = f"http://127.0.0.1:{port}"
        stopped, log = self.watch()
        self.assertFalse(stopped)
        self.assertIn("runtime is unreachable", log)

    def test_an_identity_change_stops_serving(self):
        for changes in (
            {"runtime_sha256": "f" * 64},
            {"models": [{"release_id": "zils-adapter-claims", "fingerprint": "c" * 64}]},
        ):
            with self.subTest(changes=changes):
                self.served.changed = None
                self.runtime.health = health(**changes)
                stopped, log = self.watch(5.0)
                self.assertTrue(stopped)
                self.assertIn("identity changed", log)
                self.assertTrue(self.served.changed)


if __name__ == "__main__":
    unittest.main()
