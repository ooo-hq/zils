import contextlib
import http.client
import importlib.util
import io
import json
import socket
import threading
import time
import unittest

from zils import serving, serving_server

NOW = 1_000_000
ENDPOINT = "http://8.8.8.8:8091"
WEIGHTS = "a" * 64
MODEL = "d3-mini-4.5b-v1"


class FakeServed:
    """Stands in for serving_server.Served: `fail` is raised instead of answering."""

    model = MODEL
    weights_sha256 = WEIGHTS
    max_input_tokens = 100

    def __init__(self, gate=None, fail=None):
        self.calls, self.gate, self.fail = 0, gate, fail
        self.entered = threading.Event()

    def predict(self, state, questions):
        self.calls += 1
        self.entered.set()
        if self.gate:
            self.gate.wait(5)
        if self.fail:
            raise self.fail
        return {
            qid: {"probabilities": {"true": 0.25, "false": 0.75}, "input_tokens": 10}
            for qid in questions
        }


@unittest.skipUnless(importlib.util.find_spec("bittensor_wallet"), "needs bittensor_wallet")
class ServingServerTest(unittest.TestCase):
    def setUp(self):
        from bittensor_wallet import Keypair

        self.miner = Keypair.create_from_seed("0x" + (bytes([1]) * 32).hex())
        self.caller = Keypair.create_from_seed("0x" + (bytes([2]) * 32).hex())
        self.fake = FakeServed()
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            server.shutdown()
            server.server_close()

    def start(
        self,
        served=None,
        allowed=lambda hotkey: True,
        slots=8,
        deadline=15.0,
        clock=lambda: NOW,
        **extra,
    ):
        server = serving_server.make_server(
            self.miner,
            ENDPOINT,
            [served or self.fake],
            allowed,
            ("127.0.0.1", 0),
            slots,
            clock=clock,
            deadline=deadline,
            **extra,
        )
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()
        self.servers.append(server)
        return server

    def body(self, **extra):
        question = {"type": "noul", "instructions": None, "criteria": None}
        return {"state": "x", "model": MODEL, "questions": {"q1": question}, **extra}

    def message(self, body=None, now=NOW, **fields):
        payload = serving.request_payload(
            self.caller.ss58_address,
            self.miner.ss58_address,
            MODEL,
            WEIGHTS,
            self.body() if body is None else body,
            now,
        )
        payload.update(fields)
        return serving.sign(self.caller, "request", payload)

    def post(self, server, message, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        try:
            conn.request("POST", "/v1/systemone", raw if raw is not None else json.dumps(message))
            response = conn.getresponse()
            raw = response.read()
        finally:
            conn.close()
        return response, json.loads(raw) if raw else None

    def post_soon(self, server, message):
        """POST, retrying a refusal briefly: a just-closed connection frees its slot a moment later, and a
        connection refused on accept may break the pipe before its 503 is read."""
        for _ in range(100):
            try:
                response, data = self.post(server, message)
                if response.status != 503:
                    return response, data
            except (BrokenPipeError, ConnectionResetError):
                pass
            threading.Event().wait(0.02)
        return self.post(server, message)

    def test_manifest_lists_served_models(self):
        server = self.start()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request("GET", "/v1/serving")
        signed = json.loads(conn.getresponse().read())
        payload = serving.verify(signed, "manifest", self.miner.ss58_address, NOW)
        self.assertEqual(
            serving.validate_manifest(payload),
            [{"model": MODEL, "weights_sha256": WEIGHTS, "max_input_tokens": 100}],
        )

    def test_signed_request_returns_signed_answer_bound_to_request(self):
        server = self.start()
        message = self.message()
        response, data = self.post(server, message)
        self.assertEqual(response.status, 200)
        payload = serving.verify(data, "answer", self.miner.ss58_address, NOW)
        self.assertEqual(payload["miner"], self.miner.ss58_address)
        self.assertEqual(payload["request_sha256"], serving.request_digest(message["payload"]))
        prediction = payload["predictions"]["q1"]
        self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1)
        self.assertEqual(prediction["input_tokens"], 10)
        self.assertEqual(self.fake.calls, 1)

    def test_replayed_request_is_refused_without_inference(self):
        server = self.start()
        message = self.message()
        self.assertEqual(self.post(server, message)[0].status, 200)
        self.assertEqual(self.post(server, message)[0].status, 401)
        self.assertEqual(self.fake.calls, 1)

    def test_unknown_caller_is_forbidden(self):
        server = self.start(allowed=lambda hotkey: False)
        self.assertEqual(self.post(server, self.message())[0].status, 403)
        self.assertEqual(self.fake.calls, 0)

    def test_wrong_audience_or_stale_timestamp_is_refused(self):
        server = self.start()
        other = self.caller.ss58_address
        for message in (self.message(audience=other), self.message(timestamp=NOW - 500)):
            self.assertEqual(self.post(server, message)[0].status, 401)
        for junk in ([], {"payload": []}, {"payload": {"caller": 1}, "signature": "x"}):
            self.assertEqual(self.post(server, junk)[0].status, 401)
        self.assertEqual(self.fake.calls, 0)

    def test_unknown_model_and_wrong_weights(self):
        server = self.start()
        self.assertEqual(self.post(server, self.message(model="nope"))[0].status, 404)
        self.assertEqual(self.post(server, self.message(weights_sha256="b" * 64))[0].status, 409)
        self.assertEqual(self.fake.calls, 0)

    def test_images_and_oversize_are_rejected_before_inference(self):
        server = self.start()
        body = {**self.body(images=["pic"]), "state": "SECRET-STATE"}
        response, data = self.post(server, self.message(body))
        self.assertEqual(response.status, 422)
        self.assertNotIn("SECRET", json.dumps(data))
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.putrequest("POST", "/v1/systemone")
        conn.putheader("Content-Length", str(1024 * 1024 + 1))
        conn.endheaders()  # refused on the header alone, body never sent
        self.assertEqual(conn.getresponse().status, 413)
        self.assertEqual(self.fake.calls, 0)

    def test_overload_returns_503_retry_after(self):
        gate = threading.Event()
        fake = FakeServed(gate=gate)
        server = self.start(fake, slots=1)
        first = threading.Thread(target=self.post, args=(server, self.message()))
        first.start()
        self.assertTrue(fake.entered.wait(5))
        sock = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5)
        sock.settimeout(5)
        reply = sock.recv(4096)  # refused on accept: the server speaks first, before reading
        sock.close()
        self.assertIn(b"503", reply.split(b"\r\n")[0])
        self.assertIn(b"Retry-After: 1", reply)
        gate.set()
        first.join(5)

    def test_stalled_client_is_dropped_and_frees_the_slot(self):
        server = self.start(slots=1, header_deadline=15.0)
        server.RequestHandlerClass.timeout = 0.2
        stalled = socket.create_connection(("127.0.0.1", server.server_address[1]))
        stalled.sendall(b"POST /v1/systemone HTTP/1.1\r\nContent-Length: 100\r\n\r\n")
        stalled.settimeout(5)
        self.assertEqual(stalled.recv(1), b"")  # server closed on us
        stalled.close()
        self.assertEqual(self.post_soon(server, self.message())[0].status, 200)

    def test_missing_content_length_is_refused(self):
        server = self.start()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.putrequest("POST", "/v1/systemone")
        conn.endheaders()
        self.assertEqual(conn.getresponse().status, 411)

    def test_trickling_client_hits_total_deadline_and_frees_the_slot(self):
        server = self.start(slots=1, deadline=0.5)
        sock = socket.create_connection(("127.0.0.1", server.server_address[1]))
        sock.settimeout(5)
        try:
            for _ in range(10):  # one byte every 0.2 s would outlast any per-read timeout
                sock.sendall(b"P")
                threading.Event().wait(0.2)
        except OSError:
            pass
        try:
            self.assertEqual(sock.recv(1), b"")
        except ConnectionResetError:
            pass  # dropped with unread bytes: also a drop
        sock.close()
        self.assertEqual(self.post_soon(server, self.message())[0].status, 200)

    def test_encoding_error_and_malformed_types_are_clean_errors(self):
        bad = FakeServed(fail=serving_server.WorkerError("invalid"))
        server = self.start(bad, slots=1)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            response, data = self.post(server, self.message())
            self.assertEqual(response.status, 422)
            self.assertNotIn("SECRET", json.dumps(data))
            for field in ({"model": ["x"]}, {"weights_sha256": {}}, {"body": []}):
                self.assertEqual(self.post(server, self.message(**field))[0].status, 400)
            self.assertEqual(
                self.post(server, self.message(self.body() | {"model": "other"}))[0].status, 400
            )
            bad.fail = None  # lock and slot were released: it serves again
            self.assertEqual(self.post_soon(server, self.message())[0].status, 200)
        self.assertEqual(err.getvalue(), "")

    def test_body_model_must_match_signed_model(self):
        server = self.start()
        response, data = self.post(server, self.message(self.body() | {"model": "other"}))
        self.assertEqual((response.status, data["error"]["code"]), (400, "model_mismatch"))
        self.assertEqual(self.fake.calls, 0)

    def test_worker_refusals_map_to_http(self):
        for failure, status, code in (
            (serving_server.WorkerError("too_long"), 413, "input_too_long"),
            (serving_server.WorkerError("invalid"), 422, "invalid_request"),
        ):
            server = self.start(FakeServed(fail=failure))
            response, data = self.post(server, self.message())
            self.assertEqual((response.status, data["error"]["code"]), (status, code))

    def test_runtime_failure_and_lost_worker_are_503_and_logged(self):
        for failure, logged in (
            (serving_server.WorkerError("runtime"), "worker reported runtime"),
            (OSError("serving worker exited"), "OSError: serving worker exited"),
        ):
            server = self.start(FakeServed(fail=failure))
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                response, data = self.post(
                    server, self.message(self.body() | {"state": "SECRET-STATE"})
                )
            self.assertEqual((response.status, data["error"]["code"]), (503, "unavailable"))
            self.assertIn(logged, err.getvalue())
            self.assertNotIn("SECRET", err.getvalue() + json.dumps(data))

    def test_unexpected_error_is_500_and_logged(self):
        server = self.start(FakeServed(fail=KeyError("SECRET-STATE")))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            response, data = self.post(server, self.message())
        self.assertEqual(response.status, 500)
        self.assertEqual(err.getvalue(), "zils serve: internal error (KeyError)\n")

    def test_too_many_questions_is_refused_before_inference(self):
        question = {"type": "noul", "instructions": None, "criteria": None}
        server = self.start()
        many = self.body() | {
            "questions": {f"q{i}": question for i in range(serving_server.MAX_QUESTIONS + 1)}
        }
        response, data = self.post(server, self.message(many))
        self.assertEqual((response.status, data["error"]["code"]), (413, "too_many_questions"))
        self.assertEqual(self.fake.calls, 0)
        exact = self.body() | {
            "questions": {f"q{i}": question for i in range(serving_server.MAX_QUESTIONS)}
        }
        self.assertEqual(self.post(server, self.message(exact))[0].status, 200)

    def test_non_ascii_digit_content_length_is_refused(self):
        server = self.start()
        sock = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5)
        sock.sendall("POST /v1/systemone HTTP/1.0\r\nContent-Length: \u00b2\r\n\r\n".encode())
        reply = sock.recv(4096)
        sock.close()
        self.assertIn(b" 411 ", reply.split(b"\r\n")[0])

    def test_one_address_holds_at_most_two_connections(self):
        server = self.start(slots=8)
        idle = [
            socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5)
            for _ in range(2)
        ]
        third = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5)
        reply = third.recv(4096)  # refused on accept, before any bytes are read
        third.close()
        self.assertIn(b"503", reply.split(b"\r\n")[0])
        self.assertIn(b"Retry-After: 1", reply)
        for sock in idle:
            sock.close()
        self.assertEqual(self.post_soon(server, self.message())[0].status, 200)

    def test_idle_connection_is_dropped_at_the_header_deadline(self):
        server = self.start(header_deadline=0.2)  # total deadline stays 15 s, per-read 10 s
        sock = socket.create_connection(("127.0.0.1", server.server_address[1]))
        sock.sendall(b"POST /v1/sys")
        sock.settimeout(5)
        try:
            self.assertEqual(sock.recv(1), b"")
        except ConnectionResetError:
            pass
        sock.close()

    def test_trickled_headers_are_dropped_at_the_header_deadline(self):
        server = self.start(header_deadline=0.3)  # total deadline stays 15 s
        sock = socket.create_connection(("127.0.0.1", server.server_address[1]))
        sock.sendall(b"POST /v1/systemone HTTP/1.0\r\n")
        started = time.monotonic()
        try:
            for _ in range(20):  # one header line every 0.2 s would last 4 s
                sock.sendall(b"X-Slow: 1\r\n")
                time.sleep(0.2)
        except OSError:
            pass
        self.assertLess(time.monotonic() - started, 2.0)
        sock.close()

    def test_old_nonces_are_pruned_and_fresh_replays_still_refused(self):
        now = [NOW]
        server = self.start(clock=lambda: now[0])
        first = self.message()
        self.assertEqual(self.post(server, first)[0].status, 200)
        now[0] = NOW + 100
        self.assertEqual(self.post(server, first)[0].status, 401)  # replay inside the window
        now[0] = NOW + 2 * serving.FRESH_SECONDS + 1
        later = self.message(now=now[0])
        self.assertEqual(self.post(server, later)[0].status, 200)
        self.assertEqual(
            list(server.nonces), [(self.caller.ss58_address, later["payload"]["nonce"])]
        )
        self.assertEqual(self.post(server, later)[0].status, 401)

    def test_two_served_checkpoints_of_one_model_are_refused(self):
        with self.assertRaises(ValueError):
            serving_server.make_server(
                self.miner, ENDPOINT, [FakeServed(), FakeServed()], lambda h: True, ("127.0.0.1", 0)
            )
