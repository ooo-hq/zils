import contextlib
import importlib.util
import io
import itertools
import json
import socket
import threading
import time
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from bittensor_wallet import Keypair

from zils import serving, serving_server

SHA = "ab" * 32
MODEL = "d3-mini-4.5b-v1"
MINER_KEY = Keypair.create_from_seed("0x" + "11" * 32)
CALLER_KEY = Keypair.create_from_seed("0x" + "22" * 32)
OTHER_KEY = Keypair.create_from_seed("0x" + "33" * 32)
HOTKEY = MINER_KEY.ss58_address


@unittest.skipUnless(importlib.util.find_spec("bittensor_wallet"), "needs bittensor_wallet")
class ServingProtocolTest(unittest.TestCase):
    def setUp(self):
        from bittensor_wallet import Keypair

        self.miner = Keypair.create_from_seed("0x" + (bytes([1]) * 32).hex())
        self.other = Keypair.create_from_seed("0x" + (bytes([2]) * 32).hex())
        self.served = [
            {"model": "d3-mini-4.5b-v1", "weights_sha256": "a" * 64, "max_input_tokens": 1024}
        ]

    def test_manifest_round_trip(self):
        body = serving.manifest_body(
            self.miner.ss58_address, "http://8.8.8.8:8091", self.served, now=1000
        )
        signed = serving.sign(self.miner, "manifest", body)
        payload = serving.verify(signed, "manifest", self.miner.ss58_address, now=1050)
        self.assertEqual(serving.validate_manifest(payload), self.served)

    def test_manifest_rejects_stale_wrong_signer_and_kind_confusion(self):
        signed = serving.sign(
            self.miner,
            "manifest",
            serving.manifest_body(
                self.miner.ss58_address, "http://8.8.8.8:8091", self.served, now=1000
            ),
        )
        with self.assertRaises(ValueError):
            serving.verify(signed, "manifest", self.miner.ss58_address, now=1121)
        with self.assertRaises(ValueError):
            serving.verify(signed, "manifest", self.other.ss58_address, now=1000)
        with self.assertRaises(ValueError):
            serving.verify(signed, "request", self.miner.ss58_address, now=1000)

    def test_manifest_rejects_adapter_free_violations(self):
        for bad in (
            [{"model": "not-a-model", "weights_sha256": "a" * 64, "max_input_tokens": 1}],
            self.served * 2,
            [{**self.served[0], "max_input_tokens": 0}],
            [{**self.served[0], "weights_sha256": "xyz"}],
            [],
        ):
            body = serving.manifest_body(self.miner.ss58_address, "http://8.8.8.8:8091", bad, now=1)
            with self.assertRaises(ValueError):
                serving.validate_manifest(body)

    def test_request_digest_binds_answer(self):
        p = serving.request_payload(
            self.other.ss58_address,
            self.miner.ss58_address,
            "d3-mini-4.5b-v1",
            "a" * 64,
            {"state": "x", "questions": {}},
            now=5,
        )
        q = {**p, "nonce": str(uuid.uuid4())}
        self.assertNotEqual(serving.request_digest(p), serving.request_digest(q))
        self.assertRegex(serving.request_digest(p), "^[a-f0-9]{64}$")

    def test_request_and_answer_round_trip(self):
        p = serving.request_payload(
            self.other.ss58_address,
            self.miner.ss58_address,
            "d3-mini-4.5b-v1",
            "a" * 64,
            {"state": "x"},
            now=100,
        )
        signed = serving.sign(self.other, "request", p)
        self.assertEqual(serving.verify(signed, "request", self.other.ss58_address, now=200), p)
        with self.assertRaises(ValueError):
            serving.verify(signed, "request", self.other.ss58_address, now=221)
        a = serving.answer_payload(
            self.miner.ss58_address,
            serving.request_digest(p),
            "d3-mini-4.5b-v1",
            "a" * 64,
            {"q": 0.5},
        )
        signed = serving.sign(self.miner, "answer", a)
        self.assertEqual(serving.verify(signed, "answer", self.miner.ss58_address, now=10**9), a)
        with self.assertRaises(ValueError):
            serving.verify(signed, "answer", self.other.ss58_address, now=0)

    def test_answer_names_its_miner(self):
        # A miner's valid signature over an answer naming another miner is refused.
        a = serving.answer_payload(
            self.other.ss58_address, "b" * 64, "d3-mini-4.5b-v1", "a" * 64, {}
        )
        self._rejects(serving.sign(self.miner, "answer", a), "answer", self.miner.ss58_address)

    def test_manifest_rejects_bad_endpoint_and_version(self):
        for body in (
            serving.manifest_body(
                self.miner.ss58_address, "http://example.com:80", self.served, now=1
            ),
            {
                **serving.manifest_body(
                    self.miner.ss58_address, "http://8.8.8.8:8091", self.served, now=1
                ),
                "version": "x",
            },
        ):
            with self.assertRaises(ValueError):
                serving.validate_manifest(body)

    def _signed(self):
        body = serving.manifest_body(
            self.miner.ss58_address, "http://8.8.8.8:8091", self.served, now=1000
        )
        return serving.sign(self.miner, "manifest", body)

    def _rejects(self, message, kind="manifest", hotkey=None, now=1000):
        with self.assertRaises(ValueError):
            serving.verify(message, kind, hotkey or self.miner.ss58_address, now=now)

    def test_rejects_non_public_endpoints(self):
        for host in ("127.0.0.1", "10.0.0.1", "169.254.1.1", "0.0.0.0", "224.0.0.1"):
            body = serving.manifest_body(
                self.miner.ss58_address, f"http://{host}:8091", self.served, now=1
            )
            with self.assertRaises(ValueError, msg=host):
                serving.validate_manifest(body)

    def test_rejects_tampered_payload(self):
        signed = self._signed()
        self._rejects({**signed, "payload": {**signed["payload"], "issued_at": 1001}})
        models = [{**self.served[0], "weights_sha256": "b" * 64}]
        self._rejects({**signed, "payload": {**signed["payload"], "models": models}})

    def test_rejects_malformed_signatures(self):
        signed = self._signed()
        sig = signed["signature"]
        for bad in (
            sig[:-2],
            sig + "00",
            "z" * 128,
            sig[:10] + " " + sig[10:],
            sig.upper(),
            None,
            5,
            bytes.fromhex(sig),
        ):
            self._rejects({**signed, "signature": bad})

    def test_rejects_extra_or_missing_fields(self):
        signed = self._signed()
        self._rejects({**signed, "payload": {**signed["payload"], "extra": 1}})
        self._rejects(
            {**signed, "payload": {k: v for k, v in signed["payload"].items() if k != "models"}}
        )
        self._rejects({**signed, "extra": 1})

    def test_request_caller_must_match_hotkey(self):
        p = serving.request_payload(
            self.other.ss58_address,
            self.miner.ss58_address,
            "d3-mini-4.5b-v1",
            "a" * 64,
            {},
            now=100,
        )
        signed = serving.sign(self.other, "request", p)
        self._rejects(signed, "request", self.miner.ss58_address, now=100)

    def test_rejects_bad_timestamps(self):
        for stamp in (1000.0, True):
            body = serving.manifest_body(
                self.miner.ss58_address, "http://8.8.8.8:8091", self.served, now=stamp
            )
            self._rejects(serving.sign(self.miner, "manifest", body))
        self._rejects(self._signed(), now=879)


@unittest.skipUnless(importlib.util.find_spec("bittensor_wallet"), "needs bittensor_wallet")
class ServingDiscoveryTest(unittest.TestCase):
    def setUp(self):
        from bittensor_wallet import Keypair

        self.miner = Keypair.create_from_seed("0x" + (bytes([1]) * 32).hex())
        self.other = Keypair.create_from_seed("0x" + (bytes([2]) * 32).hex())
        self.served = [
            {"model": "d3-mini-4.5b-v1", "weights_sha256": "a" * 64, "max_input_tokens": 1024}
        ]

    def _manifest(self, endpoint):
        body = serving.manifest_body(self.miner.ss58_address, endpoint, self.served, now=100)
        return serving.sign(self.miner, "manifest", body)

    def test_discover_accepts_matching_manifest(self):
        ep = "http://8.8.8.8:8091"
        out = serving.discover(
            {3: (self.miner.ss58_address, ep)},
            clock=lambda: 100,
            fetch=lambda e: self._manifest(ep),
        )
        self.assertEqual(
            out[3], {"hotkey": self.miner.ss58_address, "endpoint": ep, "models": self.served}
        )

    def test_discover_rejects_copied_manifest_and_wrong_signer(self):
        ep_a, ep_b = "http://8.8.8.8:8091", "http://9.9.9.9:8091"
        m = self._manifest(ep_a)
        out = serving.discover(
            {3: (self.miner.ss58_address, ep_b), 4: (self.other.ss58_address, ep_a)},
            clock=lambda: 100,
            fetch=lambda e: m,
        )
        self.assertIn("error", out[3])
        self.assertIn("error", out[4])

    def test_discover_turns_fetch_failure_into_error(self):
        def boom(_):
            raise OSError("refused")

        out = serving.discover(
            {3: (self.miner.ss58_address, "http://8.8.8.8:8091")}, clock=lambda: 1, fetch=boom
        )
        self.assertIn("error", out[3])

    def test_discover_never_fetches_non_public_endpoints(self):
        calls = []
        axons = {
            i: (self.miner.ss58_address, ep)
            for i, ep in enumerate(
                ["http://127.0.0.1:8091", "http://10.0.0.5:8091", "http://169.254.169.254:80"]
            )
        }
        out = serving.discover(axons, clock=lambda: 1, fetch=calls.append)
        self.assertEqual(calls, [])
        self.assertTrue(all("error" in v for v in out.values()))
        with self.assertRaises(ValueError):
            serving.public_endpoint("http://192.168.1.1:8091")
        self.assertEqual(serving.public_endpoint("http://8.8.8.8:8091"), "http://8.8.8.8:8091")

    def test_discover_reads_the_clock_per_miner(self):
        # A slow scan: each fetch takes 100 s. One `now` for the whole scan would age out the last
        # manifests; the clock is read after each fetch, so every fresh manifest passes.
        clock = [100]
        endpoints = ["http://8.8.8.%d:8091" % i for i in range(1, 6)]

        def slow_fetch(endpoint):
            clock[0] += 100
            body = serving.manifest_body(
                self.miner.ss58_address, endpoint, self.served, now=clock[0]
            )
            return serving.sign(self.miner, "manifest", body)

        axons = {i: (self.miner.ss58_address, ep) for i, ep in enumerate(endpoints)}
        out = serving.discover(axons, clock=lambda: clock[0], fetch=slow_fetch)
        self.assertEqual([v.get("endpoint") for v in out.values()], endpoints)

    def test_fetch_manifest_refuses_non_public_endpoints_itself(self):
        for endpoint in (
            "http://127.0.0.1:8091",
            "http://10.0.0.5:8091",
            "http://169.254.169.254:80",
        ):
            with self.assertRaisesRegex(ValueError, "public IPv4"):
                serving.fetch_manifest(endpoint)

    def test_validate_manifest_missing_endpoint_is_value_error(self):
        with self.assertRaises(ValueError):
            serving.validate_manifest({"version": serving.VERSION, "models": self.served})

    def test_fetch_manifest_enforces_total_deadline_against_drip_feed(self):
        import time
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from threading import Thread

        class Drip(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                try:
                    for _ in range(100):
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(0.2)
                except OSError:
                    pass

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), Drip)
        Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        started = time.monotonic()
        with self.assertRaises(ValueError):
            serving._fetch_manifest("http://127.0.0.1:%d" % server.server_port, timeout=0.5)
        self.assertLess(time.monotonic() - started, 2)

    def test_fetch_manifest_enforces_total_deadline_against_header_drip(self):
        import time
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from threading import Thread

        class HeaderDrip(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    self.wfile.write(b"HTTP/1.0 200 OK\r\n")
                    for _ in range(100):
                        self.wfile.write(b"X")
                        self.wfile.flush()
                        time.sleep(0.2)
                except OSError:
                    pass

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), HeaderDrip)
        Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        started = time.monotonic()
        with self.assertRaises(ValueError):
            serving._fetch_manifest("http://127.0.0.1:%d" % server.server_port, timeout=0.5)
        self.assertLess(time.monotonic() - started, 2)

    def test_fetch_manifest_reads_json_and_caps_size(self):
        import json
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from threading import Thread

        class Handler(BaseHTTPRequestHandler):
            big = False

            def do_GET(self):
                body = json.dumps(
                    {"x": "y" * serving.MAX_MANIFEST_BYTES} if self.big else {"ok": 1}
                ).encode()
                if self.path == "/v1/serving":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(body)

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        ep = "http://127.0.0.1:%d" % server.server_port
        self.assertEqual(serving._fetch_manifest(ep, 5.0), {"ok": 1})
        Handler.big = True
        with self.assertRaisesRegex(ValueError, "too large"):
            serving._fetch_manifest(ep, 5.0)


BODY = {
    "state": "A ticket",
    "model": "zils-shared-r1",
    "questions": {
        "q1": {"type": "noul"},
        "q2": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
    },
}
GOOD = {
    "q1": {"probabilities": {"true": 0.75, "false": 0.25}, "input_tokens": 7},
    "q2": {"probabilities": {"a": 0.5, "b": 0.5}, "input_tokens": 7},
}


def allow_all(endpoint):
    return endpoint


class FakeMiner:
    """Signs answers for the pinned serving protocol; `opts` bends one thing at a time."""

    def __init__(self, **opts):
        opts_ = opts

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payload = request["payload"]
                if opts_.get("reset"):
                    self.close_connection = True
                    return
                if opts_.get("drip"):
                    try:
                        for byte in b"HTTP/1.0 200 OK\r\nX: " + b"y" * 50:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                            time.sleep(0.05)
                    except OSError:
                        pass
                    return
                if "drip_body" in opts_:
                    self.send_response(opts_["status"])
                    self.send_header("Content-Length", "100")
                    self.end_headers()
                    try:
                        for _ in range(100):
                            self.wfile.write(b" ")
                            self.wfile.flush()
                            time.sleep(0.05)
                    except OSError:
                        pass
                    return
                if "status" in opts_:
                    return self.reply(opts_["status"], opts_.get("body", b"{}"))
                if "oversize" in opts_:
                    return self.reply(200, b" " * (1 << 20) + b"{}")
                predictions = dict(opts_.get("predictions", GOOD))
                if opts_.get("missing"):
                    predictions.pop("q2")
                digest = opts_.get("digest") or serving.request_digest(payload)
                answer = serving.answer_payload(
                    MINER_KEY.ss58_address,
                    digest,
                    opts_.get("model", payload["model"]),
                    opts_.get("weights", payload["weights_sha256"]),
                    predictions,
                )
                signed = serving.sign(opts_.get("signer", MINER_KEY), "answer", answer)
                self.reply(200, json.dumps(signed).encode())

            def reply(self, status, data):
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except OSError:
                    pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass

        self.server = Server(("127.0.0.1", 0), Handler)
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()
        self.miner = "http://127.0.0.1:%d" % self.server.server_port

    def __enter__(self):
        return self.miner

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def call(endpoint, timeout=2.0):
    return serving.ask(endpoint, HOTKEY, CALLER_KEY, MODEL, SHA, BODY, timeout, check=allow_all)


class AskTests(unittest.TestCase):
    def test_ask_returns_verified_predictions_and_signed_message(self):
        with FakeMiner() as miner:
            message = call(miner)
        self.assertEqual(message["payload"]["predictions"], GOOD)
        self.assertEqual(message["payload"]["miner"], MINER_KEY.ss58_address)
        self.assertIn("signature", message)
        serving.verify(
            message, "answer", MINER_KEY.ss58_address, int(time.time())
        )  # still signed as sent

    def test_ask_rejects_wrong_signer_digest_model_weights_and_missing_question(self):
        bad_probs = {
            **GOOD,
            "q1": {"probabilities": {"true": 0.9, "false": 0.9}, "input_tokens": 7},
        }
        # sums to 1 within 1e-6 but p > 1: make_response would answer 502
        above_one = {
            **GOOD,
            "q1": {"probabilities": {"true": 1.0000005, "false": 0.0}, "input_tokens": 7},
        }
        cases = {
            "signer": ({"signer": OTHER_KEY}, "unverified"),
            "digest": ({"digest": "0" * 64}, "invalid"),
            "model": ({"model": "other"}, "invalid"),
            "weights": ({"weights": "cd" * 32}, "invalid"),
            "missing": ({"missing": True}, "invalid"),
            "sum": ({"predictions": bad_probs}, "invalid"),
            "above_one": ({"predictions": above_one}, "invalid"),
        }
        for name, (opts, reason) in cases.items():
            with self.subTest(name), FakeMiner(**opts) as miner:
                with self.assertRaises(serving.MinerError) as caught:
                    call(miner)
                self.assertEqual(caught.exception.reason, reason)

    def test_ask_times_out_on_dripping_headers_within_budget(self):
        with FakeMiner(drip=True) as miner:
            start = time.monotonic()
            with self.assertRaises(serving.MinerError) as caught:
                call(miner, timeout=0.5)
            self.assertEqual(caught.exception.reason, "timeout")
            self.assertLess(time.monotonic() - start, 2.0)

    def test_ask_rejects_non_200_and_oversize(self):
        for name, opts in {"500": {"status": 500}, "oversize": {"oversize": True}}.items():
            with self.subTest(name), FakeMiner(**opts) as miner:
                with self.assertRaises(serving.MinerError) as caught:
                    call(miner)
                self.assertEqual(caught.exception.reason, "http" if name == "500" else "invalid")

    def test_ask_classifies_http_status(self):
        cases = {503: "busy", 429: "busy", 401: "refused", 403: "refused", 413: "rejected"}
        cases.update({404: "rejected", 409: "rejected", 502: "http"})
        for status, reason in cases.items():
            with self.subTest(status), FakeMiner(status=status) as miner:
                with self.assertRaises(serving.MinerError) as caught:
                    call(miner)
                self.assertEqual(caught.exception.reason, reason)

    def test_ask_classifies_error_codes_in_json_bodies(self):
        def err(code):
            return json.dumps({"error": {"code": code}}).encode()

        cases = [
            (503, err("unavailable"), "unavailable"),
            (503, err("other"), "busy"),
            (503, b"", "busy"),
            (503, b"not json", "busy"),
            (429, err("unavailable"), "busy"),
            (404, err("model_not_found"), "contradicts_manifest"),
            (409, err("weights_mismatch"), "contradicts_manifest"),
            (404, err("other"), "rejected"),
            (409, b"{}", "rejected"),
            (404, b"", "rejected"),
            (409, err("model_not_found"), "rejected"),
        ]
        for status, body, reason in cases:
            with self.subTest(status=status, body=body), FakeMiner(status=status, body=body) as m:
                with self.assertRaises(serving.MinerError) as caught:
                    call(m)
                self.assertEqual(caught.exception.reason, reason)

    def test_dripped_error_body_is_a_timeout(self):
        for status in (503, 429):
            with self.subTest(status), FakeMiner(status=status, drip_body=True) as miner:
                start = time.monotonic()
                with self.assertRaises(serving.MinerError) as caught:
                    call(miner, timeout=0.5)
                self.assertEqual(caught.exception.reason, "timeout")
                self.assertLess(time.monotonic() - start, 1.0)

    def test_connection_error_past_deadline_is_a_timeout(self):
        with FakeMiner(reset=True) as miner:
            with self.assertRaises(serving.MinerError) as caught:
                call(miner)
            self.assertEqual(caught.exception.reason, "http")
            clock = itertools.chain([0.0], itertools.repeat(1000.0))
            with (
                patch("time.monotonic", lambda: next(clock)),
                self.assertRaises(serving.MinerError) as late,
            ):
                call(miner)
            self.assertEqual(late.exception.reason, "timeout")

    def test_ask_refuses_non_public_endpoint_by_default(self):
        with FakeMiner() as miner:
            with self.assertRaises(serving.MinerError):
                serving.ask(miner, HOTKEY, CALLER_KEY, MODEL, SHA, BODY, 1.0)


class BrokenServed:
    model, weights_sha256, max_input_tokens = MODEL, SHA, 8192

    def predict(self, state, questions):
        raise OSError("worker gone")


class StubServed:
    """The served-object interface `zils.serving_server.make_server` uses."""

    model, weights_sha256, max_input_tokens = MODEL, SHA, 8192

    def __init__(self):
        self.calls = 0

    def predict(self, state, questions):
        self.calls += 1
        return {qid: GOOD[qid] for qid in questions}


class PinnedServerTests(unittest.TestCase):
    """The gateway against the pinned miner server, not a fake."""

    def setUp(self):
        self.served = StubServed()
        self.server = serving_server.make_server(
            MINER_KEY,
            "http://127.0.0.1:1",
            [self.served],
            allowed=lambda caller: caller == CALLER_KEY.ss58_address,
            bind=("127.0.0.1", 0),
            slots=1,
        )
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()
        self.miner = "http://127.0.0.1:%d" % self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_good_answer(self):
        message = call(self.miner)
        self.assertEqual(message["payload"]["predictions"], GOOD)
        self.assertEqual(message["payload"]["miner"], MINER_KEY.ss58_address)

    def test_unadmitted_caller_is_refused(self):
        with self.assertRaises(serving.MinerError) as caught:
            serving.ask(self.miner, HOTKEY, OTHER_KEY, MODEL, SHA, BODY, 2.0, check=allow_all)
        self.assertEqual(caught.exception.reason, "refused")

    def test_busy_miner_is_busy(self):
        holder = socket.create_connection(("127.0.0.1", self.server.server_port))
        try:
            deadline = time.monotonic() + 2
            while not self.server.peers and time.monotonic() < deadline:
                time.sleep(0.005)  # until the server has given the only slot to `holder`
            with self.assertRaises(serving.MinerError) as caught:
                call(self.miner)
        finally:
            holder.close()
        self.assertEqual(caught.exception.reason, "busy")
        self.assertEqual(self.served.calls, 0)

    def test_crashed_worker_is_unavailable(self):
        broken = serving_server.make_server(
            MINER_KEY,
            "http://127.0.0.1:1",
            [BrokenServed()],
            allowed=lambda caller: True,
            bind=("127.0.0.1", 0),
        )
        threading.Thread(
            target=broken.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()
        self.addCleanup(broken.server_close)
        self.addCleanup(broken.shutdown)
        endpoint = "http://127.0.0.1:%d" % broken.server_port
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(serving.MinerError) as caught,
        ):
            call(endpoint)
        self.assertEqual(caught.exception.reason, "unavailable")

    def test_unserved_model_and_wrong_weights_contradict_the_manifest(self):
        for model, sha in (("other-model", SHA), (MODEL, "cd" * 32)):
            with self.assertRaises(serving.MinerError) as caught:
                serving.ask(self.miner, HOTKEY, CALLER_KEY, model, sha, BODY, 2.0, check=allow_all)
            self.assertEqual(caught.exception.reason, "contradicts_manifest")


if __name__ == "__main__":
    unittest.main()
