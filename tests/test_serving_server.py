import http.client
import importlib.util
import json
import socket
import threading
import unittest

from zils import serving, serving_server

NOW = 1_000_000
ENDPOINT = "http://8.8.8.8:8091"
WEIGHTS = "a" * 64
MODEL = "d3-mini-4.5b-v1"


class FakeServed:
    model = MODEL
    weights_sha256 = WEIGHTS
    max_input_tokens = 100

    def __init__(self, tokens=10, gate=None):
        self.calls, self.tokens, self.gate = 0, tokens, gate
        self.entered = threading.Event()

    def count_tokens(self, state, question):
        return self.tokens

    def predict(self, state, question):
        self.calls += 1
        self.entered.set()
        if self.gate:
            self.gate.wait(5)
        return {"true": 0.25, "false": 0.75}, self.tokens


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

    def start(self, served=None, allowed=lambda hotkey: True, slots=8):
        server = serving_server.make_server(
            self.miner, ENDPOINT, [served or self.fake], allowed, ("127.0.0.1", 0), slots, clock=lambda: NOW
        )
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.servers.append(server)
        return server

    def body(self, **extra):
        question = {"type": "noul", "instructions": None, "criteria": None}
        return {"state": "x", "model": MODEL, "questions": {"q1": question}, **extra}

    def message(self, body=None, **fields):
        payload = serving.request_payload(
            self.caller.ss58_address, self.miner.ss58_address, MODEL, WEIGHTS, body or self.body(), NOW
        )
        payload.update(fields)
        return serving.sign(self.caller, "request", payload)

    def post(self, server, message, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request("POST", "/v1/systemone", raw if raw is not None else json.dumps(message))
        response = conn.getresponse()
        data = json.loads(response.read())
        conn.close()
        return response, data

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
        payload = data["payload"]
        self.assertTrue(self.miner.verify(serving.canonical("answer", payload), bytes.fromhex(data["signature"])))
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
        self.fake.tokens = 101
        self.assertEqual(self.post(server, self.message())[0].status, 413)
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
        server = self.start(slots=1)
        server.RequestHandlerClass.timeout = 0.2
        stalled = socket.create_connection(("127.0.0.1", server.server_address[1]))
        stalled.sendall(b"POST /v1/systemone HTTP/1.1\r\nContent-Length: 100\r\n\r\n")
        stalled.settimeout(5)
        self.assertEqual(stalled.recv(1), b"")  # server closed on us
        stalled.close()
        self.assertEqual(self.post(server, self.message())[0].status, 200)

    def test_missing_content_length_is_refused(self):
        server = self.start()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.putrequest("POST", "/v1/systemone")
        conn.endheaders()
        self.assertEqual(conn.getresponse().status, 411)
