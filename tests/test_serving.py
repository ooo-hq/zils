import importlib.util
import unittest
import uuid

from zils import serving


@unittest.skipUnless(importlib.util.find_spec("bittensor_wallet"), "needs bittensor_wallet")
class ServingProtocolTest(unittest.TestCase):
    def setUp(self):
        from bittensor_wallet import Keypair

        self.miner = Keypair.create_from_seed("0x" + (bytes([1]) * 32).hex())
        self.other = Keypair.create_from_seed("0x" + (bytes([2]) * 32).hex())
        self.served = [{"model": "d3-mini-4.5b-v1", "weights_sha256": "a" * 64, "max_input_tokens": 1024}]

    def test_manifest_round_trip(self):
        body = serving.manifest_body(self.miner.ss58_address, "http://8.8.8.8:8091", self.served, now=1000)
        signed = serving.sign(self.miner, "manifest", body)
        payload = serving.verify(signed, "manifest", self.miner.ss58_address, now=1050)
        self.assertEqual(serving.validate_manifest(payload), self.served)

    def test_manifest_rejects_stale_wrong_signer_and_kind_confusion(self):
        signed = serving.sign(self.miner, "manifest", serving.manifest_body(self.miner.ss58_address, "http://8.8.8.8:8091", self.served, now=1000))
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
        p = serving.request_payload(self.other.ss58_address, self.miner.ss58_address, "d3-mini-4.5b-v1", "a" * 64, {"state": "x", "questions": {}}, now=5)
        q = {**p, "nonce": str(uuid.uuid4())}
        self.assertNotEqual(serving.request_digest(p), serving.request_digest(q))
        self.assertRegex(serving.request_digest(p), "^[a-f0-9]{64}$")

    def test_request_and_answer_round_trip(self):
        p = serving.request_payload(self.other.ss58_address, self.miner.ss58_address, "d3-mini-4.5b-v1", "a" * 64, {"state": "x"}, now=100)
        signed = serving.sign(self.other, "request", p)
        self.assertEqual(serving.verify(signed, "request", self.other.ss58_address, now=200), p)
        with self.assertRaises(ValueError):
            serving.verify(signed, "request", self.other.ss58_address, now=221)
        a = serving.answer_payload(serving.request_digest(p), "d3-mini-4.5b-v1", "a" * 64, {"q": 0.5})
        signed = serving.sign(self.miner, "answer", a)
        self.assertEqual(serving.verify(signed, "answer", self.miner.ss58_address, now=10**9), a)
        with self.assertRaises(ValueError):
            serving.verify(signed, "answer", self.other.ss58_address, now=0)

    def test_manifest_rejects_bad_endpoint_and_version(self):
        for body in (
            serving.manifest_body(self.miner.ss58_address, "http://example.com:80", self.served, now=1),
            {**serving.manifest_body(self.miner.ss58_address, "http://8.8.8.8:8091", self.served, now=1), "version": "x"},
        ):
            with self.assertRaises(ValueError):
                serving.validate_manifest(body)

    def _signed(self):
        body = serving.manifest_body(self.miner.ss58_address, "http://8.8.8.8:8091", self.served, now=1000)
        return serving.sign(self.miner, "manifest", body)

    def _rejects(self, message, kind="manifest", hotkey=None, now=1000):
        with self.assertRaises(ValueError):
            serving.verify(message, kind, hotkey or self.miner.ss58_address, now=now)

    def test_rejects_non_public_endpoints(self):
        for host in ("127.0.0.1", "10.0.0.1", "169.254.1.1", "0.0.0.0", "224.0.0.1"):
            body = serving.manifest_body(self.miner.ss58_address, f"http://{host}:8091", self.served, now=1)
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
        for bad in (sig[:-2], sig + "00", "z" * 128, sig[:10] + " " + sig[10:], sig.upper(), None, 5, bytes.fromhex(sig)):
            self._rejects({**signed, "signature": bad})

    def test_rejects_extra_or_missing_fields(self):
        signed = self._signed()
        self._rejects({**signed, "payload": {**signed["payload"], "extra": 1}})
        self._rejects({**signed, "payload": {k: v for k, v in signed["payload"].items() if k != "models"}})
        self._rejects({**signed, "extra": 1})

    def test_request_caller_must_match_hotkey(self):
        p = serving.request_payload(self.other.ss58_address, self.miner.ss58_address, "d3-mini-4.5b-v1", "a" * 64, {}, now=100)
        signed = serving.sign(self.other, "request", p)
        self._rejects(signed, "request", self.miner.ss58_address, now=100)

    def test_rejects_bad_timestamps(self):
        for stamp in (1000.0, True):
            body = serving.manifest_body(self.miner.ss58_address, "http://8.8.8.8:8091", self.served, now=stamp)
            self._rejects(serving.sign(self.miner, "manifest", body))
        self._rejects(self._signed(), now=879)


if __name__ == "__main__":
    unittest.main()
