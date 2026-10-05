"""Scheduling tests use events, not a GPU or timing-dependent model outputs."""

import importlib
import importlib.util
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor


class RuntimeTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("zils.jev_server"), "private runtime is missing"
        )
        return importlib.import_module("zils.jev_server")

    def test_timeout_does_not_release_running_work(self):
        m = self.module()
        from zils.decisions import DecisionError

        entered, release = threading.Event(), threading.Event()

        class Engine:
            def prepare(self, body):
                return body

            def predict(self, prepared):
                entered.set()
                release.wait(3)
                return prepared

        service = m.SerialEngine(Engine(), timeout=0.04)
        try:
            with self.assertRaises(DecisionError) as err:
                service.evaluate({"first": True})
            self.assertEqual(err.exception.status, 504)
            self.assertTrue(entered.is_set())
            self.assertTrue(service.running)
            release.set()
            self.assertEqual(service.evaluate({"second": True}), {"second": True})
        finally:
            release.set()
            service.close()

    def test_bulk_and_realtime_have_bounded_fair_admission(self):
        m = self.module()
        from zils.decisions import DecisionError

        entered, release = threading.Event(), threading.Event()
        order = []

        class Engine:
            def prepare(self, body):
                return body

            def predict(self, body):
                order.append(body)
                if body == "first":
                    entered.set()
                    release.wait(3)
                return body

        service = m.SerialEngine(Engine(), timeout=2)
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(service.evaluate, "first")
            self.assertTrue(entered.wait(1))
            bulk = pool.submit(service.evaluate, "bulk", "bulk")
            live = pool.submit(service.evaluate, "live")
            deadline = time.monotonic() + 1
            while service.pending_count != 2 and time.monotonic() < deadline:
                time.sleep(0.001)
            with self.assertRaises(DecisionError) as err:
                service.evaluate("excess")
            self.assertEqual(err.exception.status, 529)
            release.set()
            self.assertEqual(first.result(), "first")
            self.assertEqual(live.result(), "live")
            self.assertEqual(bulk.result(), "bulk")
        service.close()
        self.assertEqual(order, ["first", "live", "bulk"])

    def test_invalid_preparation_never_runs_model(self):
        m = self.module()
        from zils.decisions import DecisionError

        class Engine:
            def prepare(self, body):
                raise DecisionError(413, "context_limit", "Too long")

            def predict(self, body):
                raise AssertionError("GPU ran")

        service = m.SerialEngine(Engine())
        try:
            with self.assertRaises(DecisionError) as err:
                service.evaluate({})
            self.assertEqual(err.exception.status, 413)
        finally:
            service.close()


class ModelWrapperTest(unittest.TestCase):
    def module(self):
        import zils.jev_server as m

        self.assertTrue(hasattr(m, "JevEngine"), "JevK5 wrapper is missing")
        return m

    def test_large_choice_token_accounting_and_preflight(self):
        m = self.module()
        from zils.decisions import DecisionError, make_response

        class Model:
            calls = 0

            def encode(self, state, criterion, texts):
                return list(range(10 + len(texts)))

            def letter_logits(self, ids, count):
                self.calls += 1
                return [0.0] * count

        model = Model()
        engine = m.JevEngine(model, max_pass_tokens=65536, max_request_tokens=1000000)
        for count in (2, 17, 50, 255):
            body = {
                "model": "m",
                "state": {},
                "questions": {
                    "q": {"type": "choice", "criteria": {str(i): None for i in range(count)}}
                },
            }
            prepared = engine.prepare(body)
            before = model.calls
            result = engine.predict(prepared)
            response = make_response("m", body, result)
            passes = 1 if count <= 16 else (count + 15) // 16 + 1
            self.assertEqual(model.calls - before, passes)
            self.assertEqual(len(response["answers"]["q"]["probabilities"]), count)
            expected = 10 + count if count <= 16 else 10 * ((count + 15) // 16) + count + 26
            self.assertEqual(response["usage"]["input_tokens"], expected)
            self.assertGreaterEqual(prepared["reserved_tokens"], expected)
        small = m.JevEngine(model, max_pass_tokens=11, max_request_tokens=100)
        before = model.calls
        with self.assertRaises(DecisionError) as err:
            small.prepare({"model": "m", "state": {}, "questions": {"q": {"type": "noul"}}})
        self.assertEqual(err.exception.status, 413)
        self.assertEqual(model.calls, before)

    def test_token_reservation_stops_at_first_over_budget_question(self):
        m = self.module()
        from zils.decisions import DecisionError

        class Model:
            calls = 0

            def encode(self, *args):
                self.calls += 1
                return [1] * 100

        model = Model()
        engine = m.JevEngine(model, max_pass_tokens=1000, max_request_tokens=1000)
        with self.assertRaises(DecisionError):
            engine.prepare(
                {
                    "model": "m",
                    "state": "x",
                    "questions": {str(i): {"type": "noul"} for i in range(5000)},
                }
            )
        self.assertEqual(model.calls, 11)

    def test_release_manifest_detects_file_tampering(self):
        m = self.module()
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}")
            manifest = m.build_manifest(root)
            (root / "release.json").write_text(json.dumps(manifest))
            self.assertEqual(m.verify_manifest(root)["fingerprint"], manifest["fingerprint"])
            (root / "config.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                m.verify_manifest(root)
