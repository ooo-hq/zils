"""The private image runtime verifies content before serialized model execution."""

import copy
import hashlib
import importlib
import importlib.util
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler

from tests.test_image_assets import png
from tests.test_image_contract import BODY
from tests.test_queue import server
from zils.decisions import DecisionError
from zils.image_assets import canonicalize


class Engine:
    def __init__(self):
        self.executions = 0
        self.active = 0
        self.peak = 0
        self.tokens = 442
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False

    def activate_release(self, path, release):
        pass

    def prepare(self, image, state, question):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.set()
        if self.block:
            self.release.wait(2)
        self.active -= 1
        return {"input_tokens": self.tokens}

    def billable_input(self, request, prepared):
        return 173

    def predict(self, prepared, temperature=1.0):
        self.executions += 1
        return {
            "probabilities": {"normal": 0.8, "damaged": 0.1, "__unknown__": 0.1},
            "input_tokens": prepared["input_tokens"],
        }


class ImageRuntimeTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("zils.image_server"), "image runtime missing")
        return importlib.import_module("zils.image_server")

    def setUp(self):
        self.image = canonicalize(png())
        raw = self.image.data

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        self.context = server(Handler)
        self.origin = self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        self.engine = Engine()
        self.envelope = {
            "request": copy.deepcopy(BODY),
            "fingerprint": "a" * 64,
            "image": {
                "id": BODY["images"][0]["asset_id"],
                "sha256": hashlib.sha256(raw).hexdigest(),
                "url": self.origin
                + "/storage/v1/object/sign/zils-images/11111111-1111-4111-8111-111111111111/"
                + BODY["images"][0]["asset_id"]
                + "/canonical.png?token=fixture",
                "width": 8,
                "height": 8,
                "bytes": len(raw),
                "preprocessor": "zils-image-rgb-png/v1",
                "expires_at": "2099-01-01T00:00:00+00:00",
            },
        }

    def runtime(self, timeout=2):
        rt = self.module().ImageRuntime(
            self.engine,
            {"image-release": {"fingerprint": "a" * 64, "temperature": 1.0}},
            self.origin,
            token="secret",
            timeout=timeout,
        )
        self.addCleanup(rt.close)
        return rt

    def test_auth_content_origin_and_release_are_verified_before_execution(self):
        rt = self.runtime()
        for name, change in (
            ("hash", lambda b: b["image"].update(sha256="0" * 64)),
            ("origin", lambda b: b["image"].update(url="https://elsewhere.example/photo")),
            ("path", lambda b: b["image"].update(url=self.origin + "/untrusted")),
            ("release", lambda b: b.update(fingerprint="b" * 64)),
            ("expiry", lambda b: b["image"].update(expires_at="2000-01-01T00:00:00+00:00")),
        ):
            body = copy.deepcopy(self.envelope)
            change(body)
            with self.subTest(name=name), self.assertRaises(DecisionError):
                rt.dispatch("POST", "/v1/systemone", "secret", body, "test")
        with self.assertRaises(DecisionError) as ctx:
            rt.dispatch("POST", "/v1/systemone", "wrong", self.envelope, "test")
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(self.engine.executions, 0)

    def test_actual_processor_tokens_and_native_distribution(self):
        rt = self.runtime()
        status, prep = rt.dispatch("POST", "/v1/prepare", "secret", self.envelope, "prepare")
        self.assertEqual((status, prep["reserved_tokens"]), (200, 442))
        self.assertEqual(prep.get("billable_tokens"), 173)
        _, result = rt.dispatch("POST", "/v1/systemone", "secret", self.envelope, "predict")
        self.assertEqual(result["predictions"]["inspection"]["input_tokens"], 442)
        self.assertEqual(result["predictions"]["inspection"]["probabilities"]["__unknown__"], 0.1)
        self.assertEqual(self.engine.executions, 1)
        self.engine.tokens = 4097
        with self.assertRaises(DecisionError) as ctx:
            rt.dispatch("POST", "/v1/systemone", "secret", self.envelope, "too-long")
        self.assertEqual(ctx.exception.status, 413)
        self.assertEqual(self.engine.executions, 1)

    def test_timed_out_preparation_keeps_execution_slot_until_worker_finishes(self):
        rt = self.runtime(timeout=0.3)
        self.engine.block = True
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(rt.dispatch, "POST", "/v1/prepare", "secret", self.envelope, "one")
            self.assertTrue(self.engine.started.wait(2))
            with self.assertRaises(DecisionError) as ctx:
                first.result(2)
            self.assertEqual(ctx.exception.status, 504)
            second = pool.submit(
                rt.dispatch, "POST", "/v1/systemone", "secret", self.envelope, "two"
            )
            time.sleep(0.01)
            self.assertEqual(self.engine.peak, 1)
            self.engine.release.set()
            self.assertEqual(second.result(2)[0], 200)
        self.assertEqual(self.engine.peak, 1)


class ImageReferenceTest(unittest.TestCase):
    def test_reference_hashes_detect_changed_weights_and_configuration(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from zils import imajev

        pins = {
            "base": {"config.json": hashlib.sha256(b"{}").hexdigest()},
            "adapter": {"weights": hashlib.sha256(b"stock").hexdigest()},
            "runtime": {"engine.py": hashlib.sha256(b"source").hexdigest()},
        }
        with tempfile.TemporaryDirectory() as temp, patch.object(imajev, "PINS", pins):
            root = Path(temp)
            for folder, name, raw in [
                ("base", "config.json", b"{}"),
                ("adapter", "weights", b"stock"),
                ("runtime", "engine.py", b"source"),
            ]:
                (root / folder).mkdir()
                (root / folder / name).write_bytes(raw)
            manifest = imajev.build_manifest(root)
            (root / "release.json").write_text(json.dumps(manifest))
            self.assertEqual(imajev.verify_reference(root), manifest)
            (root / "adapter/weights").write_bytes(b"other")
            with self.assertRaises(ValueError):
                imajev.verify_reference(root)

    def test_image_base_preparation_never_downloads_a_text_base(self):
        from unittest.mock import patch

        from zils import models, runtime

        with (
            patch.dict("os.environ", {"ZILS_IMAGE_REFERENCE": ""}),
            patch(
                "huggingface_hub.snapshot_download",
                side_effect=AssertionError("image request attempted text download"),
            ),
        ):
            with self.assertRaises(ValueError):
                runtime.prepare_base(models.IMAJEV)


class RecordingEngine(Engine):
    def __init__(self):
        super().__init__()
        self.slot = None
        self.records = []
        self.corrupt = set()

    def activate_release(self, path, release):
        self.slot = None
        if release["release_id"] in self.corrupt:
            raise ValueError("Corrupt readout")
        self.slot = (
            release["files"]["adapter_model.safetensors"],
            release["files"]["decision_readout.safetensors"],
            release["temperature"],
        )

    def predict(self, prepared, temperature=1.0):
        if self.slot is None:
            raise AssertionError("Unverified image slot was used")
        self.records.append(self.slot)
        self.asserted_temperature = temperature
        return super().predict(prepared, temperature)


class ImageSwitchingTest(ImageRuntimeTest):
    def runtime(self, timeout=2):
        self.engine = RecordingEngine()
        question = copy.deepcopy(next(iter(BODY["questions"].values())))
        self.releases = {
            name: {
                "release_id": name,
                "fingerprint": letter * 64,
                "temperature": temperature,
                "files": {
                    "adapter_model.safetensors": letter + "-adapter",
                    "decision_readout.safetensors": letter + "-head",
                },
                **(
                    {"task": {"question": question, "outcome_order": list(question["criteria"])}}
                    if name != "image-release"
                    else {}
                ),
            }
            for name, letter, temperature in [
                ("image-release", "a", 1.0),
                ("client-a", "b", 0.5),
                ("client-b", "c", 2.0),
            ]
        }
        rt = self.module().ImageRuntime(
            self.engine, self.releases, self.origin, token="secret", timeout=timeout
        )
        self.addCleanup(rt.close)
        return rt

    def call(self, rt, model):
        body = copy.deepcopy(self.envelope)
        body["request"]["model"] = model
        body["fingerprint"] = self.releases[model]["fingerprint"]
        return rt.dispatch("POST", "/v1/systemone", "secret", body, "test")

    def test_adapter_head_and_temperature_switch_together_and_stock_is_restored(self):
        rt = self.runtime()
        for model in ("client-a", "image-release", "client-b", "client-a"):
            self.call(rt, model)
        self.assertEqual(
            self.engine.records,
            [
                ("b-adapter", "b-head", 0.5),
                ("a-adapter", "a-head", 1.0),
                ("c-adapter", "c-head", 2.0),
                ("b-adapter", "b-head", 0.5),
            ],
        )
        self.assertEqual(rt.active_release, "client-a")

    def test_failed_switch_invalidates_slot_and_next_request_reloads(self):
        rt = self.runtime()
        self.call(rt, "client-a")
        self.engine.corrupt.add("client-b")
        with self.assertRaises(DecisionError):
            self.call(rt, "client-b")
        self.assertIsNone(rt.active_release)
        self.assertEqual(len(self.engine.records), 1)
        self.call(rt, "client-a")
        self.assertEqual(self.engine.records[-1], ("b-adapter", "b-head", 0.5))

    def test_concurrent_calls_cannot_interleave_activation_and_prediction(self):
        rt = self.runtime()
        self.engine.block = True
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(self.call, rt, "client-a")
            self.assertTrue(self.engine.started.wait(2))
            second = pool.submit(self.call, rt, "client-b")
            time.sleep(0.03)
            self.assertEqual(self.engine.slot, ("b-adapter", "b-head", 0.5))
            self.engine.release.set()
            self.assertEqual(first.result(2)[0], 200)
            self.assertEqual(second.result(2)[0], 200)
        self.assertEqual(
            self.engine.records, [("b-adapter", "b-head", 0.5), ("c-adapter", "c-head", 2.0)]
        )
        self.assertEqual(self.engine.peak, 1)

    def test_private_model_refuses_a_different_question_before_activation(self):
        rt = self.runtime()
        body = copy.deepcopy(self.envelope)
        body["request"]["model"] = "client-a"
        body["fingerprint"] = "b" * 64
        body["request"]["questions"]["inspection"]["instructions"] = "Different task"
        with self.assertRaises(DecisionError):
            rt.dispatch("POST", "/v1/systemone", "secret", body, "wrong-task")
        self.assertEqual(self.engine.executions, 0)

    def test_a_reference_that_expires_while_queued_never_executes(self):
        from datetime import datetime, timedelta, timezone

        rt = self.runtime()
        self.engine.block = True
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(self.call, rt, "client-a")
            self.assertTrue(self.engine.started.wait(2))
            body = copy.deepcopy(self.envelope)
            body["image"]["expires_at"] = (
                datetime.now(timezone.utc) + timedelta(seconds=0.12)
            ).isoformat()
            second = pool.submit(rt.dispatch, "POST", "/v1/systemone", "secret", body, "expired")
            time.sleep(0.18)
            self.engine.release.set()
            first.result(2)
            with self.assertRaises(DecisionError):
                second.result(2)
        self.assertEqual(self.engine.executions, 1)


class ImageCatalogTest(unittest.TestCase):
    def test_discovery_adds_only_verified_image_releases_without_restart(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from tests.test_adapter_releases import Source, fixture as text_fixture
        from tests.test_image_releases import SCHEMA, fixture
        from zils.adapter_releases import publish
        from zils.image_server import ImageRuntime

        with tempfile.TemporaryDirectory() as tmp, patch("zils.imajev.CHECKPOINT_SCHEMA", SCHEMA):
            root = Path(tmp)
            releases = root / "releases"
            releases.mkdir()
            rt = ImageRuntime(
                RecordingEngine(),
                {"stock": {"fingerprint": "a" * 64, "temperature": 1}},
                "https://storage.example",
                token="secret",
                release_root=releases,
            )
            self.addCleanup(rt.close)
            job = fixture(root / "image")
            release = publish(Source(job, root / "image"), job["id"], releases)
            text = text_fixture(root / "text")
            other = publish(Source(text, root / "text"), text["id"], releases)
            bad = fixture(root / "bad")
            invalid = publish(Source(bad, root / "bad"), bad["id"], releases)
            path = releases / invalid["release_id"] / "decision_readout.safetensors"
            path.chmod(0o600)
            path.write_bytes(b"corrupt")
            health = rt.dispatch("GET", "/health", "secret", {}, "health")[1]["models"]
            self.assertIn(release["release_id"], health)
            self.assertNotIn(other["release_id"], health)
            self.assertNotIn(invalid["release_id"], health)


class NativeInputBoundaryTest(unittest.TestCase):
    def test_native_context_overflow_and_question_validation_are_client_errors(self):
        import tempfile
        from pathlib import Path
        from types import ModuleType, SimpleNamespace
        from unittest.mock import Mock, patch

        from zils.imajev import ImageEngine

        question = {
            "type": "choice",
            "instructions": "Inspect",
            "criteria": {"normal": None, "damaged": None},
        }
        native = ModuleType("vision_decision.jev_api")
        native.to_request_with_plan = Mock(
            return_value=(SimpleNamespace(fields=[object()], state={}), None)
        )
        scoring = ModuleType("vision_decision.scoring")
        scoring.compile_question = Mock(
            return_value=(
                "Prompt",
                [("normal", None), ("damaged", None), ("__unknown__", None)],
                ["normal", "damaged", "unknown"],
            )
        )
        engine = object.__new__(ImageEngine)
        engine.engine = SimpleNamespace(
            max_options=255,
            max_length=4096,
            prompt_layout="fixture",
            labels=lambda *a: ["A", "B", "C"],
            prepare=Mock(side_effect=ValueError("Processed request exceeds the 4096-token limit")),
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                "sys.modules",
                {"vision_decision.jev_api": native, "vision_decision.scoring": scoring},
            ),
        ):
            image = Path(tmp) / "photo.png"
            image.write_bytes(png())
            with self.assertRaises(DecisionError) as error:
                engine.prepare(image, {}, question)
            self.assertEqual(error.exception.status, 413)
            native.to_request_with_plan.side_effect = ValueError("native validation rejected input")
            with self.assertRaises(DecisionError) as error:
                engine.prepare(image, {}, question)
            self.assertEqual(error.exception.status, 422)


class ImageBillingMeterTest(unittest.TestCase):
    def test_counts_visual_tokens_plus_logical_input_without_prompt_or_asset_metadata(self):
        import json
        from types import SimpleNamespace

        import torch

        from zils.imajev import ImageEngine

        class Tokenizer:
            def encode(self, text, add_special_tokens):
                self.text = text
                assert add_special_tokens is False
                return [1] * 37

        tokenizer = Tokenizer()
        engine = object.__new__(ImageEngine)
        engine.engine = SimpleNamespace(
            processor=SimpleNamespace(tokenizer=tokenizer),
            model=SimpleNamespace(config=SimpleNamespace(image_token_id=77)),
        )
        prepared = {"inputs": {"input_ids": torch.tensor([[2, 77, 77, 3, 4]])}, "input_tokens": 5}
        self.assertEqual(engine.billable_input(BODY, prepared), 39)
        self.assertEqual(
            json.loads(tokenizer.text), {"state": BODY["state"], "questions": BODY["questions"]}
        )
        self.assertNotIn("asset_id", tokenizer.text)
        prepared["inputs"]["input_ids"] = torch.tensor([[2, 3, 4]])
        with self.assertRaises(DecisionError):
            engine.billable_input(BODY, prepared)
