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

    def prepare(self, image, state, question):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.set()
        if self.block:
            self.release.wait(2)
        self.active -= 1
        return {"input_tokens": self.tokens}

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
