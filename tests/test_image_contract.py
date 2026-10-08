"""Image requests retain unknown mass and cannot change legacy model identities."""

import copy
import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from zils import checkpoint_hash, models
from zils.api import FileRegistry, Registry
from zils.decisions import DecisionError, validate_request

BODY = {
    "model": "image-release",
    "state": {},
    "images": [{"asset_id": "10000000-0000-4000-8000-000000000001"}],
    "questions": {
        "inspection": {
            "type": "choice",
            "criteria": {"normal": None, "damaged": None},
        }
    },
}


class ImageContractTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("zils.image_contract"), "image contract missing"
        )
        return importlib.import_module("zils.image_contract")

    def response(self, probabilities, count=442):
        return self.module().make_image_response(
            "image-release",
            BODY,
            {
                "inspection": {"probabilities": probabilities, "input_tokens": count},
            },
        )

    def test_unknown_mass_and_processor_tokens_are_preserved(self):
        result = self.response({"normal": 0.2, "damaged": 0.3, "__unknown__": 0.5})
        answer = result["answers"]["inspection"]
        self.assertAlmostEqual(answer["probabilities"]["damaged"], 0.6)
        self.assertEqual(answer["unknown_probability"], 0.5)
        self.assertAlmostEqual(answer["confidence"], 0.1)
        self.assertTrue(answer["abstained"])
        self.assertEqual(result["usage"], {"input_tokens": 442, "output_tokens": 0})

    def test_ties_follow_declared_options_not_response_dictionary_order(self):
        answer = self.response({"__unknown__": 0.5, "damaged": 0, "normal": 0.5})["answers"][
            "inspection"
        ]
        self.assertFalse(answer["abstained"])
        self.assertEqual(answer["choice"], "normal")
        answer = self.response({"__unknown__": 0, "damaged": 0.5, "normal": 0.5})["answers"][
            "inspection"
        ]
        self.assertEqual(answer["choice"], "normal")
        answer = self.response({"normal": 0, "damaged": 0, "__unknown__": 1})["answers"][
            "inspection"
        ]
        self.assertTrue(answer["abstained"])
        self.assertEqual(answer["probabilities"], {"normal": 0.5, "damaged": 0.5})
        self.assertEqual(answer["confidence"], 0)

    def test_invalid_native_distributions_never_reach_customer(self):
        for values in (
            {"normal": 0.2, "damaged": 0.3},
            {"normal": 0.2, "damaged": 0.3, "__unknown__": 0.6},
            {"normal": float("nan"), "damaged": 0.3, "__unknown__": 0.5},
            {"normal": True, "damaged": 0, "__unknown__": 0},
            {"normal": -0.1, "damaged": 0.6, "__unknown__": 0.5},
        ):
            with self.subTest(values=values), self.assertRaises(DecisionError) as ctx:
                self.response(values)
            self.assertEqual(ctx.exception.status, 502)
        for count in (-1, True, 4097):
            with self.subTest(count=count), self.assertRaises(DecisionError):
                self.response({"normal": 0.8, "damaged": 0.1, "__unknown__": 0.1}, count)

    def test_image_shape_rejects_untrusted_references_and_unsupported_questions(self):
        module = self.module()
        self.assertEqual(module.validate_image_request(BODY), BODY)
        cases = []
        for images in (
            [],
            BODY["images"] * 2,
            [{"asset_id": "../x"}],
            [{"asset_id": BODY["images"][0]["asset_id"], "url": "https://evil"}],
        ):
            cases.append({**BODY, "images": images})
        cases.append(
            {
                **BODY,
                "questions": {
                    "a": BODY["questions"]["inspection"],
                    "b": BODY["questions"]["inspection"],
                },
            }
        )
        for question in (
            {"type": "noul"},
            {"type": "choice", "criteria": {str(i): None for i in range(17)}},
            {"type": "choice", "criteria": {"normal": None, "__unknown__": None}},
            {"type": "choice", "criteria": {"normal": None, " ": None}},
        ):
            cases.append({**BODY, "questions": {"inspection": question}})
        cases.extend([{**BODY, "owner": "forged"}, {**BODY, "state": {"x": float("nan")}}])
        for body in cases:
            with self.subTest(body=body), self.assertRaises(DecisionError):
                module.validate_image_request(body)
        # Existing validator continues rejecting images; dispatch is explicit.
        with self.assertRaises(DecisionError):
            validate_request(BODY)

    def test_image_artifact_roundtrip_does_not_reinterpret_legacy_files(self):
        self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models.write_metadata(root)
            for name in models.JEVK5_FILES[:-1]:
                (root / name).write_bytes(b"fixture")
            original = checkpoint_hash(root)
            self.assertEqual(models.checkpoint_model(root), models.JEVK5)
            self.assertEqual(models.artifact_files(root), models.JEVK5_FILES)
            self.assertEqual(checkpoint_hash(root), original)
            models.write_metadata(root, model=models.IMAJEV)
            for name in ("decision_readout.json", "decision_readout.safetensors"):
                (root / name).write_bytes(b"fixture")
            self.assertEqual(models.checkpoint_model(root), models.IMAJEV)
            before = checkpoint_hash(root)
            models.set_temperature(root, 1.5)
            self.assertEqual(models.checkpoint_model(root), models.IMAJEV)
            self.assertNotEqual(checkpoint_hash(root), before)
            (root / "decision_readout.safetensors").unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                checkpoint_hash(root)
            meta = json.loads((root / "model.json").read_text())
            meta["profile"]["base_revision"] = "0" * 40
            (root / "model.json").chmod(0o600)
            (root / "model.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                models.metadata(root)

    def test_capabilities_are_immutable_and_legacy_defaults_are_equivalent(self):
        module = self.module()
        entry = {
            "id": "stock",
            "fingerprint": "a" * 64,
            "aliases": [],
            "owners": None,
            "url": "http://127.0.0.1:8900",
            "token_env": "TEST_RUNTIME_TOKEN",
            "release_date": "2026-10-08",
            "description": "Fixture",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            path.write_text(json.dumps({"models": [entry]}))
            registry = FileRegistry(path)
            legacy = copy.deepcopy(registry.resolve("stock", None))
            path.write_text(
                json.dumps({"models": [{**entry, "capabilities": module.TEXT_CAPABILITIES}]})
            )
            self.assertEqual(registry.resolve("stock", None), legacy)
            path.write_text(
                json.dumps({"models": [{**entry, "capabilities": module.IMAGE_CAPABILITIES}]})
            )
            self.assertEqual(registry.resolve("stock", None), legacy)
        image = Registry([{**entry, "capabilities": module.IMAGE_CAPABILITIES}])
        self.assertEqual(image.listing(None)["models"][0]["capabilities"]["max_images"], 1)
