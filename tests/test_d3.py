"""d3-mini runtime: pins, weight-free encoder contract, FP32 readout and parity."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from zils import d3, models


class FakeBackbone(torch.nn.Module):
    def forward(self, input_ids, attention_mask, use_cache):
        hidden = torch.zeros(1, input_ids.shape[1], 255, dtype=torch.bfloat16)
        hidden[0, -1, 0] = 1
        return SimpleNamespace(last_hidden_state=hidden)


class D3ModuleTest(unittest.TestCase):
    def test_pins_and_constants(self):
        self.assertEqual(d3.MAX_TOKENS, 4096)
        self.assertEqual(d3.PINS["revision"], models.spec(models.D3)["base_revision"])
        self.assertEqual(len(d3.TARGETS), 9)

    def test_pinned_base_path_rejects_hash_mismatch(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"ZILS_D3_BASE_DIR": tmp}),
        ):
            for name in d3.PINS["files"]:
                (Path(tmp) / name).parent.mkdir(parents=True, exist_ok=True)
                (Path(tmp) / name).write_bytes(b"x")
            with self.assertRaisesRegex(OSError, "d3 base file .* differs from its pin"):
                d3.pinned_base_path()

    def test_verify_runtime_rejects_wrong_torchvision(self):
        with (
            patch(
                "zils.d3.importlib.metadata.version",
                side_effect=lambda p: "0.23.0" if p == "torchvision" else d3.PINS["packages"][p],
            ),
            patch("zils.d3.sys.version_info", (3, 12, 12)),
        ):
            with self.assertRaisesRegex(OSError, "torchvision"):
                d3.verify_runtime()

    def test_validate_adapter_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            # jevk5.train records the profile's base (the zils-ai mirror), not the vendor repo.
            good = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "modules_to_save": None,
                "target_modules": list(d3.TARGETS),
                "base_model_name_or_path": models.spec(models.D3)["base"],
                "revision": models.spec(models.D3)["base_revision"],
            }
            (Path(tmp) / "adapter_config.json").write_text(json.dumps(good))
            d3.validate_adapter_config(tmp, d3.TARGETS)
            (Path(tmp) / "adapter_config.json").write_text(
                json.dumps({**good, "revision": "0" * 40})
            )
            with self.assertRaisesRegex(ValueError, "revision"):
                d3.validate_adapter_config(tmp, d3.TARGETS)

    def test_logits_and_probabilities_use_fp32_readout(self):
        model = d3.DecisionModel.__new__(d3.DecisionModel)
        model.device = "cpu"
        model.readout = torch.eye(255, dtype=torch.float32)
        model.base = FakeBackbone()
        self.assertEqual(model.logits([1, 2, 3], 3).dtype, torch.float32)
        probabilities = model.probabilities([1, 2, 3], ["a", "b", "c"], {"type": "choice"}, 1.0)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)
        self.assertEqual(max(probabilities, key=probabilities.get), "a")

    def test_encode_refuses_images_until_release(self):
        encoder = d3.Encoder.__new__(d3.Encoder)
        with self.assertRaisesRegex(ValueError, "later release"):
            encoder.encode(
                {}, {"type": "choice", "criteria": {"a": "a", "b": "b"}}, images=[b"png"]
            )

    def test_cli_help_lists_commands(self):
        out = subprocess.run(
            [sys.executable, "-m", "zils.d3", "--help"], capture_output=True, text=True, check=True
        ).stdout
        for word in ("reference", "train", "parity"):
            self.assertIn(word, out)

    def test_candidate_hashes_only_contract_files(self):
        pins = {name: hashlib.sha256(name.encode()).hexdigest() for name in d3.PINS["files"]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(d3.PINS["files"], pins):
            models.write_metadata(tmp, kind="candidate", model=models.D3)
            for name in models.D3_CANDIDATE_FILES[1:]:
                (Path(tmp) / name).write_bytes(name.encode())
            (Path(tmp) / "model-00001-of-00002.safetensors").write_bytes(b"new weights")
            d3.validate_candidate(tmp)  # MODEL_MANIFEST.json, requirements.txt, d3_server.py absent
            (Path(tmp) / "d3_runtime.py").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "d3_runtime.py differs"):
                d3.validate_candidate(tmp)

    def test_parity_reports_and_rejects_drift(self):
        reference = [0.7, 0.3]
        model = SimpleNamespace(
            encode=lambda state, question: ([1, 2], ["a", "b"]),
            probabilities=lambda ids, keys, question, t: dict(zip(keys, [0.7 + 4e-7, 0.3 - 4e-7])),
            runtime=SimpleNamespace(probabilities=lambda sequences, counts: [reference]),
        )
        case = {"id": "c1", "state": {}, "question": {"type": "choice"}}
        result = d3.parity(model, [case, case])
        self.assertEqual(result["cases"], 2)
        self.assertAlmostEqual(result["max_difference"], 4e-7)
        reference[:] = [0.3, 0.7]
        with self.assertRaisesRegex(RuntimeError, "c1"):
            d3.parity(model, [case])


if __name__ == "__main__":
    unittest.main()
