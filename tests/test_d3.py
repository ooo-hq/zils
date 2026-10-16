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


SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
INDEX = "model.safetensors.index.json"


def weights(root, weight_map, tensors=None):
    """Tiny shards and an index; tensors[shard] overrides the keys a shard actually holds."""
    from safetensors.torch import save_file

    root.mkdir(exist_ok=True)
    (root / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    for shard in SHARDS:
        keys = (tensors or {}).get(shard, [k for k, v in weight_map.items() if v == shard])
        save_file({key: torch.zeros(1) for key in keys}, str(root / shard))


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
        model.base = FakeBackbone().eval()
        self.assertEqual(model.logits([1, 2, 3], 3).dtype, torch.float32)
        probabilities = model.probabilities([1, 2, 3], ["a", "b", "c"], {"type": "choice"}, 1.0)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)
        self.assertEqual(max(probabilities, key=probabilities.get), "a")
        # A model left in training mode is a node fault (exit 78), not the miner's.
        model.base.train()
        with self.assertRaisesRegex(OSError, "eval mode"):
            model.probabilities([1, 2, 3], ["a", "b", "c"], {"type": "choice"}, 1.0)

    def test_setup_invariants_raise_oserror(self):
        runtime = SimpleNamespace(
            fast=None,
            attention_mode=d3.PINS["attention"],
            temperature=d3.PINS["temperature"],
            readout_dtype=d3.PINS["readout_dtype"],
            prompt="d3",
            backbone_type="qwen3_5",
            max_length=2048,
        )
        native = SimpleNamespace(D3=SimpleNamespace(from_pretrained=lambda *a, **k: runtime))
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=True),
            patch("zils.d3.verify_runtime"),
            patch("zils.d3.base_path", return_value=Path(tmp)),
            patch("zils.d3.Encoder", return_value=SimpleNamespace(native=native, tokenizer=None)),
        ):
            models.write_metadata(tmp, kind="base", model=models.D3)
            with self.assertRaisesRegex(OSError, "pinned configuration"):
                d3.DecisionModel(tmp, "cuda")

    def test_vendor_code_comes_from_the_pinned_base_only(self):
        with tempfile.TemporaryDirectory() as pinned, tempfile.TemporaryDirectory() as candidate:
            (Path(pinned) / "d3_runtime.py").write_text(
                "def answer_codes(tokenizer):\n    return ['A'], [1]\n"
            )
            (Path(candidate) / "d3_runtime.py").write_text("raise ImportError('candidate code')\n")
            (Path(candidate) / "decision_config.json").write_text(
                json.dumps({"prompt": "d3", "codes": ["A"]})
            )
            d3.vendor.cache_clear()
            self.addCleanup(d3.vendor.cache_clear)
            self.addCleanup(sys.modules.pop, "d3_runtime", None)
            self.addCleanup(lambda: sys.path.remove(pinned) if pinned in sys.path else None)
            with (
                patch("zils.d3.pinned_base_path", return_value=Path(pinned)),
                patch("transformers.AutoTokenizer.from_pretrained"),
            ):
                encoder = d3.Encoder(Path(candidate))
            self.assertEqual(Path(encoder.native.__file__).parent, Path(pinned))
            self.assertNotIn(candidate, sys.path)

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

    def candidate(self, root):
        """A pinned-base fixture and a candidate whose contract files equal the patched pins."""
        parent = {"a.weight": SHARDS[0], "b.weight": SHARDS[1]}
        weights(root / "base", parent)
        candidate = root / "candidate"
        candidate.mkdir()
        models.write_metadata(candidate, kind="candidate", model=models.D3)
        for name in models.D3_CONTRACT_FILES:
            (candidate / name).write_bytes(name.encode())
        weights(candidate, parent)
        pins = {name: hashlib.sha256(name.encode()).hexdigest() for name in d3.PINS["files"]}
        self.enterContext(patch.dict(d3.PINS["files"], pins))
        self.enterContext(patch("zils.d3.pinned_base_path", return_value=root / "base"))
        return candidate, parent

    def test_candidate_hashes_only_contract_files(self):
        with tempfile.TemporaryDirectory() as root:
            tmp, _ = self.candidate(Path(root))
            d3.validate_candidate(tmp)  # MODEL_MANIFEST.json, requirements.txt, d3_server.py absent
            for extra, make in (
                ("extra.py", lambda path: path.write_text("")),
                ("__pycache__", lambda path: path.symlink_to(Path(tmp) / "config.json")),
            ):
                make(Path(tmp) / extra)
                with self.assertRaisesRegex(ValueError, f"unexpected entry: {extra}"):
                    d3.validate_candidate(tmp)
                (Path(tmp) / extra).unlink()
            (Path(tmp) / "d3_runtime.py").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "d3_runtime.py differs"):
                d3.validate_candidate(tmp)

    def test_candidate_weights_must_load_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            candidate, parent = self.candidate(Path(root))
            d3.validate_candidate(candidate)
            for message, weight_map, tensors in (
                ("index keys differ.*c.weight", {**parent, "c.weight": SHARDS[0]}, None),
                ("outside the pinned shards", {**parent, "b.weight": "x/" + SHARDS[1]}, None),
                ("differ from its index.*c.weight", parent, {SHARDS[0]: ["a.weight", "c.weight"]}),
            ):
                with self.subTest(message=message):
                    weights(candidate, weight_map, tensors)
                    with self.assertRaisesRegex(ValueError, message):
                        d3.validate_candidate(candidate)

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

    def test_capacity_floor_and_cuda_requirement(self):
        from zils.runtime import gpu_ready

        self.assertFalse(gpu_ready("mps", model=models.D3, minimum_mib=1))
        self.assertFalse(gpu_ready("cpu", model=models.D3))
        with patch("zils.runtime.subprocess.run", return_value=SimpleNamespace(stdout="12288")):
            self.assertTrue(gpu_ready("cuda", model=models.D3))
        with patch("zils.runtime.subprocess.run", return_value=SimpleNamespace(stdout="12287")):
            self.assertFalse(gpu_ready("cuda", model=models.D3))
        with (
            patch.dict(os.environ, {"ZILS_D3_MIN_FREE_MIB": "14000"}),
            patch("zils.runtime.subprocess.run", return_value=SimpleNamespace(stdout="13000")),
        ):
            self.assertFalse(gpu_ready("cuda", model=models.D3))


if __name__ == "__main__":
    unittest.main()
