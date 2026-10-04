"""Model identity, artifact integrity and legacy compatibility at the queue interface."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import fez
from fez import benchmark, jobs, models
from tests.test_jobs import POLICY, examples


class ModelTest(unittest.TestCase):
    def test_compact_adapter_roundtrip_and_invalid_tensors(self):
        import torch
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        from safetensors.torch import load_file, save_file

        from fez.jevk5 import load_adapter_weights, save_adapter_weights

        model = get_peft_model(
            torch.nn.Sequential(torch.nn.Linear(3, 2)),
            LoraConfig(r=2, target_modules=["0"], bias="none"),
        )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.uniform_(-0.2, 0.2)
        original = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
        with tempfile.TemporaryDirectory() as tmp:
            save_adapter_weights(model, tmp)
            path = str(Path(tmp) / "adapter_model.safetensors")
            compact = load_file(path)
            self.assertEqual(set(compact), set(original))
            self.assertTrue(all(v.dtype == torch.bfloat16 for v in compact.values()))
            # Saving must not reduce the precision of the optimizer's live parameters.
            self.assertTrue(all(v.dtype == torch.float32 for v in original.values()))
            self.assertTrue(
                all(
                    torch.equal(v, original[k]) for k, v in get_peft_model_state_dict(model).items()
                )
            )
            load_adapter_weights(model, tmp)
            for key, value in get_peft_model_state_dict(model).items():
                self.assertTrue(torch.equal(value, original[key].bfloat16().float()))
            save_file(original, path)
            load_adapter_weights(model, tmp)
            for key, value in get_peft_model_state_dict(model).items():
                self.assertTrue(torch.equal(value, original[key]))
            key = next(iter(compact))
            for invalid in (
                torch.zeros_like(compact[key], dtype=torch.int32),
                torch.full_like(compact[key], float("nan")),
                torch.zeros(1),
            ):
                save_file({**compact, key: invalid}, path)
                with self.assertRaisesRegex(ValueError, "tensors"):
                    load_adapter_weights(model, tmp)

    def test_runner_imports_upstream_package_and_rejects_unsupported_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            models.write_metadata(tmp, kind="base")
            spec = models.spec(models.JEVK5)
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(fez.__file__).with_name("jevk5_runner.py")),
                    "--checkpoint",
                    tmp,
                    "--base",
                    spec["base"],
                    "--base-revision",
                    spec["base_revision"],
                    "--device",
                    "cpu",
                ],
                input="[]",
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 78, result.stderr)
            self.assertIn("BF16-capable CUDA GPU", result.stderr)
            self.assertNotIn("relative import", result.stderr)

    def test_model_identity_is_frozen_in_job_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "job"
            manifest = jobs.build(
                root,
                "new-job",
                examples(),
                POLICY,
                model=models.JEVK5,
                allow_training_data_export=True,
            )
            self.assertEqual(manifest["model"], models.spec(models.JEVK5))
            self.assertEqual(models.job_model({"manifest": manifest}), models.JEVK5)
            self.assertEqual(models.job_model({"manifest": {}}), models.KEV)
            self.assertEqual(benchmark.audit(root), examples())
            manifest["model"]["base_revision"] = "0" * 40
            with self.assertRaisesRegex(ValueError, "pinned"):
                jobs.audit(root, manifest)

    def test_base_and_adapter_hashes_bind_metadata_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            models.write_metadata(base, kind="base", temperature=1.22)
            self.assertEqual(models.artifact_files(base), ("model.json",))
            entry = fez.submission(base, 0)
            fez.stage(entry, root / "copy")
            self.assertEqual(fez.checkpoint_hash(root / "copy"), entry["sha256"])
            models.set_temperature(root / "copy", 1.0)
            self.assertNotEqual(fez.checkpoint_hash(root / "copy"), entry["sha256"])
            self.assertEqual(models.temperature(base), 1.22)
            candidate = root / "candidate"
            candidate.mkdir()
            models.write_metadata(candidate)
            (candidate / "adapter_config.json").write_text("{}")
            (candidate / "adapter_model.safetensors").write_bytes(b"fixture")
            original = fez.checkpoint_hash(candidate)
            (candidate / "adapter_model.safetensors").write_bytes(b"tampered")
            self.assertNotEqual(fez.checkpoint_hash(candidate), original)
            self.assertEqual(models.artifact_files(candidate), models.JEVK5_FILES)
            (candidate / "head.pt").write_text("mixed")
            with self.assertRaisesRegex(ValueError, "mixes"):
                fez.checkpoint_hash(candidate)

    def test_unknown_models_and_symlink_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models.write_metadata(root)
            value = json.loads((root / "model.json").read_text())
            value["model"] = "untrusted/replacement"
            (root / "model.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "metadata"):
                fez.checkpoint_hash(root)
            (root / "model.json").unlink()
            (root / "model.json").symlink_to(root / "missing.json")
            with self.assertRaisesRegex(ValueError, "regular"):
                fez.checkpoint_hash(root)
