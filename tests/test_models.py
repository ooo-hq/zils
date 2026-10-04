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
