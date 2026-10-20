"""Model identity, artifact integrity and legacy compatibility at the queue interface."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import zils
from tests.test_jobs import POLICY, examples
from zils import benchmark, jobs, models


class ModelTest(unittest.TestCase):
    def test_compact_adapter_roundtrip_and_invalid_tensors(self):
        import torch
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        from safetensors.torch import load_file, save_file

        from zils.jevk5 import load_adapter_weights, save_adapter_weights

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

    def test_runner_runs_as_a_script_and_rejects_unsupported_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            models.write_metadata(tmp, kind="base", model=models.H2O)
            spec = models.spec(models.H2O)
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(zils.__file__).with_name("jevk5_runner.py")),
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
                model=models.H2O,
                allow_training_data_export=True,
            )
            self.assertEqual(manifest["model"], models.spec(models.H2O))
            self.assertEqual(models.job_model({"manifest": manifest}), models.H2O)
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
            models.write_metadata(base, kind="base", temperature=1.22, model=models.H2O)
            self.assertEqual(models.artifact_files(base), ("model.json",))
            entry = zils.submission(base, 0)
            zils.stage(entry, root / "copy")
            self.assertEqual(zils.checkpoint_hash(root / "copy"), entry["sha256"])
            models.set_temperature(root / "copy", 1.0)
            self.assertNotEqual(zils.checkpoint_hash(root / "copy"), entry["sha256"])
            self.assertEqual(models.temperature(base), 1.22)
            candidate = root / "candidate"
            candidate.mkdir()
            models.write_metadata(candidate, model=models.H2O)
            (candidate / "adapter_config.json").write_text("{}")
            (candidate / "adapter_model.safetensors").write_bytes(b"fixture")
            original = zils.checkpoint_hash(candidate)
            (candidate / "adapter_model.safetensors").write_bytes(b"tampered")
            self.assertNotEqual(zils.checkpoint_hash(candidate), original)
            self.assertEqual(models.artifact_files(candidate), models.TEXT_ADAPTER_FILES)
            (candidate / "head.pt").write_text("mixed")
            with self.assertRaisesRegex(ValueError, "mixes"):
                zils.checkpoint_hash(candidate)

    def test_unknown_models_and_symlink_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models.write_metadata(root, model=models.H2O)
            value = json.loads((root / "model.json").read_text())
            value["model"] = "untrusted/replacement"
            (root / "model.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "metadata"):
                zils.checkpoint_hash(root)
            (root / "model.json").unlink()
            (root / "model.json").symlink_to(root / "missing.json")
            with self.assertRaisesRegex(ValueError, "regular"):
                zils.checkpoint_hash(root)

    def test_candidate_base_env_names_the_profile_override(self):
        self.assertEqual(models.candidate_base_env(models.H2O), "ZILS_H2O_CANDIDATE_BASE")
        self.assertEqual(models.candidate_base_env(models.D3), "ZILS_D3_CANDIDATE_BASE")
        with self.assertRaisesRegex(ValueError, "no base-candidate support"):
            models.candidate_base_env(models.IMAJEV)

    def test_d3_profile_is_pinned_and_unique(self):
        spec = models.spec(models.D3)
        self.assertEqual(spec["base_revision"], models.pins(models.D3)["revision"])
        self.assertEqual(
            models.pins(models.D3)["vendor"],
            {"repo": "vllm-sr/d3-mini", "revision": "a475367de0d8133a3c1a2a0dc12303d89cfa2ab4"},
        )
        self.assertEqual(spec["max_input_tokens"], 4096)
        self.assertEqual(spec["inputs"], {"images": 4, "video": True})
        revisions = [models.spec(m)["base_revision"] for m in models.SPECS]
        self.assertEqual(len(revisions), len(set(revisions)))
        self.assertIn(models.D3, models.TEXT_ADAPTER_MODELS)

    def test_promoted_round_5_is_a_d3_family_profile(self):
        """zils-ai/d3-mini-4.5b-zils-r5: the round-5 winner (checkpoint a472d0a4…), new weights on d3's runtime."""
        r5, d3 = models.D3_ZILS_R5, models.D3
        self.assertEqual(
            tuple(m for m in models.d3_family() if models.pool_id(m) == "d3-mini"), (d3, r5)
        )
        self.assertIn(r5, models.TEXT_ADAPTER_MODELS)
        self.assertEqual(
            models.DEFAULT_TRAINING_MODEL, d3
        )  # channels move by configuration, not here
        spec, parent = models.spec(r5), models.spec(d3)
        self.assertEqual(spec["parent"], {"id": d3, "base_revision": parent["base_revision"]})
        same = ("runtime_revision", "prompt", "max_input_tokens", "recipe", "calibration", "inputs")
        self.assertEqual({k: spec[k] for k in same}, {k: parent[k] for k in same})
        for model in models.d3_family():  # each profile pins its own published revision
            self.assertEqual(models.spec(model)["base_revision"], models.pins(model)["revision"])
        pins, base = models.pins(r5), models.pins(d3)
        self.assertEqual(
            {k: v for k, v in pins.items() if k not in ("files", "revision")},
            {k: v for k, v in base.items() if k not in ("files", "revision")},
        )
        changed = sorted(k for k, v in pins["files"].items() if base["files"].get(k) != v)
        self.assertEqual(
            changed,
            ["NOTICE", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        )
        self.assertNotEqual(models.profile_identity(r5), models.profile_identity(d3))
        self.assertEqual(models.runtime_python_env(r5), models.runtime_python_env(d3))
        self.assertEqual(models.candidate_files(r5), models.TEXT_ADAPTER_FILES)

    def test_d3_identity_hashes_pins_file(self):
        pins = models.pins(models.D3)
        self.assertEqual(pins["python"], "3.12")
        self.assertEqual(pins["packages"]["torchvision"], "0.25.0+cu130")
        self.assertEqual(len(pins["files"]), 25)  # every manifest file except README.md, assets/*
        identity = models.profile_identity(models.D3)
        self.assertEqual(set(identity), {"profile_sha256", "runtime_sha256"})
        self.assertNotEqual(
            identity["runtime_sha256"], models.profile_identity(models.H2O)["runtime_sha256"]
        )

    def test_h2o_and_imajev_identities_are_unchanged(self):
        self.assertEqual(
            models.profile_identity(models.H2O)["profile_sha256"],
            "d28055e0296cba2afa5ba7f9052a3173ac7196b21f841898254b8e6df2b84506",
        )
        self.assertEqual(
            models.profile_identity(models.IMAJEV)["profile_sha256"],
            "3d53e354ad72ae067f33bfa8175d281507c467780a8c6c807cc0eb41eceb459c",
        )
        path = Path(models.__file__).with_name("h2o-pins.json")
        self.assertEqual(models.pins(models.H2O), json.loads(path.read_text()))

    def test_d3_metadata_requires_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            models.write_metadata(tmp, model=models.D3)
            self.assertEqual(models.metadata(tmp)["profile"], models.spec(models.D3))
            (Path(tmp) / "model.json").write_text(
                json.dumps(
                    {
                        "version": models.VERSION,
                        "model": models.D3,
                        "kind": "adapter",
                        "temperature": 1.0,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "pinned profile"):
                models.metadata(tmp)

    def test_text_runtime_and_interpreter_lookup(self):
        self.assertEqual(models.text_runtime(models.D3).__name__, "zils.d3")
        self.assertEqual(models.runtime_python_env(models.D3), "ZILS_D3_RUNTIME_PYTHON")
        self.assertEqual(models.runtime_python_env(models.H2O), "ZILS_H2O_RUNTIME_PYTHON")
        self.assertEqual(models.candidate_files(models.D3), models.TEXT_ADAPTER_FILES)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            models.text_runtime("jevk5-4b-v0.3")

    def test_retired_jevk5_metadata_stays_readable_but_cannot_be_created_or_run(self):
        retired = "jevk5-4b-v0.3"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unsupported model identity"):
                models.write_metadata(tmp, model=retired)
            (Path(tmp) / "model.json").write_text(
                json.dumps(
                    {
                        "version": models.VERSION,
                        "model": retired,
                        "kind": "adapter",
                        "temperature": 1.0,
                    }
                )
            )
            self.assertEqual(models.checkpoint_model(tmp), retired)
            self.assertEqual(models.temperature(tmp), 1.0)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                models.artifact_files(tmp)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                models.spec(retired)
        self.assertEqual(
            models.candidate_files(models.D3, kind="candidate")[-len(models.D3_CONTRACT_FILES) :],
            models.D3_CONTRACT_FILES,
        )
