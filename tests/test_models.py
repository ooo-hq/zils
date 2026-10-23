"""Model identity, artifact integrity and legacy compatibility at the queue interface."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import zils
from tests.promoted_profile import promoted_profile
from tests.test_jobs import POLICY, examples
from zils import benchmark, jobs, models


class ModelTest(unittest.TestCase):
    def test_job_history_reads_retired_profiles_without_enabling_them(self):
        archived = {
            "id": "jevk5-4b-v0.3",
            "name": "JevK5 4B",
            "base": "alibiserikbay/JevK5",
            "base_revision": "c4f7fdb3aeab5582336406e78d3bef11bf98833d",
        }
        for job in (
            {"model_profile": archived},
            {"manifest": {"model": archived}},
            {"model_profile": archived, "manifest": {"model": archived}},
        ):
            with self.subTest(job=job):
                profile = models.job_history_profile(job)
                self.assertEqual(profile, archived)
                profile["name"] = "changed"
                self.assertEqual(models.job_history_profile(job), archived)
                with self.assertRaisesRegex(ValueError, "unsupported model identity"):
                    models.job_model(job)
        with self.assertRaisesRegex(ValueError, "unsupported model identity"):
            models.validate_spec(archived)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            models.text_runtime(archived["id"])
        for model, frozen in models.RETIRED_SPECS.items():
            with self.subTest(model=model):
                self.assertEqual(
                    models.job_history_profile({"manifest": {"model": frozen}}), frozen
                )
                with self.assertRaisesRegex(ValueError, "unsupported model identity"):
                    models.job_model({"model_profile": frozen})

    def test_job_history_rejects_changed_or_inconsistent_archived_profiles(self):
        archived = {
            "id": "jevk5-4b-v0.3",
            "name": "JevK5 4B",
            "base": "alibiserikbay/JevK5",
            "base_revision": "c4f7fdb3aeab5582336406e78d3bef11bf98833d",
        }
        for profile in (
            {**archived, "base_revision": "0" * 40},
            {**archived, "name": "different"},
            {**archived, "extra": "untrusted"},
            {"id": archived["id"]},
            {**archived, "id": "unknown-retired-model"},
        ):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                models.job_history_profile({"model_profile": profile})
        for profile, frozen in (
            (archived, models.spec(models.D3)),
            (models.spec(models.D3), archived),
        ):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                models.job_history_profile(
                    {"model_profile": profile, "manifest": {"model": frozen}}
                )

    def test_job_history_preserves_current_and_legacy_validation(self):
        self.assertEqual(models.job_history_profile({}), models.spec(models.KEV))
        for model in models.SPECS:
            profile = models.spec(model)
            job = {"model_profile": profile, "manifest": {"model": profile}}
            with self.subTest(model=model):
                self.assertEqual(models.job_history_profile(job), profile)
                job["model_profile"] = {**profile, "base_revision": "0" * 40}
                with self.assertRaises(ValueError):
                    models.job_history_profile(job)

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
            models.write_metadata(tmp, kind="base", model=models.D3)
            spec = models.spec(models.D3)
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
                model=models.D3,
                allow_training_data_export=True,
            )
            self.assertEqual(manifest["model"], models.spec(models.D3))
            self.assertEqual(models.job_model({"manifest": manifest}), models.D3)
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
            models.write_metadata(base, kind="base", temperature=1.22, model=models.D3)
            self.assertEqual(models.artifact_files(base), ("model.json",))
            entry = zils.submission(base, 0)
            zils.stage(entry, root / "copy")
            self.assertEqual(zils.checkpoint_hash(root / "copy"), entry["sha256"])
            models.set_temperature(root / "copy", 1.0)
            self.assertNotEqual(zils.checkpoint_hash(root / "copy"), entry["sha256"])
            self.assertEqual(models.temperature(base), 1.22)
            candidate = root / "candidate"
            candidate.mkdir()
            models.write_metadata(candidate, model=models.D3)
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
            models.write_metadata(root, model=models.D3)
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
        self.assertEqual(models.candidate_base_env(models.D3), "ZILS_D3_CANDIDATE_BASE")
        for model in (models.KEV, *models.RETIRED_MODELS):
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, "no base-candidate support"):
                    models.candidate_base_env(model)

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

    def test_promoted_profile_is_a_d3_family_profile(self):
        """A base promoted from d3-mini: new weights on d3's runtime, same pool."""
        d3 = models.D3
        for model in models.d3_family():  # each released profile pins its own published revision
            self.assertEqual(models.spec(model)["base_revision"], models.pins(model)["revision"])
        with promoted_profile() as promoted:
            self.assertEqual(
                tuple(m for m in models.d3_family() if models.pool_id(m) == "d3-mini"),
                (d3, promoted),
            )
            self.assertIn(promoted, models.TEXT_ADAPTER_MODELS)
            self.assertEqual(
                models.DEFAULT_TRAINING_MODEL, d3
            )  # channels move by configuration, not here
            spec, parent = models.spec(promoted), models.spec(d3)
            self.assertEqual(spec["parent"], {"id": d3, "base_revision": parent["base_revision"]})
            same = (
                "runtime_revision",
                "prompt",
                "max_input_tokens",
                "recipe",
                "calibration",
                "inputs",
            )
            self.assertEqual({k: spec[k] for k in same}, {k: parent[k] for k in same})
            self.assertNotEqual(models.profile_identity(promoted), models.profile_identity(d3))
            self.assertEqual(models.runtime_python_env(promoted), models.runtime_python_env(d3))
            self.assertEqual(models.candidate_files(promoted), models.TEXT_ADAPTER_FILES)
        self.assertNotIn(promoted, models.SPECS)
        self.assertNotIn(promoted, models.TEXT_ADAPTER_MODELS)

    def test_d3_identity_hashes_pins_file(self):
        pins = models.pins(models.D3)
        self.assertEqual(pins["python"], "3.12")
        self.assertEqual(pins["packages"]["torchvision"], "0.25.0+cu130")
        self.assertEqual(len(pins["files"]), 25)  # every manifest file except README.md, assets/*
        identity = models.profile_identity(models.D3)
        self.assertEqual(set(identity), {"profile_sha256", "runtime_sha256"})
        for model in models.d3_family():
            if model != models.D3:
                self.assertNotEqual(
                    identity["runtime_sha256"], models.profile_identity(model)["runtime_sha256"]
                )

    def test_retired_h2o_and_imajev_identities_are_unchanged(self):
        for model, expected in (
            (models.H2O, "d28055e0296cba2afa5ba7f9052a3173ac7196b21f841898254b8e6df2b84506"),
            ("imajev-4b-v1", "3d53e354ad72ae067f33bfa8175d281507c467780a8c6c807cc0eb41eceb459c"),
        ):
            with self.subTest(model=model):
                frozen = json.dumps(
                    models.RETIRED_SPECS[model], sort_keys=True, separators=(",", ":")
                )
                self.assertEqual(hashlib.sha256(frozen.encode()).hexdigest(), expected)
                self.assertNotIn(model, models.SPECS)
                with self.assertRaisesRegex(ValueError, "unsupported model identity"):
                    models.profile_identity(model)

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
        for model in models.RETIRED_MODELS:
            self.assertIsNone(models.runtime_python_env(model))
        self.assertEqual(models.candidate_files(models.D3), models.TEXT_ADAPTER_FILES)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            models.text_runtime("jevk5-4b-v0.3")

    def test_retired_metadata_stays_readable_but_cannot_be_created_or_run(self):
        for retired in models.RETIRED_MODELS:
            with self.subTest(model=retired):
                self.check_retired_metadata(retired)
        self.assertEqual(
            models.candidate_files(models.D3, kind="candidate")[-len(models.D3_CONTRACT_FILES) :],
            models.D3_CONTRACT_FILES,
        )

    def check_retired_metadata(self, retired):
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
            value = json.loads((Path(tmp) / "model.json").read_text())
            value["profile"] = models.RETIRED_SPECS[retired]
            (Path(tmp) / "model.json").write_text(json.dumps(value))
            self.assertEqual(models.checkpoint_model(tmp), retired)
            value["profile"] = {**value["profile"], "base_revision": "0" * 40}
            (Path(tmp) / "model.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "pinned profile"):
                models.metadata(tmp)
