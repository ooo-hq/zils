"""Private image release provenance and immutable task contracts."""

import copy
import hashlib
import json
import tempfile
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

import zils
from tests.test_adapter_releases import OTHER, OWNER, Source
from tests.test_image_jobs import fixture as dataset_fixture
from zils import image_jobs, imajev, jobs, models
from zils.image_metrics import public_metrics, score_rows

SCHEMA = {
    "json": {
        "adapter_config.json": {"r": 64},
        "decision_readout.json": {"version": 1, "codes": ["A", "B", "C"]},
    },
    "tensors": {
        "adapter_model.safetensors": {
            "language_model.lora_A.weight": {"dtype": "F32", "shape": [2, 2]}
        },
        "decision_readout.safetensors": {"weight": {"dtype": "F32", "shape": [3, 2]}},
    },
}


def fixture(root, owner=OWNER, previous=None):
    root.mkdir()
    job, splits, assets = dataset_fixture()
    job.update(id=str(uuid.uuid4()), owner_id=owner, name="inspection")
    for asset in assets.values():
        asset.update(job_id=job["id"], owner_id=owner)
    if previous:
        job["acceptance"] = {**job["acceptance"], "previous_job_id": previous["job_id"]}
        job["selection"] = {
            "version": "zils-version-selection/v1",
            "root_job_id": previous["root"],
            "previous": {k: v for k, v in previous.items() if k != "root"},
        }
    manifest = image_jobs.build(root / "data", job, splits, assets, job["acceptance"])
    manifest_sha = hashlib.sha256((root / "data/manifest.json").read_bytes()).hexdigest()
    for name, value in SCHEMA["json"].items():
        (root / name).write_text(json.dumps(value))
    save_file(
        {"language_model.lora_A.weight": torch.ones(2, 2)}, str(root / "adapter_model.safetensors")
    )
    save_file({"weight": torch.ones(3, 2)}, str(root / "decision_readout.safetensors"))
    calibration = {
        "raw_checkpoint_sha256": "b" * 64,
        "dataset_sha256": "c" * 64,
        "benchmark_manifest_sha256": manifest_sha,
        "fit_cases": 2,
        "fit_split": "calibration",
        "method": "Full-native image macro-family NLL; 81 log-spaced temperatures in [0.25, 4]; calibration only",
    }
    models.write_metadata(root, model=models.IMAJEV, temperature=0.75, calibration=calibration)
    sha = zils.checkpoint_hash(root)
    order = ["normal", "damaged"]
    labels = order
    baseline = {
        "status": "evaluated",
        "uid": 0,
        **score_rows(
            labels, [{"normal": 0.4, "damaged": 0.3, "__unknown__": 0.3}] * 2, ["i"] * 2, order
        ),
    }
    winner = {
        "status": "evaluated",
        "uid": 1,
        "sha256": sha,
        **score_rows(
            labels,
            [
                {"normal": 0.9, "damaged": 0.05, "__unknown__": 0.05},
                {"normal": 0.05, "damaged": 0.9, "__unknown__": 0.05},
            ],
            ["i"] * 2,
            order,
        ),
    }
    delivery = jobs.select(baseline, [winner], job["acceptance"], model=models.IMAJEV)
    selection = manifest["selection"]
    comparison = previous["sha256"] if previous else "a" * 64
    result = {
        "model": models.spec(models.IMAJEV),
        "baseline": baseline,
        "miners": [{k: v for k, v in winner.items() if k != "sha256"}],
        "delivery": delivery,
        "selection": selection,
        "baseline_reference_sha256": comparison,
    }
    round_id = str(uuid.uuid4())
    release = {
        "job_id": job["id"],
        "job_sha256": manifest_sha,
        "round_id": round_id,
        "model": models.spec(models.IMAJEV),
        "base": models.spec(models.IMAJEV)["base"],
        "base_revision": models.spec(models.IMAJEV)["base_revision"],
        "initial_sha256": "a" * 64,
        "submitted_sha256": "b" * 64,
        "baseline_brier": baseline["brier"],
        "selection": selection,
        "baseline_reference_sha256": comparison,
        **delivery,
        "image_metrics": {
            "baseline": public_metrics(baseline),
            "candidate": public_metrics(winner),
        },
    }
    (root / "release.json").write_text(json.dumps(release))
    job.update(
        status="completed",
        updated_at="2026-10-08T12:00:00+00:00",
        manifest=manifest,
        job_sha256=manifest_sha,
        initial_sha256="a" * 64,
        release_prefix=f"{job['id']}/releases/{round_id}",
        result=result,
    )
    return job


class ImageReleasesTest(unittest.TestCase):
    def setUp(self):
        self.schema = patch("zils.imajev.CHECKPOINT_SCHEMA", SCHEMA, create=True)
        self.schema.start()
        self.addCleanup(self.schema.stop)

    def test_release_binds_profile_weights_question_and_is_owner_only(self):
        from zils.adapter_releases import publish, read_release, registry_entry
        from zils.api import Registry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            release = publish(Source(job, root / "source"), job["id"], root / "releases")
            self.assertEqual(release["model"], models.spec(models.IMAJEV))
            self.assertEqual(list(release["task"]["question"]["criteria"]), ["normal", "damaged"])
            self.assertEqual(release["task"]["question"], job["manifest"]["question"])
            self.assertEqual(read_release(root / "releases" / release["release_id"]), release)
            entry = registry_entry(release, "http://127.0.0.1:8931", "IMAGE_TOKEN")
            registry = Registry([entry])
            self.assertEqual(registry.listing(OTHER), {"models": []})
            self.assertEqual(registry.listing(OWNER)["models"][0]["task"], release["task"])
            self.assertEqual(entry["profile"], models.spec(models.IMAJEV))

    def test_changed_serving_inputs_are_rejected_before_registry_changes(self):
        from zils.adapter_releases import publish, read_release, register, registry_entry

        for change in (
            "readout",
            "base",
            "preprocessor",
            "owner",
            "temperature",
            "policy",
            "outcomes",
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                job = fixture(root / "source")
                release = publish(Source(job, root / "source"), job["id"], root / "releases")
                target = root / "releases" / release["release_id"]
                registry = root / "models.json"
                register(
                    registry,
                    registry_entry(release, "http://127.0.0.1:8931", "TOKEN"),
                    selection=release["selection"],
                )
                before = registry.read_bytes()
                if change == "readout":
                    path = target / "decision_readout.safetensors"
                    path.chmod(0o600)
                    path.write_bytes(b"changed")
                else:
                    path = target / (
                        "model.json"
                        if change in ("base", "preprocessor", "temperature")
                        else "source.json"
                    )
                    value = json.loads(path.read_text())
                    if change == "base":
                        value["profile"]["base_revision"] = "0" * 40
                    elif change == "preprocessor":
                        value["profile"]["preprocessor"] = "other"
                    elif change == "temperature":
                        value["temperature"] = 2
                    elif change == "owner":
                        value["owner_id"] = OTHER
                    elif change == "policy":
                        value["acceptance"]["min_positive_recall"] = 0.1
                    else:
                        value["manifest"]["question"]["criteria"] = {"pass": None, "fail": None}
                    path.chmod(0o600)
                    path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    read_release(target)
                self.assertEqual(registry.read_bytes(), before)

    def test_invalid_tensor_name_shape_dtype_and_values_cannot_publish(self):
        for kind in ("name", "shape", "dtype", "nan"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture(root / "source")
                tensor = torch.zeros(3, 2)
                if kind == "shape":
                    tensor = torch.zeros(2, 3)
                if kind == "dtype":
                    tensor = tensor.half()
                if kind == "nan":
                    tensor[0, 0] = float("nan")
                save_file(
                    {"other" if kind == "name" else "weight": tensor},
                    str(root / "source/decision_readout.safetensors"),
                )
                with self.assertRaises(ValueError):
                    imajev.validate_checkpoint(root / "source")

    def test_nonqualifying_job_leaves_registry_and_files_unchanged(self):
        from zils.adapter_releases import publish, register, registry_entry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            release = publish(Source(job, root / "source"), job["id"], root / "releases")
            registry = root / "models.json"
            register(
                registry,
                registry_entry(release, "http://127.0.0.1:8931", "TOKEN"),
                selection=release["selection"],
            )
            before = registry.read_bytes()
            paths = sorted(p.name for p in (root / "releases").iterdir())
            job["result"]["delivery"]["status"] = "no_qualifying_model"
            self.assertIsNone(publish(Source(job, root / "source"), job["id"], root / "releases"))
            self.assertEqual(registry.read_bytes(), before)
            self.assertEqual(sorted(p.name for p in (root / "releases").iterdir()), paths)

    def test_stale_concurrent_image_upgrade_cannot_replace_the_winner(self):
        from zils.adapter_releases import publish, register, registry_entry
        from zils.version_selection import StaleVersion

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = fixture(root / "first")
            release = publish(Source(first, root / "first"), first["id"], root / "releases")
            registry = root / "models.json"
            register(
                registry,
                registry_entry(release, "http://127.0.0.1:8931", "TOKEN"),
                selection=release["selection"],
            )
            previous = {
                "job_id": first["id"],
                "sha256": release["checkpoint_sha256"],
                "model_id": release["release_id"],
                "root": first["id"],
            }
            candidates = []
            for name in ("second", "third"):
                job = fixture(root / name, previous=previous)
                r = publish(Source(job, root / name), job["id"], root / "releases")
                candidates.append(r)

            def promote(r):
                try:
                    register(
                        registry,
                        registry_entry(r, "http://127.0.0.1:8931", "TOKEN"),
                        selection=r["selection"],
                    )
                    return "ready"
                except StaleVersion:
                    return "needs_review"

            with ThreadPoolExecutor(2) as pool:
                states = list(pool.map(promote, candidates))
            self.assertCountEqual(states, ["ready", "needs_review"])
            values = json.loads(registry.read_text())["models"]
            self.assertIn(release["release_id"], [x["id"] for x in values])
            self.assertEqual(
                len([x for x in values if "zils-task-" + first["id"] in x["aliases"]]), 1
            )

    def test_image_activation_uses_its_configured_runtime_and_retains_accepted_artifact(self):
        from tests.test_workflow import Store
        from zils.workflow import Workflow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            store = Store(job, root / "source")
            calls = []

            def activate(release):
                self.assertTrue((root / "image-releases" / release["release_id"]).is_dir())
                calls.append(release["release_id"])
                if len(calls) == 1:
                    raise OSError("temporary image runtime outage")
                return release["release_id"]

            def wrong_runtime(_):
                raise AssertionError("Image activation reached the text runtime")

            flow = Workflow(
                store,
                "approved",
                root / "text-releases",
                lambda: {"ready": True},
                wrong_runtime,
                image_releases=root / "image-releases",
                image_activate=activate,
            )
            flow.tick()
            self.assertEqual(job["result"]["workflow"]["state"], "activation_failed")
            self.assertFalse((root / "text-releases").exists())
            flow.tick()
            self.assertEqual(job["result"]["workflow"]["state"], "ready")
            self.assertEqual(len(calls), 2)

    def test_reload_cannot_add_a_task_contract_to_an_existing_stock_identity(self):
        from zils.adapter_releases import publish, registry_entry
        from zils.api import FileRegistry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            release = publish(Source(job, root / "source"), job["id"], root / "releases")
            entry = registry_entry(release, "http://127.0.0.1:8931", "TOKEN")
            entry = {k: v for k, v in entry.items() if k != "task"}
            entry["owners"] = None
            path = root / "registry.json"
            path.write_text(json.dumps({"models": [entry]}))
            registry = FileRegistry(path)
            changed = {**entry, "task": release["task"]}
            path.write_text(json.dumps({"models": [changed]}))
            self.assertNotIn("task", registry.resolve(entry["id"], OWNER))

    def test_upgrade_cannot_change_the_incumbents_callable_image_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = fixture(root / "parent")
            changed = copy.deepcopy(parent["manifest"])
            for asset in changed["assets"].values():
                asset["canonical_sha256"] = hashlib.sha256(
                    (asset["canonical_sha256"] + "new").encode()
                ).hexdigest()
                asset["pixel_sha256"] = asset["canonical_sha256"]
            image_jobs.validate_predecessor(changed, parent["manifest"])
            changed["question"]["instructions"] = "Decide something else"
            with self.assertRaises(ValueError):
                image_jobs.validate_predecessor(changed, parent["manifest"])
