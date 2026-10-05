"""Version comparisons use the customer's incumbent, with atomic promotion."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import zils
from tests.test_jobs import POLICY
from zils import jobs


def versioned_fixture(root, previous=None):
    from tests.test_adapter_releases import fixture
    from zils import benchmark

    job = fixture(root)
    selection = {"version": "zils-version-selection/v1", "root_job_id": job["id"], "previous": None}
    comparison_sha = job["initial_sha256"]
    if previous:
        comparison_sha = previous["result"]["delivery"]["sha256"]
        job["acceptance"] = {**POLICY, "previous_job_id": previous["id"]}
        selection.update(
            root_job_id=previous["manifest"]["selection"]["root_job_id"],
            previous={
                "job_id": previous["id"],
                "model_id": f"zils-adapter-{previous['id']}-{comparison_sha}",
                "sha256": comparison_sha,
            },
        )
    job["manifest"].update(job_id=job["id"], selection=selection, acceptance=job["acceptance"])
    (root / "data/manifest.json").write_text(json.dumps(job["manifest"]))
    job["job_sha256"] = benchmark.file_hash(root / "data/manifest.json")
    job["result"].update(selection=selection, baseline_reference_sha256=comparison_sha)
    job["result"]["delivery"]["acceptance"] = job["acceptance"]
    report = json.loads((root / "release.json").read_text())
    report.update(
        job_sha256=job["job_sha256"],
        selection=selection,
        baseline_reference_sha256=comparison_sha,
        acceptance=job["acceptance"],
    )
    (root / "release.json").write_text(json.dumps(report))
    return job


class SelectionTest(unittest.TestCase):
    def test_restricted_registration_promotes_versions_with_atomic_predecessor_check(self):
        from tests.test_adapter_releases import Source
        from zils.adapter_releases import publish, registry_entry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = versioned_fixture(root / "first")
            second = versioned_fixture(root / "second", first)
            stale = versioned_fixture(root / "stale", first)
            registry = root / "models.json"
            command = [
                sys.executable,
                "-m",
                "zils.workflow",
                "register",
                "--registry",
                str(registry),
                "--customer-only",
                "--runtime-url",
                "http://127.0.0.1:8921",
                "--token-env",
                "TOKEN",
            ]
            before = None
            for job, folder in ((first, "first"), (second, "second"), (stale, "stale")):
                release = publish(Source(job, root / folder), job["id"], root / "releases")
                entry = registry_entry(release, "http://127.0.0.1:8921", "TOKEN")
                result = subprocess.run(
                    command,
                    input=json.dumps({"entry": entry, "selection": release["selection"]}),
                    text=True,
                    capture_output=True,
                )
                if job is stale:
                    self.assertEqual(result.returncode, 3, result.stderr)
                    self.assertEqual(registry.read_bytes(), before)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout)["model_id"], release["release_id"])
                    before = registry.read_bytes()

    def test_versioned_release_publication_binds_comparison_and_exposes_task_alias(self):
        from tests.test_adapter_releases import Source
        from zils.adapter_releases import publish, registry_entry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = versioned_fixture(root / "source")
            source = Source(job, root / "source")
            release = publish(source, job["id"], root / "releases")
            entry = registry_entry(release, "http://127.0.0.1:8921", "TOKEN")
            self.assertEqual(entry["aliases"], ["zils-task-" + job["id"]])
            self.assertEqual(release["selection"], job["manifest"]["selection"])
            job["result"]["baseline_reference_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "comparison"):
                publish(source, job["id"], root / "other")

    def test_workflow_stale_promotion_requires_review_and_preserves_accepted_evidence(self):
        from tests.test_workflow import Store
        from zils.version_selection import StaleVersion
        from zils.workflow import Workflow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = versioned_fixture(root / "source")
            store = Store(job, root / "source")
            delivery = copy.deepcopy(job["result"]["delivery"])
            calls = []

            def stale(release):
                calls.append(release["release_id"])
                raise StaleVersion("Newer version already active")

            flow = Workflow(store, "approved", root / "releases", lambda: {"ready": True}, stale)
            flow.tick()
            self.assertEqual(job["result"]["workflow"]["state"], "needs_review")
            self.assertEqual(job["result"]["delivery"], delivery)
            flow.tick()
            self.assertEqual(len(calls), 1)

    def test_upgrade_evaluation_uses_incumbent_weights_and_frozen_serving_temperature(self):
        from tests.test_jobs import examples
        from zils import benchmark, models, validator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current_id, previous_id = str(uuid.uuid4()), str(uuid.uuid4())
            reference, incumbent = root / "reference", root / "comparison"
            reference.mkdir()
            models.write_metadata(reference, kind="base", temperature=1.22)
            incumbent.mkdir()
            models.write_metadata(incumbent, temperature=2.0)
            (incumbent / "adapter_config.json").write_text("{}")
            (incumbent / "adapter_model.safetensors").write_text("incumbent")
            sha = zils.checkpoint_hash(incumbent)
            selection = {
                "version": "zils-version-selection/v1",
                "root_job_id": previous_id,
                "previous": {
                    "job_id": previous_id,
                    "sha256": sha,
                    "model_id": f"zils-adapter-{previous_id}-{sha}",
                },
            }
            jobs.build(
                root / "benchmark",
                current_id,
                examples(),
                {**POLICY, "previous_job_id": previous_id},
                allow_training_data_export=True,
                model=models.JEVK5,
                selection=selection,
            )
            manifest_sha = benchmark.file_hash(root / "benchmark/manifest.json")
            config = {
                "job_id": current_id,
                "job_sha256": manifest_sha,
                "benchmark_sha256": manifest_sha,
                "initial_sha256": zils.checkpoint_hash(reference),
                "base_revision": models.spec(models.JEVK5)["base_revision"],
                "members": {},
            }
            runner = root / "fixture-runner"
            runner.write_text(
                f"#!{sys.executable}\n"
                + """import json, sys
from pathlib import Path
from zils import models
path = Path(sys.argv[sys.argv.index('--checkpoint') + 1])
t = models.temperature(path)
p = 0.8 if (path / 'adapter_config.json').exists() else 0.5
a, b = p ** (1/t), (1-p) ** (1/t)
print(json.dumps({'runtime': {'temperature': t}, 'predictions': [
    {'id': r['id'], 'elapsed_ms': 1, 'probabilities': {'true': a/(a+b), 'false': b/(a+b)}}
    for r in json.load(sys.stdin)]}))
"""
            )
            runner.chmod(0o700)
            report = validator.evaluate_round(
                config, root, root, {}, SimpleNamespace(runtime_python=str(runner), device="cpu")
            )
            self.assertAlmostEqual(report["baseline"]["brier"], 2 / 9)
            self.assertEqual(report["baseline"]["sha256"], sha)
            self.assertEqual(report["baseline"]["runtime"]["temperature"], 2.0)
            self.assertEqual(report["selection"], selection)
            self.assertEqual(report["delivery"]["status"], "no_qualifying_model")

    def test_comparison_pins_an_accepted_ready_adapter_owned_by_the_customer(self):
        from tests.test_adapter_releases import OTHER, OWNER, Source, fixture
        from zils.version_selection import freeze

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = fixture(root / "parent")
            model_id = f"zils-adapter-{parent['id']}-{parent['result']['delivery']['sha256']}"
            parent["result"]["workflow"] = {"state": "ready", "model_id": model_id}
            job = {
                "id": str(uuid.uuid4()),
                "owner_id": OWNER,
                "acceptance": {**POLICY, "previous_job_id": parent["id"]},
            }
            source = Source(parent, root / "parent")
            selection = freeze(source, job)
            self.assertEqual(selection["root_job_id"], parent["id"])
            self.assertEqual(selection["previous"]["model_id"], model_id)
            self.assertEqual(
                selection["previous"]["sha256"], parent["result"]["delivery"]["sha256"]
            )
            for changes in ({"owner_id": OTHER}, {"status": "failed"}):
                saved = copy.deepcopy(parent)
                parent.update(changes)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    freeze(source, job)
                parent.clear()
                parent.update(saved)
            parent["result"]["workflow"]["state"] = "activation_failed"
            with self.assertRaises(ValueError):
                freeze(source, job)

    def test_upgrade_policy_keeps_thresholds_and_requires_a_real_previous_job_id(self):
        policy = {**POLICY, "previous_job_id": str(uuid.uuid4())}
        jobs.validate_policy(policy)
        for previous in ("", "other-customer-model", None, 12):
            with self.subTest(previous=previous), self.assertRaises(ValueError):
                jobs.validate_policy({**POLICY, "previous_job_id": previous})

    def test_first_version_can_win_but_upgrade_must_beat_its_incumbent(self):
        candidate = {
            "status": "evaluated",
            "uid": 1,
            "sha256": "a" * 64,
            "brier": 0.30,
            "accuracy": 0.82,
            "skill": 0.4,
        }
        self.assertEqual(
            jobs.select({"status": "evaluated", "brier": 0.33}, [candidate], POLICY)["status"],
            "accepted",
        )
        upgrade = {**POLICY, "previous_job_id": str(uuid.uuid4())}
        self.assertEqual(
            jobs.select({"status": "evaluated", "brier": 0.28}, [candidate], upgrade)["status"],
            "no_qualifying_model",
        )
        self.assertEqual(
            jobs.select(
                {"status": "evaluated", "brier": 0.28}, [{**candidate, "brier": 0.25}], upgrade
            )["status"],
            "accepted",
        )

    def test_stale_upgrade_cannot_replace_newer_version_and_old_ids_remain(self):
        from zils.adapter_releases import register
        from zils.version_selection import StaleVersion

        owner, root, second, third = [str(uuid.uuid4()) for _ in range(4)]
        alias = "zils-task-" + root

        def entry(job, sha):
            return {
                "id": f"zils-adapter-{job}-{sha * 64}",
                "fingerprint": sha * 64,
                "owners": [owner],
                "aliases": [alias],
                "url": "http://127.0.0.1:8921",
                "token_env": "TOKEN",
                "release_date": "2026-10-05",
                "description": "Test",
            }

        first, better, stale = entry(root, "a"), entry(second, "b"), entry(third, "c")
        selection = {"version": "zils-version-selection/v1", "root_job_id": root, "previous": None}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            register(path, first, selection=selection)
            upgrade = {
                **selection,
                "previous": {"job_id": root, "model_id": first["id"], "sha256": "a" * 64},
            }
            register(path, better, selection=upgrade)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "selection"):
                register(path, stale)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(StaleVersion):
                register(path, stale, selection=upgrade)
            self.assertEqual(path.read_bytes(), before)
            register(path, better, selection=upgrade)
            catalog = json.loads(path.read_text())["models"]
            self.assertEqual({row["id"] for row in catalog}, {first["id"], better["id"]})
            self.assertEqual(
                next(row for row in catalog if alias in row["aliases"])["id"], better["id"]
            )
            foreign = copy.deepcopy(stale)
            foreign["owners"] = [str(uuid.uuid4())]
            with self.assertRaises(ValueError):
                register(
                    path,
                    foreign,
                    selection={
                        **selection,
                        "previous": {
                            "job_id": second,
                            "model_id": better["id"],
                            "sha256": "b" * 64,
                        },
                    },
                )


if __name__ == "__main__":
    unittest.main()
