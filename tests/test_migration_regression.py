"""Frozen-reference provenance, seed aggregation and production customer gates."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.promoted_profile import PROMOTED, promoted_profile
from tests.test_jobs import POLICY, examples
from tests.test_regression_suite import run_suite
from zils import base_round, benchmark, migration_regression as migration, models

RELEASE = {
    "job_id": "9c061f45-12aa-443a-a75d-520b20b2cb92",
    "fingerprint": "f" * 64,
    "checkpoint_sha256": "b" * 64,
    "temperature": 1.7,
    "model": models.spec(models.D3),
}


class MigrationRegressionTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(promoted_profile())

    def test_production_gates_distinguish_improvement_and_continuity(self):
        baseline = {"brier": 0.2, "accuracy": 0.9, "skill": 0.6}
        row = {**baseline, "adapter_sha256": "a" * 64}
        result = migration.compare(baseline, row, RELEASE, POLICY)
        self.assertFalse(result["strict_upgrade"])
        self.assertTrue(result["migration"])
        for changes in ({"brier": 0.200001}, {"accuracy": 0.79}, {"skill": 0}):
            self.assertFalse(
                migration.compare(baseline, row | changes, RELEASE, POLICY)["migration"]
            )
        self.assertTrue(
            migration.compare(baseline, row | {"brier": 0.18}, RELEASE, POLICY)["strict_upgrade"]
        )
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                migration.compare(baseline, row | {"brier": value}, RELEASE, POLICY)

    def test_every_seed_must_pass_not_just_the_mean(self):
        good = {"migration": True, "strict_upgrade": True}
        outcomes = {"a": {str(s): dict(good) for s in (553, 771, 991)}}
        self.assertTrue(migration.summary(outcomes)["stable_eligible"])
        outcomes["a"]["553"]["migration"] = False
        self.assertFalse(migration.summary(outcomes)["stable_eligible"])
        self.assertFalse(migration.summary({"a": {"553": good}})["stable_eligible"])
        with self.assertRaises(ValueError):
            migration.summary({})
        with self.assertRaises(ValueError):
            migration.summary({"a": {"1": good}, "b": {"2": good}})

    def test_references_bind_all_splits_and_reject_reused_test_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, current, release = root / "old", root / "a", root / "release"
            for p in (old, current, release):
                p.mkdir()
            data = examples()
            for split, rows in data.items():
                (old / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest = {"files": {p.name: benchmark.file_hash(p) for p in old.iterdir()}}
            (release / "manifest.json").write_text(json.dumps(manifest))
            fresh = copy.deepcopy(data)
            for row in fresh["test"]:
                row.update(
                    id="fresh-" + row["id"],
                    group_id="fresh-" + row["group_id"],
                    state={"ticket": "fresh-" + row["id"]},
                )
            for split, rows in fresh.items():
                (current / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            config = root / "references.json"
            entry = {"release": "release", "splits": "old", "fingerprint": RELEASE["fingerprint"]}
            config.write_text(json.dumps({"a": entry}))
            with patch("zils.migration_regression.releases.read_release", return_value=RELEASE):
                self.assertEqual(
                    migration.references(config, [current], models.D3)["a"][1], RELEASE
                )
                with self.assertRaisesRegex(ValueError, "incumbent profile"):
                    migration.references(config, [current], models.D3_POOL_MODELS["d3-lite"])
                config.write_text(json.dumps({"a": entry | {"fingerprint": "0" * 64}}))
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    migration.references(config, [current], models.D3)
                config.write_text(json.dumps({"a": entry}))
                (current / "test.jsonl").write_bytes((old / "test.jsonl").read_bytes())
                with self.assertRaisesRegex(ValueError, "fresh"):
                    migration.references(config, [current], models.D3)
                (old / "train.jsonl").write_text("changed")
                with self.assertRaisesRegex(ValueError, "source split changed"):
                    migration.references(config, [current], models.D3)

    def test_suite_scores_frozen_release_once_without_recalibrating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = root / "jobs/a"
            job.mkdir(parents=True)
            cases = examples()["test"]
            (job / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cases))
            ref = root / "reference"
            ref.mkdir()
            models.write_metadata(ref, kind="base", model=models.D3)
            candidate = root / "candidate"
            candidate.mkdir()
            models.write_metadata(candidate, kind="base", model=PROMOTED)
            predictions = [
                {"id": r["id"], "probabilities": {"true": 0.6, "false": 0.4}, "elapsed_ms": 1.0}
                for r in cases
            ]
            report = {
                "dataset_sha256": base_round.dataset_hash(cases),
                "miners": [{"uid": 1, "status": "evaluated", "predictions": predictions}],
            }
            with (
                patch(
                    "zils.migration_regression.references",
                    return_value={"a": (root / "frozen", RELEASE)},
                ),
                patch("zils.regression_suite.evaluate_adapter", return_value=report) as evaluate,
                patch(
                    "zils.base_round.fit_temperature",
                    side_effect=AssertionError("must not recalibrate frozen release"),
                ),
            ):
                calls, result = run_suite(
                    [
                        "--suite",
                        str(root),
                        "--incumbent-model",
                        models.D3,
                        "--candidate-model",
                        PROMOTED,
                        "--incumbent",
                        str(ref),
                        "--candidate",
                        str(candidate),
                        "--migration-references",
                        "references.json",
                        "--runtime-python",
                        "/incumbent/python",
                        "--candidate-runtime-python",
                        "/candidate/python",
                    ],
                    suite_jobs=[job],
                )
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(evaluate.call_args.args[0], root / "frozen")
            self.assertEqual(evaluate.call_args.args[3], "/incumbent/python")
            self.assertNotIn(models.candidate_base_env(models.D3), evaluate.call_args.args[4])
            self.assertEqual(len(calls), 6)
            self.assertEqual({c["seed"] for c in calls}, {553, 771, 991})
            self.assertTrue(result["migration"]["stable_eligible"])
            self.assertEqual(result["migration"]["references"]["a"]["temperature"], 1.7)
