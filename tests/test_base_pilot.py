"""Synthetic pilot inputs test admission mechanics, never establish real model quality."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_base_exam import inputs, sealed
from tests.test_base_metrics import predictions
from tests.test_base_release import save
from zils import base_exam, base_metrics, base_pilot, base_round, models


def pilot_fixture(root):
    runtime = base_pilot.runtime_contract(models.D3)
    cases = {p: sealed(root / p, p)[1] for p in ("development", "validation")}
    roster = {
        f"model-{i}": {
            "kind": "reference" if i < 5 else "pool",
            "model_sha256": f"{i + 1:064x}",
            "runtime_sha256": base_round.dataset_hash(runtime) if i >= 5 else f"{i + 10:064x}",
            "settings_sha256": "b" * 64,
            **(
                {
                    "edition": {
                        "date": "2026-10-21",
                        "url": "https://example.org/index",
                        "sha256": "c" * 64,
                    },
                    "full_score": 20 + 5 * i,
                }
                if i < 5
                else {"pool_id": "d3-mini"}
            ),
        }
        for i in range(7)
    }
    capacity = {
        name: save(
            root / f"{name}-capacity.json",
            {
                **{k: r[k] for k in ("model_sha256", "runtime_sha256", "settings_sha256")},
                "device": "synthetic-fixture",
                "peak_gpu_mib": 100,
                "available_gpu_mib": 200,
                "completed_at": 50,
                "probability_export_reviewed": True,
            },
        )
        for name, r in roster.items()
    }
    reports = {}
    for phase in cases:
        reports[phase] = {}
        for i, (name, r) in enumerate(roster.items()):
            reports[phase][name] = {}
            wrong = [4, 3, 2, 1, 0, 2, 1][i]

            def choose(c, index):
                import zils

                return (
                    next(k for k in zils.options(c["question"]) if k != c["label"])
                    if index % 8 < wrong
                    else c["label"]
                )

            for split, data in cases[phase].items():
                reports[phase][name][split] = save(
                    root / f"{phase}-{name}-{split}.json",
                    {
                        **{k: r[k] for k in ("model_sha256", "runtime_sha256", "settings_sha256")},
                        "dataset_sha256": base_round.dataset_hash(data),
                        "started_at": 100 if phase == "development" else 300,
                        "completed_at": 101 if phase == "development" else 301,
                        "predictions": predictions(data, choose, confidence=0.9),
                    },
                )
    bundle = {
        "version": "zils-base-pilot/v1",
        "phase": "validation",
        "pool_id": "d3-mini",
        "policy": dict(base_round.DEFAULT_POLICY),
        "blueprint": base_exam.blueprint(cases["development"]),
        "runtime_identity": runtime,
        "roster": roster,
        "pairs": [["model-5", "model-6"]],
        "within_pool_pair": ["model-5", "model-6"],
        "capacity": capacity,
        "draws": {p: {"path": str(root / p), **base_round.draw(root / p)} for p in cases},
        "reports": reports,
        "history": save(root / "history.json", inputs()[2]),
        "frozen_at": 200,
    }
    bundle["design_sha256"] = base_round.dataset_hash(base_pilot.design(bundle))
    return bundle


def precise(*args, **kwargs):
    return {"improvement": 1.0, "lower_95": 0.5, "upper_95": 1.5, "bootstrap_variance": 0.1}


class BasePilotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = pilot_fixture(self.root)
        self.cal = patch.object(base_round, "fit_temperature", return_value=1.0)
        self.cal.start()
        self.addCleanup(self.cal.stop)
        self.pair = patch.object(base_metrics, "paired_improvement", side_effect=precise)
        self.pair.start()
        self.addCleanup(self.pair.stop)

    def test_precision_boundary_and_pool_specific_verification(self):
        report = base_pilot.check(self.bundle, root=self.root)
        self.assertTrue(report["passed"], report["reasons"])
        base_pilot.verify_qualification(
            report,
            pool_id="d3-mini",
            policy=self.bundle["policy"],
            blueprint=self.bundle["blueprint"],
            runtime_identity=self.bundle["runtime_identity"],
        )
        with self.assertRaisesRegex(ValueError, "pool"):
            base_pilot.verify_qualification(
                report,
                pool_id="d3-lite",
                policy=self.bundle["policy"],
                blueprint=self.bundle["blueprint"],
                runtime_identity=self.bundle["runtime_identity"],
            )
        with patch.object(
            base_metrics, "paired_improvement", return_value={**precise(), "upper_95": 1.5001}
        ):
            self.assertFalse(base_pilot.check(self.bundle, root=self.root)["passed"])
        with patch.object(
            base_metrics, "paired_improvement", return_value={**precise(), "bootstrap_variance": 0}
        ):
            self.assertFalse(base_pilot.check(self.bundle, root=self.root)["passed"])

    def test_frozen_roster_disjointness_and_capacity_are_required(self):
        for mutate in (
            lambda b: b["roster"].pop("model-0"),
            lambda b: b.update(frozen_at=400),
            lambda b: b["draws"].update(validation=b["draws"]["development"]),
            lambda b: b["capacity"].pop("model-0"),
            lambda b: b["roster"]["model-0"].update(full_score=24),
            lambda b: b["roster"]["model-0"]["edition"].update(date="2026-10-20"),
            lambda b: b.update(pairs=[]),
        ):
            bundle = copy.deepcopy(self.bundle)
            mutate(bundle)
            bundle["design_sha256"] = base_round.dataset_hash(base_pilot.design(bundle))
            self.assertFalse(base_pilot.check(bundle, root=self.root)["passed"])
        path = Path(self.bundle["reports"]["validation"]["model-0"]["test"]["path"])
        path.write_text("{}")
        self.assertFalse(base_pilot.check(self.bundle, root=self.root)["passed"])

    def test_tied_ranks_and_exact_five_point_inversion(self):
        self.assertEqual(base_pilot.ranks([1, 1, 3]), [1.5, 1.5, 3.0])
        self.assertEqual(base_pilot.spearman([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(base_pilot.spearman([1, 2, 3], [1, 1, 1]), 0.0)
        result = base_pilot.ranking([20, 25, 30, 35, 40], [10, 9, 30, 35, 40])
        self.assertFalse(result["passed"])
        self.assertTrue(result["inversions"])
        self.assertFalse(base_pilot.ranking([21, 25, 30, 35, 40], [21, 25, 30, 35, 40])["passed"])
        self.assertFalse(base_pilot.ranking([20, 30, 40, 50], [20, 30, 40, 50])["passed"])

    def test_runtime_identity_excludes_weight_releases_but_pins_contract(self):
        original = models.pins(models.D3)
        changed = copy.deepcopy(original)
        changed["files"]["model-00001-of-00002.safetensors"] = "a" * 64
        changed["revision"] = "f" * 40
        with patch.object(models, "pins", return_value=changed):
            self.assertEqual(
                base_pilot.runtime_contract(models.D3), self.bundle["runtime_identity"]
            )
        changed["files"]["tokenizer.json"] = "d" * 64
        with patch.object(models, "pins", return_value=changed):
            self.assertNotEqual(
                base_pilot.runtime_contract(models.D3), self.bundle["runtime_identity"]
            )

    def test_direct_intake_cannot_bypass_pool_admission(self):
        from types import SimpleNamespace

        from zils import validator

        cfg = {
            "pool_id": "d3-mini",
            "family_history": "history.json",
            "base_tier": {"policy": base_round.DEFAULT_POLICY},
            "model": models.spec(models.D3),
        }
        draws = [(base_round.draw(self.root / "development"), self.root / "development")]
        with self.assertRaisesRegex(ValueError, "pilot_report"):
            validator.base_tier_epoch(cfg, self.root, self.root, {}, SimpleNamespace(), draws)
        cfg["pilot_report"] = "failed-pilot.json"
        save(self.root / cfg["pilot_report"], {"pool_id": "d3-mini", "passed": False})
        with self.assertRaisesRegex(ValueError, "failed|stale"):
            validator.base_tier_epoch(cfg, self.root, self.root, {}, SimpleNamespace(), draws)
