"""Independent oracles for full-native image scores and frozen acceptance targets."""

import copy
import math
import unittest

from zils import jobs
from zils.image_metrics import fit_image_temperature, image_policy_passes, score_rows

ORDER = ["normal", "damaged"]
POLICY = {
    "min_accuracy": 0.8,
    "min_brier_improvement": 0.01,
    "positive_class": "damaged",
    "min_positive_recall": 0.95,
    "max_false_positive_rate": 0.3,
}


class ImageMetricsTest(unittest.TestCase):
    def test_unknown_is_an_error_and_stays_in_brier(self):
        report = score_rows(
            ["damaged"],
            [{"normal": 0.2, "damaged": 0.3, "__unknown__": 0.5}],
            ["inspection"],
            ORDER,
        )
        self.assertEqual(report["accuracy"], 0)
        self.assertAlmostEqual(report["brier"], 0.78)
        self.assertEqual(report["uniform_brier"], 0.5)
        self.assertEqual(report["skill"], 0)
        self.assertEqual(report["unknown_rate"], 1)
        self.assertAlmostEqual(report["nll"], -math.log(0.3))
        self.assertEqual(report["per_class"]["damaged"]["recall"], 0)
        self.assertEqual(report["per_class"]["damaged"]["support"], 1)
        self.assertEqual(report["confusion"]["damaged"]["__unknown__"], 1)

    def test_equal_accuracy_tradeoff_follows_customer_policy(self):
        def candidate(recall, false_alarms, confidence):
            labels = ["damaged"] * 100 + ["normal"] * 100
            chosen = (
                ["damaged"] * recall
                + ["normal"] * (100 - recall)
                + ["damaged"] * false_alarms
                + ["normal"] * (100 - false_alarms)
            )
            probabilities = [
                {
                    k: (confidence if k == pick else (1 - confidence if k != "__unknown__" else 0))
                    for k in [*ORDER, "__unknown__"]
                }
                for pick in chosen
            ]
            return score_rows(labels, probabilities, ["inspection"] * 200, ORDER)

        a, b = candidate(97, 29, 0.95), candidate(87, 19, 0.75)
        self.assertEqual(a["accuracy"], b["accuracy"])
        self.assertLess(b["brier"], a["brier"])
        self.assertTrue(image_policy_passes(a, POLICY))
        self.assertFalse(image_policy_passes(b, POLICY))
        opposite = {**POLICY, "min_positive_recall": 0.85, "max_false_positive_rate": 0.2}
        self.assertFalse(image_policy_passes(a, opposite))
        self.assertTrue(image_policy_passes(b, opposite))
        baseline = {"status": "evaluated", "brier": 0.6, "accuracy": 0.5, "outcome_order": ORDER}
        rows = [
            {"uid": i + 1, "sha256": str(i) * 64, "status": "evaluated", **r}
            for i, r in enumerate((a, b))
        ]
        self.assertEqual(jobs.select(baseline, rows, POLICY, model="imajev-4b-v1")["uid"], 1)
        self.assertEqual(jobs.select(baseline, rows, opposite, model="imajev-4b-v1")["uid"], 2)

    def test_invalid_population_and_nonfinite_predictions_fail(self):
        distributions = [{"normal": 0.9, "damaged": 0.05, "__unknown__": 0.05}]
        report = score_rows(["normal"], distributions, ["inspection"], ORDER)
        self.assertFalse(image_policy_passes(report, POLICY))
        for labels, values, families in (
            (["normal"], [], ["inspection"]),
            ([], [], []),
            (["__unknown__"], distributions, ["inspection"]),
            (
                ["normal"],
                [{"normal": float("nan"), "damaged": 0, "__unknown__": 0}],
                ["inspection"],
            ),
        ):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                score_rows(labels, values, families, ORDER)

    def test_declared_tie_order_and_macro_family_weighting(self):
        p = {"damaged": 0.5, "normal": 0.5, "__unknown__": 0}
        a = score_rows(["normal"], [p], ["a"], ORDER)
        self.assertEqual(a["accuracy"], 1)
        rows = [p] * 9 + [{"normal": 0, "damaged": 1, "__unknown__": 0}]
        r = score_rows(["normal"] * 10, rows, ["a"] * 9 + ["b"], ORDER)
        self.assertAlmostEqual(r["brier"], 1.25)  # mean(.5, 2), not sample mean

    def test_grid_fitter_uses_full_unknown_logits_and_preserves_ranking(self):
        logits = [[2, 0, -1], [0, 2, -1]]
        t = fit_image_temperature(logits, [0, 1], ["a", "b"])
        self.assertEqual(t, 0.25)
        wrong = fit_image_temperature(logits, [1, 0], ["a", "b"])
        self.assertEqual(wrong, 4)
        with self.assertRaises(ValueError):
            fit_image_temperature([[float("inf"), 0, 0]], [0], ["a"])
        with self.assertRaises(ValueError):
            fit_image_temperature(logits, [0], ["a"])

    def test_image_prediction_order_and_calibration_split_are_frozen(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path

        import zils
        from tests.test_image_jobs import POLICY as IMAGE_POLICY, fixture
        from zils import calibrate, image_jobs, models
        from zils.image_metrics import probabilities_at_temperature

        job, splits, assets = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_jobs.build(root / "data", job, splits, assets, IMAGE_POLICY)
            checkpoint = root / "raw"
            checkpoint.mkdir()
            for name in models.IMAJEV_FILES:
                if name != "model.json":
                    (checkpoint / name).write_bytes(b"fixture")
            models.write_metadata(checkpoint, model=models.IMAJEV)
            cases = splits["calibration"]
            predictions = []
            for case in cases:
                logits = [2.0 if k == case["label"] else 0.0 for k in [*ORDER, "__unknown__"]]
                predictions.append(
                    {
                        "id": case["id"],
                        "logits": logits,
                        "elapsed_ms": 1,
                        "probabilities": dict(
                            zip([*ORDER, "__unknown__"], probabilities_at_temperature(logits, 1))
                        ),
                    }
                )
            with self.assertRaises(ValueError):
                zils.score(cases, list(reversed(predictions)))
            with self.assertRaises(ValueError):
                zils.score(cases, predictions[:1])
            report = {
                "base_revision": models.spec(models.IMAJEV)["base_revision"],
                "dataset_sha256": hashlib.sha256(
                    json.dumps(cases, sort_keys=True).encode()
                ).hexdigest(),
                "miners": [
                    {
                        "uid": 1,
                        "status": "evaluated",
                        "sha256": zils.checkpoint_hash(checkpoint),
                        "runtime": {
                            "temperature": 1,
                            "model": models.spec(models.IMAJEV),
                            **models.profile_identity(models.IMAJEV),
                        },
                        "predictions": predictions,
                        **zils.score(cases, predictions),
                    }
                ],
            }
            fitted = calibrate.fit(root / "data", report, 1, checkpoint, root / "fitted")
            self.assertEqual(fitted["temperature"], 0.25)
            self.assertEqual(fitted["fit_split"], "calibration")
            self.assertEqual(fitted["fit_cases"], 2)
            self.assertEqual(fitted["before"]["accuracy"], fitted["after"]["accuracy"])
            changed = copy.deepcopy(report)
            changed["dataset_sha256"] = hashlib.sha256(
                json.dumps(splits["test"], sort_keys=True).encode()
            ).hexdigest()
            with self.assertRaises(ValueError):
                calibrate.fit(root / "data", changed, 1, checkpoint, root / "bad")
