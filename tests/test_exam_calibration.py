"""Calibrated exam scoring: a temperature fitted on a held-apart part of the exam, proper scores
on the rest, and the pool setting that chooses between the scorings."""

import copy
import math
import unittest

from tests.test_base_metrics import exam, predictions
from zils import base_metrics, model_pools, soup_grade


def calibrated_noul(confidence=0.75, per_family=40):
    """A yes/no exam answered with `confidence` on the chosen side, right on exactly that share of
    each label: perfectly calibrated at temperature 1."""
    labels = ["false", "true"] * (per_family // 2)
    cases = exam("noul", labels)
    right = int(round(confidence * per_family / 2))
    rows = []
    for case in cases:
        index = int(case["id"].rsplit("-", 1)[1])
        correct = (index // 2) < right
        chosen = case["label"] if correct else ("true" if case["label"] == "false" else "false")
        p = {chosen: confidence, ("true" if chosen == "false" else "false"): 1 - confidence}
        rows.append({"id": case["id"], "probabilities": p, "elapsed_ms": 1})
    return cases, rows


def mixed_exam():
    """Every question type, with a calibrated model that is right on 70% of each family."""
    cases = []
    for kind, labels in (
        ("noul", ["false", "true"] * 20),
        ("choice", ["a", "b"] * 20),
        ("score", ["0", "1", "3", "4"] * 10),
    ):
        cases += exam(kind, labels, prefix=f"mix-{kind}")
    rows = []
    for case in cases:
        keys = (
            sorted(case["question"]["criteria"]) if case["question"]["type"] == "choice" else None
        )
        keys = keys or (["false", "true"] if case["question"]["type"] == "noul" else list("01234"))
        index = int(case["id"].rsplit("-", 1)[1])
        right = index % 10 < 7
        wrong = next(k for k in reversed(keys) if k != case["label"])
        chosen = case["label"] if right else wrong
        rest = (1 - 0.7) / (len(keys) - 1)
        rows.append(
            {
                "id": case["id"],
                "probabilities": {k: 0.7 if k == chosen else rest for k in keys},
                "elapsed_ms": 1,
            }
        )
    return cases, rows


class BrierMetricTest(unittest.TestCase):
    def test_perfect_confident_answers_score_100_and_constants_score_0(self):
        for kind in ("choice", "noul", "score"):
            cases = exam(kind)
            self.assertAlmostEqual(
                base_metrics.score(cases, predictions(cases), metric="brier")["score"], 100
            )
        cases = exam("noul")
        flat = predictions(cases, confidence=0.5)
        self.assertEqual(base_metrics.score(cases, flat, metric="brier")["score"], 0)

    def test_noul_reference_is_the_balanced_constant(self):
        # 9 false to 1 true: always saying 90% false is good overall Brier but no skill once each
        # label counts equally, as with accuracy.
        cases = exam("noul", ["false"] * 9 + ["true"])
        rows = predictions(cases, lambda c, i: "false", confidence=0.9)
        result = base_metrics.score(cases, rows, metric="brier")
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["families"][cases[0]["family"]]["baseline"], 0.5)

    def test_choice_reference_is_the_best_constant_distribution(self):
        cases = exam(labels=["a", "a", "a", "b"])
        family = cases[0]["family"]
        # the label frequencies (3/4, 1/4): Brier 2 * (1/4) * (3/4) = 0.375, below uniform's 0.5
        self.assertAlmostEqual(base_metrics.baselines(cases, metric="brier")[family], 0.375)
        rows = [
            {"id": c["id"], "probabilities": {"a": 0.75, "b": 0.25}, "elapsed_ms": 1} for c in cases
        ]
        self.assertAlmostEqual(base_metrics.score(cases, rows, metric="brier")["score"], 0)

    def test_argmax_rps_keeps_top_answers_and_scores_score_questions_by_rps(self):
        def rps(p, label):  # independent of base_metrics: cumulative squared error over K - 1
            total, cumulative = 0.0, 0.0
            for k in range(len(p) - 1):
                cumulative += p[k]
                total += (cumulative - (k >= label)) ** 2
            return total / (len(p) - 1)

        cases = exam("noul") + exam("score", prefix="s")  # score labels 0, 1, 3, 4 of 0..4
        rows = predictions(cases, confidence=0.6)
        result = base_metrics.score(cases, rows, metric="argmax_rps")
        score_cases = [c for c in cases if c["question"]["type"] == "score"]
        by_id = {r["id"]: r["probabilities"] for r in rows}
        raw = sum(
            rps([by_id[c["id"]][str(k)] for k in range(5)], int(c["label"])) for c in score_cases
        ) / len(score_cases)
        # Best constant: the empirical label distribution (a quarter each on 0, 1, 3, 4) or uniform.
        constant = min(
            sum(rps([0.25, 0.25, 0, 0.25, 0.25], y) for y in (0, 1, 3, 4)) / 4,
            sum(rps([0.2] * 5, y) for y in (0, 1, 3, 4)) / 4,
        )
        for family, values in result["families"].items():
            if family.endswith("noul"):
                self.assertEqual(values["skill"], 1)  # top answers, as argmax
                self.assertEqual(values["baseline"], 0.5)
            else:
                self.assertAlmostEqual(values["raw_metric"], raw)
                self.assertAlmostEqual(values["baseline"], constant)
                self.assertAlmostEqual(values["unclipped_skill"], 1 - raw / constant)
                # not the argmax exam's expected-value distance against the median
                self.assertNotAlmostEqual(values["baseline"], base_metrics.baselines(cases)[family])
        self.assertAlmostEqual(
            base_metrics.baselines(cases, metric="argmax_rps")[score_cases[0]["family"]], constant
        )

    def test_score_questions_use_ranked_probability_score(self):
        cases = exam("score", ["0", "4"] * 4)
        near = predictions(cases, lambda c, i: "1" if c["label"] == "0" else "3")
        far = predictions(cases, lambda c, i: "4" if c["label"] == "0" else "0")
        self.assertGreater(
            base_metrics.score(cases, near, metric="brier")["families"][cases[0]["family"]][
                "unclipped_skill"
            ],
            base_metrics.score(cases, far, metric="brier")["families"][cases[0]["family"]][
                "unclipped_skill"
            ],
        )

    def test_confidence_counts_for_brier_but_not_for_argmax(self):
        cases = exam("noul")
        sure, unsure = predictions(cases, confidence=0.99), predictions(cases, confidence=0.51)
        self.assertEqual(
            base_metrics.score(cases, sure)["score"], base_metrics.score(cases, unsure)["score"]
        )
        self.assertGreater(
            base_metrics.score(cases, sure, metric="brier")["score"],
            base_metrics.score(cases, unsure, metric="brier")["score"] + 50,
        )

    def test_only_scores_a_part_with_the_whole_exams_baselines(self):
        cases = exam(labels=["a", "b"] * 4)
        rows = predictions(cases, lambda c, i: "a" if i % 2 else c["label"])
        part = {c["id"] for i, c in enumerate(cases) if i % 2 == 0}
        result = base_metrics.score(cases, rows, only=part)
        self.assertEqual(result["score"], 100)
        self.assertEqual(result["families"][cases[0]["family"]]["baseline"], 0.5)
        self.assertEqual(result["families"][cases[0]["family"]]["count"], 4)

    def test_default_metric_is_unchanged_and_unknown_metric_fails(self):
        cases = exam()
        rows = predictions(cases, lambda c, i: "a")
        self.assertEqual(
            base_metrics.score(cases, rows), base_metrics.score(cases, rows, metric="argmax")
        )
        with self.assertRaises(ValueError):
            base_metrics.score(cases, rows, metric="logloss")

    def test_paired_improvement_takes_the_metric(self):
        cases = exam("noul")
        sure, unsure = predictions(cases, confidence=0.99), predictions(cases, confidence=0.6)
        argmax = base_metrics.paired_improvement(cases, unsure, sure, resamples=50, seed=1)
        brier = base_metrics.paired_improvement(
            cases, unsure, sure, resamples=50, seed=1, metric="brier"
        )
        self.assertEqual(argmax["improvement"], 0)
        self.assertGreater(brier["improvement"], 0)


class TemperatureTest(unittest.TestCase):
    def test_temper_keeps_the_answer_and_sums_to_one(self):
        cases, rows = mixed_exam()
        for T in (0.3, 1.0, 3.0):
            tempered = base_metrics.temper(cases, rows, T)
            for before, after in zip(rows, tempered, strict=True):
                self.assertEqual(before["id"], after["id"])
                p, q = before["probabilities"], after["probabilities"]
                self.assertAlmostEqual(sum(q.values()), 1)
                self.assertEqual(max(sorted(p), key=p.get), max(sorted(q), key=q.get))
                if T == 1.0:
                    for k in p:
                        self.assertAlmostEqual(p[k], q[k])

    def test_fit_recovers_the_temperature_of_an_overconfident_copy(self):
        cases, rows = calibrated_noul()
        self.assertAlmostEqual(base_metrics.fit_temperature(cases, rows), 1.0, delta=0.05)
        sharp = base_metrics.temper(cases, rows, 0.5)  # logits doubled
        self.assertAlmostEqual(base_metrics.fit_temperature(cases, sharp), 2.0, delta=0.1)
        soft = base_metrics.temper(cases, rows, 2.0)
        self.assertAlmostEqual(base_metrics.fit_temperature(cases, soft), 0.5, delta=0.03)

    def test_fit_prefers_one_when_temperature_changes_nothing(self):
        cases = exam("noul")
        hard = predictions(cases, lambda c, i: "false")  # 0/1 probabilities stay 0/1
        self.assertEqual(base_metrics.fit_temperature(cases, hard), 1.0)

    def test_fit_grid_reaches_far_beyond_the_old_kev_range(self):
        cases, rows = calibrated_noul()
        very_sharp = base_metrics.temper(cases, rows, 0.15)
        self.assertGreater(base_metrics.fit_temperature(cases, very_sharp), 4)


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.cases, _ = mixed_exam()

    def test_split_is_secret_dependent_disjoint_and_stratified(self):
        cal, test = base_metrics.calibration_split(self.cases, b"secret-1", 0.2)
        self.assertEqual(cal, base_metrics.calibration_split(self.cases, b"secret-1", 0.2)[0])
        self.assertNotEqual(cal, base_metrics.calibration_split(self.cases, b"secret-2", 0.2)[0])
        ids = {c["id"] for c in self.cases}
        self.assertEqual(cal | test, ids)
        self.assertFalse(cal & test)
        for family in {c["family"] for c in self.cases}:
            rows = [c for c in self.cases if c["family"] == family]
            noul = rows[0]["question"]["type"] == "noul"
            for label in {c["label"] for c in rows} if noul else {None}:
                stratum = [c["id"] for c in rows if label is None or c["label"] == label]
                held = sum(i in cal for i in stratum)
                self.assertEqual(held, math.floor(0.2 * len(stratum)))
                self.assertGreater(len(stratum) - held, 0)
        # order of the cases does not matter
        self.assertEqual(
            cal, base_metrics.calibration_split(list(reversed(self.cases)), b"secret-1", 0.2)[0]
        )

    def test_small_strata_stay_scored(self):
        cases = exam("noul", ["false"] * 9 + ["true"])
        cal, test = base_metrics.calibration_split(cases, b"s", 0.2)
        for case in cases:
            if case["label"] == "true":
                self.assertIn(case["id"], test)

    def test_split_rejects_bad_settings(self):
        for secret, share in ((b"", 0.2), ("text", 0.2), (b"s", 0), (b"s", 0.6), (b"s", True)):
            with self.assertRaises(ValueError):
                base_metrics.calibration_split(self.cases, secret, share)


class CalibratedSoupScoreTest(unittest.TestCase):
    def setUp(self):
        self.cases, self.rows = mixed_exam()

    def test_argmax_scoring_is_the_plain_exam_score(self):
        self.assertEqual(
            soup_grade.score(self.cases, self.rows, scoring="argmax", secret=b"s"),
            base_metrics.score(self.cases, self.rows)["score"],
        )

    def test_evaluate_reports_the_fit_and_scores_only_the_held_out_part(self):
        result = soup_grade.evaluate(
            self.cases, self.rows, scoring="calibrated_brier", secret=b"round"
        )
        cal, test = base_metrics.calibration_split(
            self.cases, b"round", soup_grade.CALIBRATION_SHARE
        )
        self.assertEqual(result["calibration_ids"], sorted(cal))
        self.assertEqual(sum(f["count"] for f in result["metrics"]["families"].values()), len(test))
        self.assertAlmostEqual(
            result["temperature"],
            base_metrics.fit_temperature([c for c in self.cases if c["id"] in cal], self.rows),
        )
        tempered = base_metrics.temper(self.cases, self.rows, result["temperature"])
        expected = base_metrics.score(self.cases, tempered, metric="brier", only=test)["score"]
        self.assertAlmostEqual(result["score"], expected)
        self.assertEqual(
            result["score"],
            soup_grade.score(self.cases, self.rows, scoring="calibrated_brier", secret=b"round"),
        )

    def test_temperature_twins_cannot_buy_points_once_calibrated(self):
        # The same answers at another confidence (an r5-style twin): under today's argmax rule the
        # Score families' expected value moves with temperature; calibrated scorings do not.
        for scoring in ("calibrated_brier", "calibrated_argmax"):
            base = soup_grade.score(self.cases, self.rows, scoring=scoring, secret=b"round")
            for T in (0.4, 0.7, 1.5, 2.5):
                twin = base_metrics.temper(self.cases, self.rows, T)
                self.assertAlmostEqual(
                    soup_grade.score(self.cases, twin, scoring=scoring, secret=b"round"),
                    base,
                    delta=0.15,
                    msg=f"{scoring} T={T}",
                )
        sharp = base_metrics.temper(self.cases, self.rows, 0.4)
        moved = soup_grade.score(self.cases, sharp, scoring="argmax") - soup_grade.score(
            self.cases, self.rows, scoring="argmax"
        )
        self.assertGreater(abs(moved), 0.3)

    def test_calibrated_brier_rewards_better_ranking_of_confidence(self):
        # Two models with the same answers; one puts its confidence where it is right.
        cases, rows = self.cases, self.rows
        informed = copy.deepcopy(rows)
        for case, row in zip(cases, informed, strict=True):
            p = row["probabilities"]
            top = max(sorted(p), key=p.get)
            sure = 0.95 if top == case["label"] else 0.55
            if len(p) == 2 or case["question"]["type"] != "score":
                rest = (1 - sure) / (len(p) - 1)
                row["probabilities"] = {k: sure if k == top else rest for k in p}

        def gain(scoring):
            return soup_grade.score(cases, informed, scoring=scoring, secret=b"r") - (
                soup_grade.score(cases, rows, scoring=scoring, secret=b"r")
            )

        # Top answers are the same; only Score families' RPS could move calibrated_argmax.
        self.assertAlmostEqual(gain("calibrated_argmax"), 0, delta=0.5)
        self.assertGreater(gain("calibrated_brier"), 10)

    def test_calibrated_scoring_needs_a_secret_and_a_known_name(self):
        with self.assertRaises(ValueError):
            soup_grade.score(self.cases, self.rows, scoring="calibrated_brier")
        with self.assertRaises(ValueError):
            soup_grade.score(self.cases, self.rows, scoring="brier", secret=b"s")

    def test_default_soup_scoring_is_calibrated_brier(self):
        self.assertEqual(soup_grade.DEFAULT_SCORING, "calibrated_brier")
        self.assertEqual(soup_grade.SCORINGS, ("argmax", "calibrated_argmax", "calibrated_brier"))


class PoolScoringSettingTest(unittest.TestCase):
    def entry(self, **changes):
        return {
            "type": "soup",
            "model": "d3-mini-4.5b-v1",
            "enabled": True,
            "share": 0.3,
            "house": [],
            "schedule": {"deadline_utc": "03:00"},
            **changes,
        }

    def test_default_and_explicit_scorings(self):
        # Unset, the validated entry is unchanged and rounds use the default scoring.
        self.assertNotIn("scoring", model_pools.validate_soup(self.entry()))
        for scoring in soup_grade.SCORINGS:
            self.assertEqual(
                model_pools.validate_soup(self.entry(scoring=scoring))["scoring"], scoring
            )
        for bad in ("brier", "", None, 1):
            with self.assertRaises(ValueError):
                model_pools.validate_soup(self.entry(scoring=bad))


if __name__ == "__main__":
    unittest.main()
