"""Analytical election scores: chance, ordinal error and paired uncertainty."""

import copy
import unittest

import zils
from zils import base_metrics

CATEGORIES = ("reasoning", "language", "retrieval", "tools", "arts")


def exam(kind="choice", labels=None, prefix="test"):
    labels = (
        labels
        or {
            "choice": ["a", "b"] * 4,
            "noul": ["false", "true"] * 4,
            "score": ["0", "1", "3", "4"] * 2,
        }[kind]
    )
    criteria = {"choice": {"a": "A", "b": "B"}, "noul": None, "score": list("01234")}
    result = []
    for category in CATEGORIES:
        for block in ("known", "new"):
            family = f"{category}-{block}-{kind}"
            for i, label in enumerate(labels):
                identity = f"{prefix}-{family}-{i}"
                result.append(
                    {
                        "id": identity,
                        "source_id": identity,
                        "family": family,
                        "category": category,
                        "block": block,
                        "state": {"text": identity},
                        "question": {
                            "type": kind,
                            "instructions": "Choose.",
                            "criteria": copy.deepcopy(criteria[kind]),
                        },
                        "label": label,
                    }
                )
    return result


def predictions(cases, choose=None, confidence=1.0):
    rows = []
    for i, case in enumerate(cases):
        keys = zils.options(case["question"])
        chosen = choose(case, i) if choose else case["label"]
        rows.append(
            {
                "id": case["id"],
                "probabilities": {
                    k: confidence if k == chosen else (1 - confidence) / (len(keys) - 1)
                    for k in keys
                },
                "elapsed_ms": 1,
            }
        )
    return rows


class BaseMetricsTest(unittest.TestCase):
    def test_choice_chance_correction_and_shared_score_unchanged(self):
        cases = exam(labels=["a", "b", "a", "b"])
        rows = predictions(cases, lambda c, i: "a" if i % 4 == 3 else c["label"])
        self.assertAlmostEqual(base_metrics.score(cases, rows)["score"], 50)
        shared = zils.score(cases, rows)
        self.assertEqual(shared["accuracy"], 0.75)
        self.assertEqual(shared["brier"], 0.5)

    def test_noul_balanced_accuracy_does_not_reward_majority(self):
        cases = exam("noul", ["false"] * 99 + ["true"])
        rows = predictions(cases, lambda c, i: "false")
        self.assertEqual(base_metrics.score(cases, rows)["score"], 0)

    def test_score_expected_value_uses_family_median_baseline(self):
        cases = exam("score", ["0", "1", "3", "4"])
        rows = predictions(cases)
        for i, row in enumerate(rows):
            value = [0.1, 0.35, 0.65, 0.9][i % 4]
            row["probabilities"] = {"0": 1 - value, "1": 0, "2": 0, "3": 0, "4": value}
        scored = base_metrics.score(cases, rows)
        self.assertAlmostEqual(scored["score"], 100 * (1 - 0.1 / 0.375))
        self.assertEqual(scored["families"][cases[0]["family"]]["baseline"], 0.375)
        self.assertEqual(base_metrics.score(cases, predictions(cases))["score"], 100)

    def test_strongest_position_and_missing_fixed_ids(self):
        cases = exam(labels=["a", "b", "a", "b"])
        for i, case in enumerate(cases):
            if i % 4 == 1:
                case["question"]["criteria"] = {"b": "B", "a": "A"}
        self.assertEqual(base_metrics.baselines(cases)[cases[0]["family"]], 0.75)
        for i, case in enumerate(cases):
            if i % 4 < 2:
                case["label"] = "a"
                case["question"]["criteria"] = {"a": "A", "b": "B"}
            else:
                case["label"] = "x"
                case["question"]["criteria"] = {"y": "Y", "x": "X"}
        self.assertEqual(base_metrics.baselines(cases)[cases[0]["family"]], 0.75)

    def test_category_block_and_family_weights_are_fixed(self):
        cases = exam()
        rows = predictions(
            cases,
            lambda c, i: c["label"] if c["category"] == "arts" and c["block"] == "new" else "a",
        )
        self.assertAlmostEqual(base_metrics.score(cases, rows)["score"], 4)

    def test_ties_and_clipped_losses_are_reported(self):
        cases = exam()
        ties = predictions(cases, confidence=0.5)
        self.assertEqual(base_metrics.score(cases, ties)["score"], 0)
        wrong = predictions(cases, lambda c, i: "b" if c["label"] == "a" else "a")
        result = base_metrics.score(cases, wrong)
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["families"][cases[0]["family"]]["unclipped_skill"], -1)

    def test_invalid_cases_and_predictions_fail_closed(self):
        good = exam()
        for change in (
            lambda rows: rows.pop(),
            lambda rows: rows[0].update(category="unknown"),
            lambda rows: rows[0].update(block="unknown"),
            lambda rows: rows[0].update(target_probabilities={"a": 1, "b": 0}),
        ):
            cases = copy.deepcopy(good)
            change(cases)
            if len(cases) != len(good):
                cases = [c for c in good if c["category"] != "arts"]
            with self.assertRaises(ValueError):
                base_metrics.score(cases, predictions(cases))
        for kind, labels in (("choice", ["a"] * 4), ("noul", ["true"] * 4), ("score", ["2"] * 4)):
            with self.assertRaises(ValueError):
                base_metrics.baselines(exam(kind, labels))
        for value in (float("nan"), -0.1, 1.1):
            rows = predictions(good)
            rows[0]["probabilities"]["a"] = value
            with self.assertRaises(ValueError):
                base_metrics.score(good, rows)
        with self.assertRaises(ValueError):
            base_metrics.score(good, predictions(good)[:-1])

    def test_identical_and_clipping_degenerate_pairs_have_zero_variance(self):
        cases = exam()
        a = predictions(cases, lambda c, i: "a")
        b = predictions(cases, lambda c, i: "b" if c["label"] == "a" else "a")
        for other in (a, b):
            pair = base_metrics.paired_improvement(cases, a, other, resamples=100, seed=8)
            if other is a:
                self.assertEqual(
                    (pair["improvement"], pair["lower_95"], pair["upper_95"]), (0, 0, 0)
                )
        # Uniform probabilities always tie to 'a'; two calibrated confidence levels are identical decisions.
        same = predictions(cases, lambda c, i: "a", confidence=0.7)
        pair = base_metrics.paired_improvement(cases, a, same, resamples=100, seed=8)
        self.assertEqual(pair["bootstrap_variance"], 0)

        for case in cases:
            case["question"]["criteria"]["c"] = "C"
        wrong_a = predictions(cases, lambda c, i: "b" if c["label"] == "a" else "a")
        wrong_b = predictions(cases, lambda c, i: "c")
        pair = base_metrics.paired_improvement(cases, wrong_a, wrong_b, resamples=100, seed=8)
        self.assertEqual(pair["bootstrap_variance"], 0)
        self.assertEqual(pair["improvement"], 0)

    def test_bootstrap_freezes_original_baselines_for_single_label_resamples(self):
        cases = exam(labels=["a", "b"])
        pair = base_metrics.paired_improvement(
            cases, predictions(cases, lambda c, i: "a"), predictions(cases), resamples=100, seed=1
        )
        self.assertEqual(pair["improvement"], 100)
        self.assertGreater(pair["bootstrap_variance"], 0)

    def test_paired_sampling_is_reproducible_and_stratifies_binary_labels(self):
        cases = exam("noul", ["false"] * 9 + ["true"])
        inc = predictions(cases, lambda c, i: "false")
        cand = predictions(cases, lambda c, i: c["label"] if i % 10 in (0, 9) else "true")
        pair = base_metrics.paired_improvement(cases, inc, cand, resamples=100, seed=5)
        self.assertGreater(pair["bootstrap_variance"], 0)
        self.assertAlmostEqual(pair["improvement"], 100 / 9)
        self.assertEqual(
            pair,
            base_metrics.paired_improvement(
                list(reversed(cases)),
                list(reversed(inc)),
                list(reversed(cand)),
                resamples=100,
                seed=5,
            ),
        )
        for n in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                base_metrics.paired_improvement(cases, inc, cand, resamples=n, seed=5)


if __name__ == "__main__":
    unittest.main()
