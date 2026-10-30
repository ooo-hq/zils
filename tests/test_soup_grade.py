"""Soup grading: per-key prediction cache, case-ordered rows and the seeded screen subset."""

import tempfile
import unittest
from pathlib import Path

from tests.test_base_metrics import exam, predictions
from zils import base_metrics, soup_grade


class FakeRunner:
    """Predicts the label for every case and records how many cases each call received."""

    def __init__(self):
        self.calls = []

    def __call__(self, cases):
        self.calls.append([c["id"] for c in cases])
        return list(reversed(predictions(cases)))  # order must not matter to grade()


class PredictionCacheTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.cases = exam()
        self.cache = soup_grade.PredictionCache(self.root)

    def test_second_grade_of_same_key_runs_nothing(self):
        runner = FakeRunner()
        first = soup_grade.grade("body-a", self.cases, self.cache, runner)
        second = soup_grade.grade("body-a", self.cases, self.cache, runner)
        self.assertEqual(first, second)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(len(runner.calls[0]), len(self.cases))
        reopened = soup_grade.PredictionCache(self.root)
        third = soup_grade.grade("body-a", self.cases, reopened, runner)
        self.assertEqual(third, first)
        self.assertEqual(len(runner.calls), 1)

    def test_partial_cache_fills_only_the_rest(self):
        runner = FakeRunner()
        soup_grade.grade("body-a", self.cases[:10], self.cache, runner)
        soup_grade.grade("body-a", self.cases, self.cache, runner)
        self.assertEqual(runner.calls[1], [c["id"] for c in self.cases[10:]])

    def test_keys_are_separate(self):
        runner = FakeRunner()
        soup_grade.grade("body-a", self.cases, self.cache, runner)
        soup_grade.grade("soup-b", self.cases, self.cache, runner)
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(self.cache.get("soup-b", ["missing"]), {})

    def test_rows_in_case_order(self):
        shuffled = self.cases[::3] + self.cases[1::3] + self.cases[2::3]
        rows = soup_grade.grade("body-a", shuffled, self.cache, FakeRunner())
        self.assertEqual([r["id"] for r in rows], [c["id"] for c in shuffled])
        self.assertEqual(soup_grade.score(self.cases, rows), 100.0)

    def test_crash_truncated_last_line_is_ignored(self):
        runner = FakeRunner()
        soup_grade.grade("body-a", self.cases[:5], self.cache, runner)
        path = next(self.root.iterdir())
        with path.open("a") as handle:
            handle.write('{"id": "' + self.cases[5]["id"] + '", "probab')
        reopened = soup_grade.PredictionCache(self.root)
        self.assertEqual(len(reopened.get("body-a", [c["id"] for c in self.cases])), 5)
        rows = soup_grade.grade("body-a", self.cases, reopened, runner)
        self.assertEqual(runner.calls[1], [c["id"] for c in self.cases[5:]])
        self.assertEqual(len(rows), len(self.cases))
        again = soup_grade.PredictionCache(self.root)
        self.assertEqual(len(again.get("body-a", [c["id"] for c in self.cases])), len(self.cases))

    def test_runner_must_answer_every_case(self):
        with self.assertRaises(ValueError):
            soup_grade.grade("body-a", self.cases, self.cache, lambda cases: [])

    def test_unsafe_key_rejected(self):
        with self.assertRaises(ValueError):
            self.cache.put("../escape", [])


class ScoreTest(unittest.TestCase):
    def test_score_matches_base_metrics(self):
        cases = exam()
        rows = predictions(cases, choose=lambda case, i: "a")
        self.assertEqual(soup_grade.score(cases, rows), base_metrics.score(cases, rows)["score"])

    def test_screen_score_is_top_choice_accuracy(self):
        cases = exam()
        rows = predictions(cases, choose=lambda case, i: "a")
        self.assertEqual(soup_grade.screen_score(cases, rows), 0.5)


class ScreenSubsetTest(unittest.TestCase):
    def setUp(self):
        self.cases = exam(labels=["a", "b"] * 40)  # 10 families x 80 cases

    def test_deterministic_and_seed_dependent(self):
        first = soup_grade.screen_subset(self.cases, b"seed-1", size=50)
        self.assertEqual(first, soup_grade.screen_subset(self.cases, b"seed-1", size=50))
        self.assertEqual(
            first, soup_grade.screen_subset(list(reversed(self.cases)), b"seed-1", size=50)
        )
        self.assertNotEqual(first, soup_grade.screen_subset(self.cases, b"seed-2", size=50))
        self.assertEqual(len(first), 50)
        self.assertEqual(len(set(first)), 50)

    def test_stratified_by_family(self):
        chosen = set(soup_grade.screen_subset(self.cases, b"seed", size=50))
        families = {}
        for case in self.cases:
            if case["id"] in chosen:
                families[case["family"]] = families.get(case["family"], 0) + 1
        self.assertEqual(sorted(families.values()), [5] * 10)

    def test_small_bank_returns_everything(self):
        chosen = soup_grade.screen_subset(self.cases[:20], b"seed", size=300)
        self.assertEqual(sorted(chosen), sorted(c["id"] for c in self.cases[:20]))


if __name__ == "__main__":
    unittest.main()
