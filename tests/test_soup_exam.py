"""Round-bank subsets and the exam use ledger for soup rounds."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tests.test_base_metrics import exam, predictions
from zils import base_metrics, soup_exam


def bank():
    labels = {
        "choice": ["a", "b"] * 20,
        "noul": ["false"] * 34 + ["true"] * 6,
        "score": ["0", "1", "3", "4"] * 10,
    }
    return [row for kind in labels for row in exam(kind, labels[kind], prefix="bank")]


class SubsetTest(unittest.TestCase):
    def test_subset_keeps_every_family_and_the_grid(self):
        cases = bank()
        chosen = soup_exam.subset(cases, b"round-1", 500)
        self.assertEqual(len(chosen), 500)
        self.assertEqual(len({c["id"] for c in chosen}), 500)
        counts = Counter(c["family"] for c in chosen)
        self.assertEqual(set(counts), {c["family"] for c in cases})
        self.assertGreaterEqual(min(counts.values()), 10)
        for family in counts:
            if family.endswith("noul"):
                labels = {c["label"] for c in chosen if c["family"] == family}
                self.assertEqual(labels, {"false", "true"})
        result = base_metrics.score(chosen, predictions(chosen))
        self.assertAlmostEqual(result["score"], 100)

    def test_minimum_holds_for_small_families(self):
        small = exam("choice", ["a", "b"] * 6, prefix="small")
        for case in small:
            case["family"] = "small-" + case["family"]
        cases = bank() + small
        chosen = soup_exam.subset(cases, b"seed", 500)
        counts = Counter(c["family"] for c in chosen if c["id"].startswith("small"))
        self.assertEqual(set(counts.values()), {10})

    def test_same_seed_same_subset_in_stable_order(self):
        cases = bank()
        first = soup_exam.subset(cases, b"seed", 400)
        again = soup_exam.subset(list(reversed(cases)), b"seed", 400)
        self.assertEqual(first, again)
        self.assertEqual([c["id"] for c in first], sorted(c["id"] for c in first))

    def test_different_seeds_differ(self):
        cases = bank()
        first = {c["id"] for c in soup_exam.subset(cases, b"one", 400)}
        second = {c["id"] for c in soup_exam.subset(cases, b"two", 400)}
        self.assertNotEqual(first, second)

    def test_excluded_ids_are_never_sampled(self):
        cases = bank()
        retired = {c["id"] for c in cases[::3]}
        for seed in (b"a", b"b", b"c"):
            chosen = soup_exam.subset(cases, seed, 600, exclude=retired)
            self.assertFalse(retired & {c["id"] for c in chosen})

    def test_impossible_bank_raises(self):
        cases = bank()
        missing_cell = [c for c in cases if c["category"] != "arts"]
        with self.assertRaises(ValueError):
            soup_exam.subset(missing_cell, b"seed", 300)
        noul = {c["id"] for c in cases if c["family"].endswith("noul") and c["label"] == "true"}
        with self.assertRaises(ValueError):
            soup_exam.subset(cases, b"seed", 600, exclude=noul)
        with self.assertRaises(ValueError):
            soup_exam.subset(cases, b"seed", len(cases) + 1)
        with self.assertRaises(ValueError):
            soup_exam.subset(cases, b"seed", 100)
        with self.assertRaises(ValueError):
            soup_exam.subset(cases, "seed", 300)


class UseLedgerTest(unittest.TestCase):
    def test_record_is_idempotent_per_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "uses.jsonl"
            ledger = soup_exam.UseLedger(path)
            self.assertEqual(ledger.exposure("x"), 0)
            ledger.record("r1", ["x", "y", "x"])
            ledger.record("r1", ["x", "z"])
            ledger.record("r2", ["x"])
            self.assertEqual(ledger.exposure("x"), 2)
            self.assertEqual(ledger.exposure("y"), 1)
            self.assertEqual(ledger.exposure("z"), 0)
            self.assertEqual(len(path.read_text().splitlines()), 2)
            self.assertEqual(soup_exam.UseLedger(path).exposure("x"), 2)

    def test_retirement_at_cap_feeds_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = soup_exam.UseLedger(Path(tmp) / "uses.jsonl")
            for i in range(24):
                ledger.record(f"r{i}", ["old"] if i < 23 else ["old", "worn"])
            self.assertEqual(ledger.retired(), {"old"})
            self.assertEqual(ledger.retired(cap=1), {"old", "worn"})
            cases = bank()
            worn = {c["id"] for c in cases[::4]}
            for i, identity in enumerate(sorted(worn)):
                ledger.record(f"x{i}", [identity])
            chosen = soup_exam.subset(cases, b"s", 500, exclude=ledger.retired(cap=1))
            self.assertFalse(worn & {c["id"] for c in chosen})


if __name__ == "__main__":
    unittest.main()
