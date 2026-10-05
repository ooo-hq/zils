"""Checks for wrong oracle boundaries, leaked splits, and mismatched evaluation reports."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import zils


class BenchmarkTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("zils.benchmark"), "benchmark builder is missing"
        )
        from zils import benchmark

        return benchmark

    def test_rule_boundaries_have_independent_answers(self):
        b = self.module()
        for values, expected in [
            ((10, 10, True, False, False), "true"),
            ((11, 10, True, False, False), "false"),
            ((5, 10, False, True, False), "true"),
            ((5, 10, None, True, False), "true"),
            ((5, 10, True, True, True), "false"),
            ((5, 10, True, False, None), "false"),
            ((None, 10, True, False, False), "false"),
        ]:
            self.assertEqual(b.policy_answer(*values), expected)
        self.assertEqual(
            b.routing_answer(["delivery", "billing"], ["security", "billing", "delivery"]),
            "billing",
        )
        for facts, value, negated, expected in [
            ({"color": "blue"}, "blue", False, "supported"),
            ({"color": "blue"}, "red", False, "contradicted"),
            ({}, "blue", False, "unknown"),
            ({"color": "blue"}, "red", True, "supported"),
            ({"color": "blue"}, "blue", True, "contradicted"),
            ({}, "red", True, "unknown"),
        ]:
            self.assertEqual(b.evidence_answer(facts, "color", value, negated), expected)
        for value, override, expected in [
            (9, False, "0"),
            (10, False, "1"),
            (19, False, "1"),
            (20, False, "2"),
            (0, True, "2"),
        ]:
            self.assertEqual(b.severity_answer(value, 10, 20, override), expected)

    def test_frozen_splits_reject_leaks_and_changes(self):
        b = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmark"
            b.build(root, seed=553)
            splits = b.audit(root)
            self.assertEqual(
                {k: len(v) for k, v in splits.items()},
                {"train": 224, "calibration": 112, "test": 224},
            )
            self.assertEqual(len({c["group_id"] for c in splits["test"]}), 112)
            self.assertEqual(len({c["scenario_id"] for c in splits["test"]}), 16)
            for family in ("policy", "routing", "evidence", "severity"):
                rows = [c for c in splits["test"] if c["family"] == family]
                self.assertEqual(
                    set(c["label"] for c in rows), set(zils.options(rows[0]["question"]))
                )
            training = b.read_jsonl(root / "miner-training.jsonl")
            self.assertEqual(len(training), 224)
            self.assertTrue(all(set(row) == {"state", "questions"} for row in training))
            self.assertTrue(all("label" in row["questions"]["decision"] for row in training))
            for row in training:
                question = row["questions"]["decision"]
                expected_type = {"noul": bool, "choice": str, "score": int}[question["type"]]
                self.assertIs(
                    type(question["label"]),
                    expected_type,
                    "training labels must use Kev's native types",
                )
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)
            self.assertEqual((root / "test.jsonl").stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                b.build(root, seed=554)

            # Rewording IDs cannot sneak a scenario or exact prompt into another split.
            leaked = copy.deepcopy(splits)
            leaked["test"][0]["scenario_id"] = leaked["train"][0]["scenario_id"]
            with self.assertRaisesRegex(ValueError, "scenario"):
                b.validate_splits(leaked)
            leaked = copy.deepcopy(splits)
            origin = next(c for c in leaked["train"] if c["family"] == "policy")
            target = next(c for c in leaked["test"] if c["family"] == "policy")
            target["state"] = origin["state"]
            target["question"] = origin["question"]
            with self.assertRaisesRegex(ValueError, "duplicate|overlap"):
                b.validate_splits(leaked)
            corrupt = copy.deepcopy(splits)
            first = corrupt["test"][0]
            first["label"] = next(k for k in zils.options(first["question"]) if k != first["label"])
            with self.assertRaisesRegex(ValueError, "oracle"):
                b.validate_splits(corrupt)
            (root / "test.jsonl").write_text((root / "test.jsonl").read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                b.audit(root)

    def test_summary_checks_dataset_and_scores_variants(self):
        b = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmark"
            b.build(root, seed=2)
            cases = b.audit(root)["test"]
            predictions = [
                {
                    "id": c["id"],
                    "elapsed_ms": 10,
                    "probabilities": {
                        k: float(k == c["label"]) for k in zils.options(c["question"])
                    },
                }
                for c in cases
            ]
            report = {
                "dataset_sha256": hashlib.sha256(
                    json.dumps(cases, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
                "miners": [
                    {
                        "uid": 1,
                        "status": "evaluated",
                        "predictions": predictions,
                        "runtime": {"fixture": True},
                    }
                ],
            }
            summary = b.summarize(root, report, "test")
            row = summary["miners"][0]
            self.assertEqual(row["overall"]["skill"], 1)
            self.assertEqual(row["pair_agreement"], 1)
            self.assertEqual(row["both_variants_correct"], 1)
            self.assertEqual(row["by_family"]["evidence"]["accuracy"], 1)
            self.assertEqual(row["by_variant"]["clean"]["cases"], 112)
            stress = next(c for c in cases if c["variant"] != "clean")
            prediction = next(p for p in predictions if p["id"] == stress["id"])
            wrong = next(k for k in prediction["probabilities"] if k != stress["label"])
            prediction["probabilities"] = {
                k: float(k == wrong) for k in prediction["probabilities"]
            }
            changed = b.summarize(root, report, "test")["miners"][0]
            self.assertAlmostEqual(changed["pair_agreement"], 111 / 112)
            self.assertAlmostEqual(changed["both_variants_correct"], 111 / 112)
            report["dataset_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "dataset"):
                b.summarize(root, report, "test")


if __name__ == "__main__":
    unittest.main()
