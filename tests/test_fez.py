"""Offline contract checks. Prediction fixtures are not model benchmarks."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LocalSubnetTest(unittest.TestCase):
    def test_validator_contract(self):
        self.assertIsNotNone(importlib.util.find_spec("fez"), "local validator is missing")
        import zils

        cases = [
            {
                "id": "one",
                "family": "policy",
                "state": "Approved.",
                "question": {"type": "noul", "instructions": "Is this approved?"},
                "label": "true",
            },
            {
                "id": "two",
                "family": "routing",
                "state": "Password reset.",
                "question": {
                    "type": "choice",
                    "instructions": "Which queue?",
                    "criteria": {"accounts": "Logins", "billing": "Payments"},
                },
                "label": "accounts",
            },
        ]
        zils.validate_cases(cases)
        good = [
            {"id": "one", "probabilities": {"false": 0.1, "true": 0.9}, "elapsed_ms": 10},
            {"id": "two", "probabilities": {"accounts": 0.8, "billing": 0.2}, "elapsed_ms": 20},
        ]
        uniform = copy.deepcopy(good)
        wrong = copy.deepcopy(good)
        for row in uniform:
            row["probabilities"] = dict.fromkeys(row["probabilities"], 0.5)
        wrong[0]["probabilities"] = {"false": 1.0, "true": 0.0}
        wrong[1]["probabilities"] = {"accounts": 0.0, "billing": 1.0}
        score = zils.score(cases, good)
        self.assertAlmostEqual(score["brier"], 0.05)
        self.assertAlmostEqual(score["skill"], 0.9)
        self.assertEqual(score["accuracy"], 1)
        self.assertEqual(score["p95_ms"], 20)
        self.assertEqual(zils.score(cases, uniform)["skill"], 0)
        self.assertEqual(zils.score(cases, wrong)["skill"], 0)
        self.assertEqual(zils.score(cases, wrong)["confident_errors"], 2)
        self.assertEqual(
            zils.weight_vector([{"uid": 4, **score}, {"uid": 9, **zils.score(cases, uniform)}]),
            {4: 1.0},
        )
        self.assertEqual(zils.weight_vector([{"uid": 4, **zils.score(cases, uniform)}]), {})

        # Corpus size cannot silently make one task family dominate the reward.
        extra = copy.deepcopy(cases[0])
        extra["id"] = "three"
        prediction = copy.deepcopy(good[0])
        prediction["id"] = "three"
        self.assertAlmostEqual(zils.score(cases + [extra], good + [prediction])["skill"], 0.9)
        self.assertEqual(zils.score(cases, list(reversed(good))), score)

        ordinal = {
            "id": "rating",
            "family": "rating",
            "state": "Medium.",
            "question": {"type": "score", "criteria": ["Low", "Medium", "High"]},
            "label": "1",
        }
        zils.validate_cases([ordinal])
        self.assertEqual(
            zils.score(
                [ordinal],
                [{"id": "rating", "probabilities": {"0": 0, "1": 1, "2": 0}, "elapsed_ms": 1}],
            )["skill"],
            1,
        )

        for bad in (
            {"false": float("nan"), "true": 0.9},
            {"false": -0.1, "true": 1.1},
            {"false": 0.2, "true": 0.2},
            {"true": 1},
            {"false": False, "true": True},
        ):
            rows = copy.deepcopy(good)
            rows[0]["probabilities"] = bad
            with self.subTest(probabilities=bad), self.assertRaises(ValueError):
                zils.score(cases, rows)
        for rows in (good[:1], good + good[:1]):
            with self.assertRaises(ValueError):
                zils.score(cases, rows)
        with self.assertRaises(ValueError):
            zils.validate_cases(cases + cases[:1])
        bad_case = copy.deepcopy(cases[0])
        bad_case["label"] = "maybe"
        with self.assertRaises(ValueError):
            zils.validate_cases([bad_case])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            for name in zils.ARTIFACT_FILES:
                (checkpoint / name).write_bytes(b"test-only-artifact")
            submission = zils.submission(checkpoint, 4)
            self.assertEqual(submission["uid"], 4)
            self.assertEqual(len(submission["sha256"]), 64)
            with self.assertRaises(ValueError):
                zils.submission(checkpoint, -1)
            with self.assertRaises(ValueError):
                zils.validate_submissions([submission, submission])
            # Verify an immutable copy, then detect changed source bytes.
            frozen = root / "frozen"
            zils.stage(submission, frozen)
            self.assertEqual(zils.checkpoint_hash(frozen), submission["sha256"])
            (checkpoint / "head.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                zils.stage(submission, root / "changed")
            self.assertEqual(zils.checkpoint_hash(frozen), submission["sha256"])

            report = root / "report.json"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([submission]))
            data = root / "cases.jsonl"
            data.write_text("\n".join(map(json.dumps, cases)))
            # Changed artifacts are disqualified before Kev can execute.
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "zils",
                    "evaluate",
                    "--submissions",
                    str(manifest),
                    "--cases",
                    str(data),
                    "--base-revision",
                    "a" * 40,
                    "--report",
                    str(report),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(report.read_text())
            self.assertEqual(output["mode"], "local-dry-run")
            self.assertEqual(output["weights"], {})
            self.assertEqual(output["miners"][0]["status"], "rejected")
            self.assertIn("hash", output["miners"][0]["error"])

            # Operator setup failures abort the round instead of scoring miners down.
            fresh = zils.submission(checkpoint, 4)
            manifest.write_text(json.dumps([fresh]))
            failed_report = root / "setup-failure.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "zils",
                    "evaluate",
                    "--submissions",
                    str(manifest),
                    "--cases",
                    str(data),
                    "--base-revision",
                    "a" * 40,
                    "--runner-python",
                    str(root / "missing-python"),
                    "--report",
                    str(failed_report),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("cannot start evaluation runtime", result.stderr)
            self.assertFalse(failed_report.exists())


if __name__ == "__main__":
    unittest.main()
