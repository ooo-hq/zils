"""Policy tests use trusted evidence, never a miner-supplied score."""

import copy
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from zils import miner_grading as grading, models

NOW = datetime(2026, 10, 9, tzinfo=UTC)
CONTEXT = {
    "model": models.JEVK5,
    **models.profile_identity(models.JEVK5),
    "trainer_sha256": "a" * 64,
    "rubric": "test/v1",
    "benchmark_sha256": "b" * 64,
    "band": "tokens-512",
}


def report(number, improvement=0.12, rate=0.1):
    return {
        "id": str(number),
        "context": dict(CONTEXT),
        "verified_at": (NOW - timedelta(days=1)).isoformat(),
        "expires_at": (NOW + timedelta(days=7)).isoformat(),
        "artifact_valid": True,
        "baseline_brier": 0.5,
        "candidate_brier": 0.5 - improvement,
        "uniform_brier": 1.0,
        "accuracy": 0.9,
        "min_accuracy": 0.8,
        "quality_floor": 0.0,
        "seconds_per_token": rate,
        "capacity": {"examples": 100, "total_tokens": 20000, "max_tokens": 512},
    }


def job(**overrides):
    return {
        "id": "job",
        "job_sha256": "c" * 64,
        "context": dict(CONTEXT),
        "workload": grading.workload_profile([100] * 10, model=models.JEVK5),
        "deadline": (NOW + timedelta(hours=2)).isoformat(),
        "consent": True,
        "purpose": "customer",
        **overrides,
    }


def worker(hotkey="worker", improvement=0.12, rate=0.1, **overrides):
    return {
        "hotkey": hotkey,
        "enabled": True,
        "approved": True,
        "resource_id": hotkey,
        "profile": models.profile_identity(models.JEVK5),
        "received_at": NOW.isoformat(),
        "ready": True,
        "reserved": False,
        "reports": [report(f"{hotkey}-{i}", improvement, rate) for i in range(3)],
        "attempts": [],
        "last_assigned_at": None,
        **overrides,
    }


class GradingTests(unittest.TestCase):
    def test_workload_frozen_in_new_manifest_and_legacy_unchanged(self):
        from tests.test_jobs import POLICY, examples
        from zils import jobs

        splits = examples()
        workload = grading.workload_profile([100] * len(splits["train"]), model=models.JEVK5)
        with tempfile.TemporaryDirectory() as tmp:
            old = jobs.build(
                Path(tmp) / "old",
                "old",
                splits,
                POLICY,
                model=models.JEVK5,
                allow_training_data_export=True,
            )
            self.assertNotIn("workload", old)
            jobs.audit(Path(tmp) / "old", old)
            new = jobs.build(
                Path(tmp) / "new",
                "new",
                splits,
                POLICY,
                model=models.JEVK5,
                allow_training_data_export=True,
                workload=workload,
            )
            self.assertEqual(new["workload"], workload)
            jobs.audit(Path(tmp) / "new", new)
            new["workload"]["examples"] += 1
            with self.assertRaises(ValueError):
                jobs.audit(Path(tmp) / "new", new)

    def test_quality_precedes_speed_and_deadline_is_hard_gate(self):
        slow = worker("slow", improvement=0.2, rate=0.2)
        fast = worker("fast", improvement=0.1, rate=0.04)
        for workers in ([fast, slow], [slow, fast]):
            self.assertEqual(
                grading.rank_candidates(job(), workers, NOW)["selected_hotkey"], "slow"
            )
        decision = grading.rank_candidates(
            job(deadline=(NOW + timedelta(seconds=180)).isoformat()), [slow, fast], NOW
        )
        self.assertEqual(decision["selected_hotkey"], "fast")
        self.assertEqual(decision["exclusions"][0]["reason"], "deadline")

    def test_workload_boundaries_and_invalid_lengths(self):
        for length, band in [
            (512, "tokens-512"),
            (513, "tokens-1024"),
            (1024, "tokens-1024"),
            (1025, "tokens-2048"),
        ]:
            self.assertEqual(grading.workload_profile([length], model=models.JEVK5)["band"], band)
        for lengths in ([], [0], [2049], [True], [1.5]):
            with self.assertRaises(ValueError):
                grading.workload_profile(lengths, model=models.JEVK5)

    def test_grade_requires_three_unique_comparable_recent_reports(self):
        reports = [report(i) for i in range(3)]
        self.assertEqual(grading.qualification_grade(reports, CONTEXT, NOW)["status"], "qualified")
        for key in CONTEXT:
            context = {**CONTEXT, key: "changed"}
            self.assertNotEqual(
                grading.qualification_grade(reports, context, NOW)["status"], "qualified"
            )
        for bad in [reports[:2], [reports[0]] * 3]:
            self.assertEqual(
                grading.qualification_grade(bad, CONTEXT, NOW)["status"], "provisional"
            )
        for r in reports:
            r["expires_at"] = NOW.isoformat()
        self.assertEqual(grading.qualification_grade(reports, CONTEXT, NOW)["status"], "expired")

    def test_invalid_evidence_cannot_create_grade(self):
        for key, value in [
            ("uniform_brier", 0),
            ("candidate_brier", float("nan")),
            ("accuracy", float("inf")),
            ("artifact_valid", False),
            ("seconds_per_token", -1),
            ("verified_at", "2099-01-01T00:00:00Z"),
        ]:
            reports = [report(i) for i in range(3)]
            for r in reports:
                r[key] = value
            self.assertNotEqual(
                grading.qualification_grade(reports, CONTEXT, NOW)["status"], "qualified"
            )

    def test_each_eligibility_gate(self):
        for changes, reason in [
            ({"enabled": False}, "disabled"),
            ({"approved": False}, "not_in_pool"),
            ({"resource_id": None}, "unmapped_resource"),
            ({"profile": {}}, "profile_mismatch"),
            ({"received_at": (NOW - timedelta(seconds=45)).isoformat()}, "offline"),
            ({"reserved": True}, "busy"),
            ({"ready": False}, "not_ready"),
            ({"cooldown_until": (NOW + timedelta(seconds=1)).isoformat()}, "cooldown"),
            ({"reports": []}, "provisional"),
        ]:
            with self.subTest(reason=reason):
                result = grading.rank_candidates(job(), [worker(**changes)], NOW)
                self.assertIsNone(result["selected_hotkey"])
                self.assertEqual(result["exclusions"][0]["reason"], reason)
        result = grading.rank_candidates(job(workload=None), [worker()], NOW)
        self.assertIsNone(result["selected_hotkey"])
        oversized = job()
        oversized["workload"] = grading.workload_profile([300] * 100, model=models.JEVK5)
        self.assertEqual(
            grading.rank_candidates(oversized, [worker()], NOW)["exclusions"][0]["reason"],
            "capacity",
        )

    def test_reliability_and_speed_are_separate_from_customer_acceptance(self):
        attempts = []
        for i, outcome in enumerate(
            [
                "valid",
                "invalid_artifact",
                "capacity_deferred",
                "validator_error",
                "cancelled",
                "pending_review",
            ]
        ):
            attempts.append(
                {
                    "id": str(i),
                    "context": CONTEXT,
                    "outcome": outcome,
                    "finished_at": NOW.isoformat(),
                    "started_at": (NOW - timedelta(seconds=200)).isoformat(),
                    "submitted_at": NOW.isoformat(),
                    "total_tokens": 1000,
                    "delivery_status": "no_qualifying_model",
                    "median_ms": 0.1,
                }
            )
        grade = grading.performance_grade(attempts, CONTEXT, NOW)
        self.assertEqual((grade["successes"], grade["failures"], grade["neutral"]), (1, 1, 4))
        self.assertEqual(grade["reliability"], 0.5)
        self.assertEqual(grade["p90_seconds_per_token"], 0.2)
        self.assertEqual(grading.performance_grade([], CONTEXT, NOW)["reliability"], 0.5)

    def test_deterministic_ties_inputs_immutable_and_hash_stable(self):
        workers = [worker("b"), worker("a")]
        original = copy.deepcopy(workers)
        first = grading.rank_candidates(job(), workers, NOW)
        second = grading.rank_candidates(job(), list(reversed(workers)), NOW)
        self.assertEqual(first, second)
        self.assertEqual(first["selected_hotkey"], "a")
        self.assertEqual(workers, original)
        self.assertEqual(len(first["snapshot_sha256"]), 64)

    def test_oldest_qualification_opportunity_and_no_grade_forgery(self):
        a, b = worker("a", reports=[]), worker("b", reports=[])
        # Provisional work still needs a verified capacity report (one run is sufficient).
        a["reports"] = [report("a")]
        b["reports"] = [report("b")]
        a["last_qualification_at"] = NOW.isoformat()
        b["quality_band"] = 100000
        result = grading.rank_candidates(job(purpose="qualification"), [a, b], NOW)
        self.assertEqual(result["selected_hotkey"], "b")
        self.assertIsNone(grading.rank_candidates(job(), [b], NOW)["selected_hotkey"])


if __name__ == "__main__":
    unittest.main()
