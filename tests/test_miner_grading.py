"""Policy tests use trusted evidence, never a miner-supplied score."""

import copy
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from zils import miner_grading as grading, models

NOW = datetime(2026, 10, 9, tzinfo=UTC)
CONTEXT = {
    "model": models.H2O,
    **models.profile_identity(models.H2O),
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
        "workload": grading.workload_profile([100] * 10, model=models.H2O),
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
        "profile": models.profile_identity(models.H2O),
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
        workload = grading.workload_profile([100] * len(splits["train"]), model=models.H2O)
        with tempfile.TemporaryDirectory() as tmp:
            old = jobs.build(
                Path(tmp) / "old",
                "old",
                splits,
                POLICY,
                model=models.H2O,
                allow_training_data_export=True,
            )
            self.assertNotIn("workload", old)
            jobs.audit(Path(tmp) / "old", old)
            new = jobs.build(
                Path(tmp) / "new",
                "new",
                splits,
                POLICY,
                model=models.H2O,
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
            self.assertEqual(grading.workload_profile([length], model=models.H2O)["band"], band)
        for lengths in ([], [0], [2049], [True], [1.5]):
            with self.assertRaises(ValueError):
                grading.workload_profile(lengths, model=models.H2O)

    def test_workload_cap_follows_the_profile(self):
        self.assertEqual(grading.workload_profile([4096], model=models.D3)["band"], "tokens-4096")
        with self.assertRaisesRegex(ValueError, "1..4096"):
            grading.workload_profile([4097], model=models.D3)
        with self.assertRaisesRegex(ValueError, "1..2048"):
            grading.workload_profile([4096], model=models.H2O)
        self.assertNotEqual(
            grading.trainer_identity(models.D3), grading.trainer_identity(models.H2O)
        )
        d3 = grading.workload_profile([4096] * 2, model=models.D3)
        self.assertEqual(grading.validate_workload(d3, model=models.D3, examples=2), d3)
        with self.assertRaisesRegex(ValueError, "model mismatch|invalid workload tokens"):
            grading.validate_workload(d3, model=models.H2O, examples=2)

    def test_contexts_are_keyed_by_model_and_band(self):
        self.assertEqual(grading.context_key(CONTEXT), f"{models.H2O}/tokens-512")
        identity = {k: "f" * 64 for k in ("profile_sha256", "runtime_sha256")}
        d3 = {**CONTEXT, **identity, "model": models.D3, "band": "tokens-4096"}
        h2o = {**CONTEXT, **identity, "model": models.H2O, "band": "tokens-2048"}
        self.assertTrue(grading.valid_context(d3))
        # Stored H2O evidence stays readable; a band above H2O's cap was never produced.
        self.assertTrue(grading.valid_context(h2o))
        self.assertFalse(grading.valid_context({**h2o, "band": "tokens-4096"}))
        self.assertEqual(grading.bands(models.H2O), ("tokens-512", "tokens-1024", "tokens-2048"))
        self.assertEqual(grading.bands(models.D3)[-1], "tokens-4096")
        # The same band under another model never grades this context.
        attempt = {
            "id": "other-model",
            "context": {**CONTEXT, "model": models.D3},
            "outcome": "valid",
            "finished_at": NOW.isoformat(),
            "started_at": (NOW - timedelta(seconds=100)).isoformat(),
            "submitted_at": NOW.isoformat(),
            "total_tokens": 1000,
        }
        self.assertEqual(grading.performance_grade([attempt], CONTEXT, NOW)["successes"], 0)
        self.assertEqual(
            grading.qualification_grade(
                [{**report(1), "context": {**CONTEXT, "model": models.D3}}], CONTEXT, NOW
            )["quality_runs"],
            0,
        )
        crossed = job(workload=grading.workload_profile([100] * 10, model=models.D3))
        result = grading.rank_candidates(crossed, [worker()], NOW)
        self.assertIsNone(result["selected_hotkey"])
        self.assertEqual(result["exclusions"][0]["reason"], "unverified_workload_or_consent")

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

    def test_recent_low_quality_runs_reduce_grade_instead_of_disappearing(self):
        reports = [report(str(i)) for i in range(3)]
        for i in range(3, 6):
            bad = report(str(i), improvement=-0.2)
            bad["verified_at"] = NOW.isoformat()
            reports.append(bad)
        grade = grading.qualification_grade(reports, CONTEXT, NOW)
        self.assertNotEqual(grade["status"], "qualified")
        self.assertLess(grade["quality_band"], 0)

    def test_invalid_evidence_cannot_create_grade(self):
        for key, value in [
            ("expires_at", None),
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
        # Size beyond the measured probe is the miner's own fit check, not a routing exclusion.
        larger = job()
        larger["workload"] = grading.workload_profile([300] * 100, model=models.H2O)
        self.assertEqual(
            grading.rank_candidates(larger, [worker()], NOW)["selected_hotkey"], "worker"
        )
        # A job longer than the context's band ceiling is not this context's work.
        longer = job(workload=grading.workload_profile([1000], model=models.H2O))
        self.assertEqual(
            grading.rank_candidates(longer, [worker()], NOW)["exclusions"][0]["reason"],
            "unverified_workload_or_consent",
        )
        refused = {"id": "r", "job_id": "job", "outcome": "does_not_fit", "context": CONTEXT}
        result = grading.rank_candidates(job(), [worker("a", attempts=[refused]), worker("b")], NOW)
        self.assertEqual(result["selected_hotkey"], "b")
        self.assertEqual(result["exclusions"], [{"hotkey": "a", "reason": "does_not_fit"}])

    def test_band_is_a_ceiling_and_the_smallest_covering_context_wins(self):
        small, large = {**CONTEXT, "band": "tokens-512"}, {**CONTEXT, "band": "tokens-2048"}
        contexts = {grading.context_key(c): c for c in (large, small)}
        short = grading.workload_profile([100], model=models.H2O)
        long = grading.workload_profile([1500], model=models.H2O)
        self.assertIs(grading.covering_context(contexts, short), small)
        self.assertIs(grading.covering_context(contexts, long), large)
        self.assertIsNone(grading.covering_context({"x": small}, long))
        self.assertIsNone(
            grading.covering_context(contexts, grading.workload_profile([100], model=models.D3))
        )

    def test_training_memory_estimate_covers_measured_d3_peaks(self):
        self.assertGreaterEqual(grading.training_mib(models.D3, 1895), 10538)
        self.assertGreaterEqual(grading.training_mib(models.D3, 6931), 13036)
        self.assertLess(grading.training_mib(models.D3, 8192), 24 * 1024)
        self.assertIsNone(grading.training_mib(models.H2O, 1000))
        self.assertIsNone(grading.training_mib(models.D3, None))

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
        # Provisional work still needs one verified report.
        a["reports"] = [report("a")]
        b["reports"] = [report("b")]
        a["last_qualification_at"] = NOW.isoformat()
        b["quality_band"] = 100000
        result = grading.rank_candidates(job(purpose="qualification"), [a, b], NOW)
        self.assertEqual(result["selected_hotkey"], "b")
        self.assertIsNone(grading.rank_candidates(job(), [b], NOW)["selected_hotkey"])


if __name__ == "__main__":
    unittest.main()
