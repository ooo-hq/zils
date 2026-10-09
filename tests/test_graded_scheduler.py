"""Read-only preview and opt-in scheduling integration."""

import copy
import unittest
from datetime import timedelta
from unittest.mock import patch

from tests.test_miner_grading import CONTEXT, NOW, job, worker
from zils import graded_scheduler, miner_grading

CONFIG = {
    "mode": "graded",
    "policy_version": miner_grading.POLICY,
    "pool": ["pc", "mini"],
    "contexts": {"tokens-512": CONTEXT},
    "qualification_slots": 1,
}


class Store:
    def __init__(self, workers, job=None):
        self.snapshot = {
            "job": job or globals()["job"](),
            "workers": workers,
            "config": CONFIG,
            "state_sha256": "a" * 64,
        }
        self.writes, self.reads = [], 0
        self.retry = False

    def rpc(self, name, args):
        if name == "zils_grading_snapshot":
            self.reads += 1
            return copy.deepcopy(self.snapshot)
        if name == "zils_reserve_graded_job":
            self.writes.append(args)
            return {"status": "retry" if self.retry else "reserved"}
        raise AssertionError(name)


class SchedulerTests(unittest.TestCase):
    def test_workflow_routes_pending_text_through_grades(self):
        import tempfile
        from pathlib import Path

        from tests.test_workflow import Store as WorkflowStore
        from zils import models
        from zils.workflow import Workflow

        pending = {
            **job(),
            "status": "awaiting_approval",
            "result": None,
            "manifest": {
                "model": models.spec(models.JEVK5),
                "data_access": "approved-workers-training-export",
                "workload": job()["workload"],
            },
        }
        graded = Store([worker("pc", reserved=True), worker("mini")])

        class Combined(WorkflowStore):
            def rpc(self, name, args):
                if name in ("zils_reap_graded", "zils_next_qualification_job"):
                    return None
                return graded.rpc(name, args)

        with tempfile.TemporaryDirectory() as tmp:
            store = Combined(pending, Path(tmp))
            flow = Workflow(
                store, "pc", tmp, lambda: {"ready": False}, lambda _: None, routing=CONFIG
            )
            with patch("zils.workflow.datetime") as clock:
                clock.now.return_value = NOW
                flow.tick()
            self.assertEqual(graded.writes[0]["p_hotkey"], "mini")
            self.assertFalse(store.assigned)

    def test_preview_read_only_and_busy_pc_does_not_block_mini(self):
        store = Store([worker("pc", improvement=0.2, reserved=True), worker("mini")])
        scheduler = graded_scheduler.GradedScheduler(store, CONFIG)
        result = scheduler.preview({"id": "job"}, NOW)
        self.assertEqual(result["selected_hotkey"], "mini")
        self.assertEqual(store.writes, [])
        scheduler.assign({"id": "job"}, NOW)
        self.assertEqual(store.writes[0]["p_hotkey"], "mini")

    def test_each_platform_can_win_and_oversized_mini_is_excluded(self):
        for winner in ("pc", "mini"):
            workers = [worker(k, improvement=0.2 if k == winner else 0.1) for k in ("pc", "mini")]
            result = graded_scheduler.GradedScheduler(Store(workers), CONFIG).preview(
                {"id": "job"}, NOW
            )
            self.assertEqual(result["selected_hotkey"], winner)
        small = worker("mini", improvement=0.2)
        for report in small["reports"]:
            report["capacity"]["examples"] = 1
        result = graded_scheduler.GradedScheduler(Store([small, worker("pc")]), CONFIG).preview(
            {"id": "job"}, NOW
        )
        self.assertEqual(result["selected_hotkey"], "pc")

    def test_stale_snapshot_recomputed_once_and_no_candidate_never_reserves(self):
        store = Store([worker("mini")])
        store.retry = True
        scheduler = graded_scheduler.GradedScheduler(store, CONFIG)
        self.assertEqual(scheduler.assign({"id": "job"}, NOW)["status"], "retry")
        self.assertEqual((store.reads, len(store.writes)), (2, 2))
        store = Store([worker("mini", received_at=(NOW - timedelta(minutes=1)).isoformat())])
        self.assertEqual(
            graded_scheduler.GradedScheduler(store, CONFIG).assign({"id": "job"}, NOW)["status"],
            "waiting",
        )
        self.assertEqual(store.writes, [])

    def test_configuration_fails_closed_and_legacy_missing_metadata_is_explicit(self):
        for change in (
            {"policy_version": "unknown"},
            {"pool": []},
            {"qualification_slots": 2},
            {"contexts": {}},
        ):
            with self.assertRaises(ValueError):
                graded_scheduler.GradedScheduler(Store([]), {**CONFIG, **change})
        store = Store([worker("mini")])
        store.snapshot["config"] = {**CONFIG, "pool": ["pc"]}
        with self.assertRaises(ValueError):
            graded_scheduler.GradedScheduler(store, CONFIG).preview({"id": "job"}, NOW)
        store = Store([worker("mini")], job(workload=None))
        self.assertIsNone(
            graded_scheduler.GradedScheduler(store, CONFIG).preview({"id": "job"}, NOW)[
                "selected_hotkey"
            ]
        )

    def test_unconfigured_workload_context_is_an_exclusion(self):
        store = Store([worker("mini")], job(context=None))
        self.assertIsNone(
            graded_scheduler.GradedScheduler(store, CONFIG).preview({"id": "job"}, NOW)[
                "selected_hotkey"
            ]
        )

    def test_provisional_lane_only_uses_trusted_job_purpose(self):
        provisional = worker("mini")
        provisional["reports"] = provisional["reports"][:1]
        store = Store(
            [provisional],
            job(purpose="qualification", benchmark_sha256=CONTEXT["benchmark_sha256"]),
        )
        scheduler = graded_scheduler.GradedScheduler(store, CONFIG)
        self.assertEqual(
            scheduler.qualification_candidate({"id": "job"}, NOW)["selected_hotkey"], "mini"
        )
        store.snapshot["job"]["purpose"] = "customer"
        self.assertIsNone(
            scheduler.preview({"id": "job", "purpose": "qualification"}, NOW)["selected_hotkey"]
        )
        with self.assertRaises(ValueError):
            scheduler.qualification_candidate({"id": "job"}, NOW)
