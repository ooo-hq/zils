"""Independent validator budgets, fair intake and recovery without reward-path GPU work."""

import copy
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import zils
from tests.promoted_profile import promoted_profile
from tests.test_base_exam import inputs, sealed
from tests.test_base_tier_intake import RUNNER, candidate, fake_hub, revision
from tests.test_pool_rewards import pools
from zils import (
    base_queue as queue,
    base_round,
    commitments,
    model_pools,
    models,
    protocol as wire,
    validator,
)
from zils.runtime import CapacityUnavailable


def config():
    return {
        "base_pools": pools(),
        "base_pool_rounds": {},
        "base_validation_queue": {
            "window_seconds": 3600,
            "max_candidates": 4,
            "max_seconds": 1800,
            "max_download_bytes": 24 * 1024**3,
            "max_cache_bytes": 100 * 1024**3,
        },
    }


def snapshot(count):
    return {
        "members": {str(i): f"miner-{i}" for i in range(1, count + 1)},
        "commitments": {
            str(i): {
                "block": 100 + i,
                "data": commitments.format_commitment(f"{i:064x}", "a" * 40, f"owner/miner-{i}"),
            }
            for i in range(1, count + 1)
        },
    }


class BaseQueueTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(promoted_profile())
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = config()

    def test_real_queue_fetch_score_confirmation_and_cached_repeat(self):
        root = self.root
        (root / "draws").mkdir()
        for name in ("first", "second"):
            sealed(root / "draws" / name, name)
        (root / "history.json").write_text(json.dumps(inputs()[2]))
        (root / "pilot.json").write_text("{}")
        reference = root / "reference"
        reference.mkdir()
        models.write_metadata(reference, kind="base", model=models.D3)
        challenger = candidate(root / "challenger", 0.85)
        cfg = {
            "chain": {"network": "test", "netuid": 579},
            "base_pools": {
                "share": 0.5,
                "burn_uid": 0,
                "pools": {
                    "d3-mini": {
                        "model": models.D3,
                        "enabled": True,
                        "share": 1,
                        "incumbent_sha256": zils.checkpoint_hash(reference),
                        "kings": [],
                        "draws": [base_round.draw(root / "draws" / n) for n in ("first", "second")],
                    }
                },
            },
            "base_pool_rounds": {
                "d3-mini": {
                    "base_round_dir": "draws",
                    "base_reference": "reference",
                    "family_history": "history.json",
                    "pilot_report": "pilot.json",
                    "suite_manifest": "unused-until-release.json",
                }
            },
            "base_validation_queue": self.cfg["base_validation_queue"],
        }
        snap = snapshot(1)
        snap["commitments"]["1"]["data"] = commitments.format_commitment(
            zils.checkpoint_hash(challenger), revision("strong"), "alice/d3"
        )
        runner = root / "runner-python"
        runner.write_text(f"#!{sys.executable}\n{RUNNER}")
        runner.chmod(0o700)
        log = root / "runner.log"
        log.touch()
        requested = []
        args = SimpleNamespace(device="cpu", runtime_python=sys.executable, poll=0.1)
        with (
            wire.local_server(
                fake_hub(
                    {"alice/d3": {revision("strong"): challenger}},
                    requested,
                    truncate=["model-00001-of-00002.safetensors"] * 3,
                )
            ) as hub,
            patch("zils.commitments.HF_ENDPOINT", hub),
            patch("zils.base_pilot.verify_qualification"),
            patch("zils.d3.validate_candidate"),
            patch("zils.runtime.gpu_ready", return_value=True),
            patch("zils.base_queue.shutil.disk_usage", return_value=SimpleNamespace(free=10**12)),
            patch.dict(
                os.environ,
                {
                    "ZILS_D3_RUNTIME_PYTHON": str(runner),
                    "ZILS_TEST_RUNNER_LOG": str(log),
                    "ZILS_COMPUTE_LOCK": str(root / "compute.lock"),
                },
            ),
        ):
            queue.enqueue(cfg, root, snap)
            with (
                patch("zils.base_queue.time.time", return_value=1000),
                patch("zils.commitments.fetch", side_effect=URLError("offline")),
            ):
                self.assertTrue(queue.step(cfg, root, args))
            with queue.state_at(root) as state:
                self.assertEqual(state["entries"]["miner-1"]["status"], "pending")
                self.assertEqual(state["decisions"], {})
            with patch("zils.base_queue.time.time", return_value=4600):
                self.assertTrue(queue.step(cfg, root, args))
            with queue.state_at(root) as state:
                row = state["entries"]["miner-1"]
                self.assertEqual(row["status"], "pending")
                self.assertIn("truncated artifact", row["error"])
                self.assertIn("model-00001-of-00002.safetensors", row["error"])
                self.assertEqual(row["ready_at"], 8200)
                self.assertEqual(state["decisions"], {})
                self.assertEqual(state["budget"]["download_bytes"], models.MAX_CANDIDATE_BYTES)
            self.assertEqual(log.read_text(), "")
            self.assertFalse(list((root / "base-pools").rglob("fetch-*")))
            self.assertFalse(list((root / "base-pools").rglob("checkpoint")))
            requested.clear()
            with patch("zils.base_queue.time.time", return_value=8199):
                self.assertFalse(queue.step(cfg, root, args))
            with patch("zils.base_queue.time.time", return_value=8200):
                self.assertTrue(queue.step(cfg, root, args))
            saved = queue.decisions(cfg, root)
            self.assertIn("d3-mini", saved, (root / "base-queue/state.json").read_text())
            first = saved["d3-mini"]
            self.assertEqual(first["pending_winner"]["uid"], 1)
            with patch("zils.base_queue.time.time", return_value=8201):
                self.assertFalse(queue.step(cfg, root, args))
            with patch("zils.base_queue.time.time", return_value=51400):
                self.assertTrue(queue.step(cfg, root, args))
            confirmed = queue.decisions(cfg, root)["d3-mini"]
            self.assertEqual(confirmed["confirmed_winner"]["uid"], 1)
            queue.enqueue(cfg, root, snap)
            self.assertFalse(queue.step(cfg, root, args))
            self.assertEqual(
                sorted(name for _, name in requested if name != "model.json"),
                sorted(name for name in models.D3_CANDIDATE_FILES if name != "model.json"),
            )
            self.assertEqual(len(log.read_text().splitlines()), 8)
            self.assertEqual(
                model_pools.weights(
                    cfg["base_pools"], {"d3-mini": confirmed}, {"1": {"hotkey": "miner-1"}}
                )[0],
                {0: 1.0},
            )

    def test_256_miners_are_covered_under_persistent_budget(self):
        queue.enqueue(self.cfg, self.root, snapshot(256))
        seen = []
        for window in range(64):
            for _ in range(4):
                job = queue.reserve(self.cfg, self.root, 1000 + window * 3600)
                self.assertIsNotNone(job)
                seen.append(job["claim"]["hotkey"])
                queue.finish(self.cfg, self.root, job, 10)
            self.assertIsNone(queue.reserve(self.cfg, self.root, 1001 + window * 3600))
        self.assertEqual(len(set(seen)), 256)
        self.assertEqual(seen, [f"miner-{i}" for i in range(1, 257)])

    def test_two_validators_keep_independent_queues_and_budgets(self):
        other = self.root / "other"
        other.mkdir()
        for root in (self.root, other):
            queue.enqueue(self.cfg, root, snapshot(2))
        first = queue.reserve(self.cfg, self.root, 1000)
        queue.finish(self.cfg, self.root, first, 1800)
        self.assertIsNone(queue.reserve(self.cfg, self.root, 1001))
        self.assertEqual(queue.reserve(self.cfg, other, 1001)["claim"], first["claim"])

    def test_churn_keeps_position_and_started_nomination_is_frozen(self):
        snap = snapshot(3)
        queue.enqueue(self.cfg, self.root, snap)
        snap["commitments"]["1"]["data"] = commitments.format_commitment(
            "b" * 64, "b" * 40, "owner/new"
        )
        queue.enqueue(self.cfg, self.root, snap)
        job = queue.reserve(self.cfg, self.root, 1000)
        self.assertEqual(job["claim"]["sha256"], "b" * 64)
        snap["commitments"]["1"]["data"] = commitments.format_commitment(
            "d" * 64, "b" * 40, "owner/next"
        )
        queue.enqueue(self.cfg, self.root, snap)
        queue.finish(self.cfg, self.root, job, 5, ready_at=1000 + 12 * 3600)
        with queue.state_at(self.root) as state:
            self.assertEqual(state["entries"]["miner-1"]["claim"], job["claim"])
        second = queue.reserve(self.cfg, self.root, 1001)
        self.assertEqual(second["claim"]["hotkey"], "miner-2")
        queue.finish(self.cfg, self.root, second, 5)

    def test_lost_worker_keeps_charged_budget_and_moves_behind_waiters(self):
        queue.enqueue(self.cfg, self.root, snapshot(2))
        queue.reserve(self.cfg, self.root, 1000)
        self.assertIsNone(queue.reserve(self.cfg, self.root, 1001))
        recovered = queue.reserve(self.cfg, self.root, 4600)
        self.assertEqual(recovered["claim"]["hotkey"], "miner-2")

    def test_uid_reassignment_and_stale_completion_cannot_inherit_decision(self):
        snap = snapshot(1)
        queue.enqueue(self.cfg, self.root, snap)
        job = queue.reserve(self.cfg, self.root, 1000)
        snap["members"]["1"] = "new-owner"
        snap["commitments"]["1"]["data"] = commitments.format_commitment(
            "e" * 64, "b" * 40, "owner/new"
        )
        queue.enqueue(self.cfg, self.root, snap)
        queue.finish(self.cfg, self.root, job, 5, result=("d3-mini", {}))
        self.assertEqual(queue.decisions(self.cfg, self.root), {})
        with queue.state_at(self.root) as state:
            self.assertNotIn("miner-1", state["entries"])
            self.assertFalse(state["entries"]["new-owner"]["frozen"])

    def test_old_checkpoint_cannot_be_reclaimed_after_owner_changes_nomination(self):
        snap = snapshot(1)
        original = snap["commitments"]["1"]["data"]
        queue.enqueue(self.cfg, self.root, snap)
        snap["commitments"]["1"]["data"] = commitments.format_commitment(
            "e" * 64, "b" * 40, "owner/new"
        )
        snap["members"]["2"] = "copier"
        snap["commitments"]["2"] = {"block": 200, "data": original}
        queue.enqueue(self.cfg, self.root, snap)
        with queue.state_at(self.root) as state:
            self.assertNotIn("copier", state["entries"])

    def test_same_block_copy_invalidates_a_frozen_nomination(self):
        snap = snapshot(1)
        queue.enqueue(self.cfg, self.root, snap)
        job = queue.reserve(self.cfg, self.root, 1000)
        snap["members"]["2"] = "copier"
        snap["commitments"]["2"] = dict(snap["commitments"]["1"])
        queue.enqueue(self.cfg, self.root, snap)
        queue.finish(self.cfg, self.root, job, 5, result=("d3-mini", {}))
        with queue.state_at(self.root) as state:
            self.assertEqual(state["entries"], {})
            self.assertEqual(state["decisions"], {})

    def test_context_change_invalidates_results_without_resetting_budget(self):
        queue.enqueue(self.cfg, self.root, snapshot(1))
        job = queue.reserve(self.cfg, self.root, 1000)
        changed = copy.deepcopy(self.cfg)
        changed["base_pool_rounds"] = {"new": "exam"}
        queue.enqueue(changed, self.root, snapshot(1))
        queue.finish(self.cfg, self.root, job, 1800, result=("d3-mini", {}))
        self.assertEqual(queue.decisions(changed, self.root), {})
        self.assertIsNone(queue.reserve(changed, self.root, 1001))

    def test_download_allowance_is_reserved_before_io_and_not_refunded(self):
        queue.enqueue(self.cfg, self.root, snapshot(2))
        first = queue.reserve(self.cfg, self.root, 1000)
        queue.reserve_download(self.cfg, self.root, first, 20 * 1024**3)
        queue.finish(self.cfg, self.root, first, 10)
        second = queue.reserve(self.cfg, self.root, 1001)
        with self.assertRaisesRegex(queue.BudgetExhausted, "download"):
            queue.reserve_download(self.cfg, self.root, second, 20 * 1024**3)
        with queue.state_at(self.root) as state:
            self.assertEqual(state["budget"]["download_bytes"], 20 * 1024**3)

    def test_weight_path_has_no_candidate_http_or_evaluation_and_preserves_kings(self):
        work = self.root / "round"
        work.mkdir()
        snap = snapshot(256)
        with (
            patch("zils.commitments.route_pools", side_effect=AssertionError("HTTP")),
            patch("zils.validator.base_tier_epoch", side_effect=AssertionError("GPU")),
        ):
            validator.base_pools_epoch(self.cfg, self.root, work, snap, None, {})
            validator.base_pools_epoch(self.cfg, self.root, work, snap, None, {})
            saved = validator.attach_base_pools(self.cfg, self.root, work, {})
        members = {"1": {"hotkey": "a"}, "2": {"hotkey": "b"}}
        actual = model_pools.weights(
            self.cfg["base_pools"], saved["base_pools"]["decisions"], members
        )
        self.assertEqual(actual[0], {1: 0.25, 2: 0.75})
        self.assertEqual(
            json.loads((work / "pool-intake.json").read_text())["queue"]["queued"], 256
        )

    def test_clock_rollback_does_not_reset_allowance(self):
        queue.enqueue(self.cfg, self.root, snapshot(2))
        job = queue.reserve(self.cfg, self.root, 1000)
        queue.finish(self.cfg, self.root, job, 1800)
        self.assertIsNone(queue.reserve(self.cfg, self.root, 999))

    def test_busy_gpu_defers_without_exposing_or_erasing_existing_results(self):
        queue.enqueue(self.cfg, self.root, snapshot(2))
        with patch("zils.base_queue.evaluate", side_effect=CapacityUnavailable()):
            self.assertTrue(queue.step(self.cfg, self.root, None))
        with queue.state_at(self.root) as state:
            row = state["entries"]["miner-1"]
            self.assertEqual(row["status"], "pending")
            self.assertTrue(row["frozen"])
            self.assertIsNone(state["active"])

    def test_http_outage_retries_but_invalid_revision_completes(self):
        for code, status in ((503, "pending"), (429, "pending"), (404, "done")):
            with self.subTest(code=code):
                root = self.root / str(code)
                root.mkdir()
                queue.enqueue(self.cfg, root, snapshot(1))
                with patch(
                    "zils.base_queue.evaluate",
                    side_effect=HTTPError("fixture", code, "error", {}, None),
                ):
                    queue.step(self.cfg, root, None)
                with queue.state_at(root) as state:
                    self.assertEqual(state["entries"]["miner-1"]["status"], status)

    def test_corrected_artifact_pointer_can_retry_without_repeating_a_scored_hash(self):
        snap = snapshot(1)
        queue.enqueue(self.cfg, self.root, snap)
        job = queue.reserve(self.cfg, self.root, 1000)
        queue.finish(self.cfg, self.root, job, 5, error="HTTP 404")
        snap["commitments"]["1"]["data"] = commitments.format_commitment(
            job["claim"]["sha256"], "b" * 40, "owner/repaired"
        )
        queue.enqueue(self.cfg, self.root, snap)
        repaired = queue.reserve(self.cfg, self.root, 1001)
        self.assertEqual(repaired["claim"]["repo"], "owner/repaired")
        queue.finish(self.cfg, self.root, repaired, 5)
        snap["commitments"]["1"]["data"] = commitments.format_commitment(
            job["claim"]["sha256"], "d" * 40, "owner/unchanged-weights"
        )
        queue.enqueue(self.cfg, self.root, snap)
        self.assertIsNone(queue.reserve(self.cfg, self.root, 1002))

    def test_hard_deadline_is_not_swallowed_as_an_artifact_error(self):
        self.assertFalse(issubclass(queue.EvaluationDeadline, (ValueError, OSError, RuntimeError)))
        with self.assertRaises(queue.EvaluationDeadline), queue.deadline(0.01):
            time.sleep(1)

    def test_cache_eviction_preserves_private_evidence_and_removes_abandoned_fetch(self):
        store = self.root / "base-pools/d3-mini/base-tier/store"
        old = store / ("a" * 64) / "checkpoint"
        old.mkdir(parents=True)
        (old / "weights").write_bytes(b"x" * 30)
        evidence = old.parent / "evaluation.json"
        evidence.write_text("private receipt")
        partial = old.parent / ("fetch-" + "b" * 32)
        partial.mkdir()
        (partial / "weights").write_bytes(b"x")
        target = store / ("c" * 64) / "checkpoint"
        with patch("zils.base_queue.shutil.disk_usage", return_value=SimpleNamespace(free=10**12)):
            queue.make_room(self.root, target, 30, 40)
        self.assertFalse(old.exists())
        self.assertFalse(partial.exists())
        self.assertEqual(evidence.read_text(), "private receipt")

    def test_cache_budget_does_not_evict_a_configured_reference(self):
        store = self.root / "base-pools/d3-mini/base-tier/store"
        reference = store / ("a" * 64) / "checkpoint"
        reference.mkdir(parents=True)
        (reference / "weights").write_bytes(b"x" * 30)
        target = store / ("b" * 64) / "checkpoint"
        with self.assertRaises(queue.BudgetExhausted):
            queue.make_room(self.root, target, 30, 40, protected={reference.resolve()})
        self.assertTrue((reference / "weights").exists())

    def test_invalid_policies_fail_before_intake(self):
        for value in (0, -1, True, 1.5):
            cfg = copy.deepcopy(self.cfg)
            cfg["base_validation_queue"]["max_candidates"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                queue.enqueue(cfg, self.root, snapshot(1))
        cfg = copy.deepcopy(self.cfg)
        cfg["base_validation_queue"]["max_download_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "fit"):
            queue.policy(cfg)


if __name__ == "__main__":
    unittest.main()
