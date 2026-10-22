"""Pool intake routes one immutable parent, freezes decisions, and uses its own capacity."""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from miner.queue import fits
from scripts import promote_base, qualify_node
from scripts.migrate_model_pools import migrate
from tests import test_qualify_node
from tests.promoted_profile import PROMOTED, promoted_profile
from tests.test_base_exam import inputs, sealed
from tests.test_base_tier_intake import RUNNER, candidate, fake_hub, revision
from tests.test_pool_rewards import pools
from zils import base_round, commitments, miner_grading, models, protocol, validator


class PoolIntakeTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(promoted_profile())

    def test_promotion_regression_must_bind_weights_and_pass_every_job(self):
        report = {
            "candidate_sha256": "a" * 64,
            "incumbent_sha256": "b" * 64,
            "incumbent_model": models.D3,
            "candidate_model": models.D3,
            "jobs": {"task": {"incumbent": {"brier": 0.3}, "candidate": {"brier": 0.2}}},
        }
        promote_base.archival_verify_regression(report, "a" * 64, "b" * 64, models.D3)
        for changes in (
            {"candidate_sha256": "c" * 64},
            {"candidate_model": "d3-lite-0.8b-v1"},
            {"jobs": {}},
            {"jobs": {"task": {"incumbent": {"brier": 0.3}, "candidate": {"brier": float("nan")}}}},
            {"jobs": {"task": {"incumbent": {"brier": 0.3}, "candidate": {"brier": 0.31}}}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                promote_base.archival_verify_regression(
                    {**report, **changes}, "a" * 64, "b" * 64, models.D3
                )

    def test_staging_preserves_mini_incumbent_history_and_outer_budget(self):
        track = {
            k: v for k, v in pools()["pools"]["d3-mini"].items() if k not in ("model", "enabled")
        }
        cfg = {
            "model": models.spec(PROMOTED),
            "base_tier": {**track, "share": 0.7},
            "base_reference": "reference",
            "base_round_dir": "draws",
        }
        staged = migrate(cfg, "/operator")
        mini = staged["base_pools"]["pools"]["d3-mini"]
        self.assertEqual(mini["model"], cfg["model"])
        self.assertEqual(mini["kings"], track["kings"])
        self.assertEqual(mini["incumbent_sha256"], track["incumbent_sha256"])
        self.assertEqual(staged["base_pools"]["share"], 0.7)
        self.assertEqual(mini["share"], 1)
        self.assertEqual(len(staged["base_pools"]["pools"]), 5)
        self.assertTrue(
            all(
                not entry["enabled"] and entry["share"] == 0
                for key, entry in staged["base_pools"]["pools"].items()
                if key != "d3-mini"
            )
        )
        self.assertIn("base_tier", cfg)

    def test_parent_preflight_downloads_only_bounded_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = candidate(root / "candidate", 0.9)
            requested = []
            repos = {"test/base": {revision("base"): source}}
            with protocol.local_server(fake_hub(repos, requested)) as endpoint:
                claim = {"repo": "test/base", "revision": revision("base")}
                self.assertEqual(commitments.parent_model(claim, endpoint), models.D3)
            self.assertEqual(requested, [("test/base", "model.json")])
            with protocol.local_server(fake_hub(repos, [], huge={"test/base"})) as endpoint:
                with self.assertRaisesRegex(ValueError, "16 KiB"):
                    commitments.parent_model(claim, endpoint)

    def test_second_draw_waits_twelve_hours_and_reuses_first_scores(self):
        import zils

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference").mkdir()
            models.write_metadata(root / "reference", kind="base", model=models.D3)
            sealed(root / "first", "first")
            sealed(root / "second", "second")
            (root / "pilot.json").write_text(json.dumps({"fixture": True}))
            (root / "history.json").write_text(json.dumps(inputs()[2]))
            draws = [(base_round.draw(root / name), root / name) for name in ("first", "second")]
            config = {
                "pool_id": "d3-mini",
                "model": models.spec(models.D3),
                "base_reference": "reference",
                "family_history": "history.json",
                "pilot_report": "pilot.json",
                "base_tier": {
                    "share": 1,
                    "incumbent_sha256": zils.checkpoint_hash(root / "reference"),
                    "draws": [item for item, _ in draws],
                    "policy": dict(base_round.DEFAULT_POLICY),
                },
            }
            runner = root / "runner"
            runner.write_text(f"#!{sys.executable}\n{RUNNER}")
            runner.chmod(0o700)
            log = root / "calls"
            args = SimpleNamespace(device="cpu", runtime_python=sys.executable, poll=0.1)
            with (
                patch("zils.base_pilot.verify_qualification"),
                patch.dict(
                    os.environ,
                    {"ZILS_D3_RUNTIME_PYTHON": str(runner), "ZILS_TEST_RUNNER_LOG": str(log)},
                ),
            ):
                for name, now, count in (
                    ("round-1", 1000, 2),
                    ("round-2", 1001, 2),
                    ("round-3", 44200, 4),
                ):
                    work = root / name
                    work.mkdir()
                    with patch("zils.validator.time.time", return_value=now):
                        result = validator.base_tier_epoch(config, root, work, {}, args, draws)
                    self.assertEqual(len(log.read_text().splitlines()), count)
                    self.assertEqual(result["pool_id"], "d3-mini")
                self.assertIn("confirmed_winner", result)

    def test_failure_of_one_pool_does_not_drop_another_decision(self):
        cfg = {"base_pools": pools()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "round"
            work.mkdir()
            rounds = {}
            for pool in cfg["base_pools"]["pools"]:
                directory = root / "base-pools" / pool
                (directory / "base-tier").mkdir(parents=True)
                protocol.write_json(directory / "base-tier/decision.json", {"stale": True})
                rounds[pool] = ({"pool_id": pool}, directory, [])

            def evaluate(config, directory, *args, **kwargs):
                (directory / "base-tier/decision.json").unlink()
                if config["pool_id"] == "d3-mini":
                    raise RuntimeError("fixture GPU failure")
                protocol.write_json(directory / "base-tier/decision.json", {"pool_id": "d3-lite"})

            with patch("zils.validator.base_tier_epoch", side_effect=evaluate):
                validator.base_pools_epoch(cfg, root, work, {}, None, rounds)
            report = validator.attach_base_pools(cfg, root, work, {})
            self.assertEqual(report["base_pools"]["decisions"], {"d3-lite": {"pool_id": "d3-lite"}})
            self.assertIn("d3-mini", json.loads((work / "pool-intake.json").read_text())["errors"])

    def test_disabled_staging_needs_no_sealed_draw(self):
        from zils import model_pools

        cfg = pools()
        cfg["pools"]["d3-lite"].update(enabled=False, share=0, draws=[])
        cfg["pools"]["d3-mini"]["share"] = 1
        self.assertEqual(model_pools.weights(cfg, {}, {"1": {"hotkey": "b"}})[0], {1: 1})

    def test_routes_once_and_rejects_stale_parent_before_download(self):
        cfg = pools()
        registry = {1: {"sha256": "1" * 64}, 2: {"sha256": "2" * 64}, 3: {"sha256": "3" * 64}}
        with patch(
            "zils.commitments.parent_model",
            side_effect=["d3-lite-0.8b-v1", models.D3, PROMOTED],
        ):
            routed, rejected = commitments.route_pools(registry, cfg)
        self.assertEqual(set(routed["d3-lite"]), {1})
        self.assertEqual(set(routed["d3-mini"]), {3})
        self.assertIn("incumbent", rejected[2])

    def test_freezes_independent_pool_results_and_config(self):
        cfg = {"base_pools": pools()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "round"
            work.mkdir()
            for pool in cfg["base_pools"]["pools"]:
                path = root / "base-pools" / pool / "base-tier"
                path.mkdir(parents=True)
                protocol.write_json(path / "decision.json", {"pool_id": pool})
            saved = validator.attach_base_pools(cfg, root, work, {})
            (root / "base-pools/d3-lite/base-tier/decision.json").unlink()
            self.assertEqual(validator.attach_base_pools(cfg, root, work, {}), saved)
            changed = copy.deepcopy(cfg)
            changed["base_pools"]["pools"]["d3-lite"]["kings"] = []
            with self.assertRaisesRegex(ValueError, "config"):
                validator.attach_base_pools(changed, root, work, {})
            self.assertEqual(set(saved["base_pools"]["decisions"]), {"d3-mini", "d3-lite"})

    def test_confirmation_requires_time_profile_and_incumbent(self):
        winner = {"uid": 1, "sha256": "a" * 64}
        first = {
            "pending_winner": winner,
            "dataset_sha256": "1" * 64,
            "evaluated_at": 100.0,
            "pool_id": "d3-lite",
            "model": models.spec("d3-lite-0.8b-v1"),
            "incumbent": {"sha256": "b" * 64},
            "policy": dict(base_round.DEFAULT_POLICY),
        }
        second = {
            **first,
            "dataset_sha256": "2" * 64,
            "evaluated_at": 43300.0,
            "candidates": [{**winner, "eligible": True}],
        }
        self.assertEqual(base_round.confirm_promotion(first, second), winner)
        for changes in (
            {"evaluated_at": 43299.0},
            {"evaluated_at": float("nan")},
            {"pool_id": "d3-mini"},
            {"model": models.spec(models.D3)},
            {"incumbent": {"sha256": "c" * 64}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                base_round.confirm_promotion(first, {**second, **changes})

    def test_other_sizes_cannot_borrow_mini_training_curve(self):
        lite = "d3-lite-0.8b-v1"
        self.assertIsNone(miner_grading.training_mib(lite, 1000))
        self.assertFalse(fits(lite, {"max_tokens": 1000}, "cuda", budget_mib=24000))
        certificate = {
            "model": lite,
            **models.profile_identity(lite),
            "trainer_sha256": miner_grading.trainer_identity(lite),
            "capacity_ok": True,
            "runtime_verified": True,
            "reload_predict_ok": True,
            "training": {"max_prompt_tokens": 4096, "peak_gpu_mib": 4000},
        }
        self.assertTrue(
            fits(lite, {"max_tokens": 1000}, "cuda", budget_mib=6000, qualification=certificate)
        )
        self.assertFalse(
            fits(lite, {"max_tokens": 1000}, "cuda", budget_mib=4500, qualification=certificate)
        )
        self.assertFalse(
            fits(lite, {"max_tokens": 4097}, "cuda", budget_mib=9000, qualification=certificate)
        )
        self.assertFalse(
            fits(
                lite,
                {"max_tokens": 1000},
                "cuda",
                budget_mib=24000,
                qualification={**certificate, "model": models.D3},
            )
        )

    def test_image_fit_requires_its_own_measured_modality(self):
        lite = "d3-lite-0.8b-v1"
        certificate = {
            "model": lite,
            **models.profile_identity(lite),
            "trainer_sha256": miner_grading.trainer_identity(lite),
            "capacity_ok": True,
            "runtime_verified": True,
            "reload_predict_ok": True,
            "training": {"max_prompt_tokens": 4096, "peak_gpu_mib": 4000, "modalities": ["text"]},
        }
        self.assertFalse(
            fits(lite, None, "cuda", budget_mib=9000, qualification=certificate, modality="images")
        )
        certificate["training"]["modalities"].append("images")
        self.assertTrue(
            fits(lite, None, "cuda", budget_mib=9000, qualification=certificate, modality="images")
        )

    def test_qualification_commands_pin_selected_model(self):
        calls = []
        lite = "d3-lite-0.8b-v1"
        with tempfile.TemporaryDirectory() as tmp:
            qualify_node.qualify(
                Path.cwd(),
                "python",
                Path(tmp),
                model=lite,
                free_mib=19500,
                run=test_qualify_node.QualifyNodeTest().fake_run(calls),
            )
        verify = next(c for c in calls if any("verify_runtime" in str(v) for v in c))
        self.assertIn(lite, " ".join(verify))
        reference = next(c for c in calls if "reference" in c)
        self.assertEqual(reference[reference.index("--model") + 1], lite)
        sizing = next(c for c in calls if any("sizing probe" in str(v) for v in c))
        self.assertIn(lite, sizing)
        self.assertEqual(promote_base.CANDIDATE_ID_PREFIX[lite], "d3-lite-0.8b")

    def test_migration_stages_legacy_policy_disabled_until_qualified(self):
        cfg = {
            "model": models.spec(models.D3),
            "base_tier": {
                "share": 0.7,
                "incumbent_sha256": "a" * 64,
                "draws": [{"manifest_sha256": "b" * 64, "dataset_sha256": "c" * 64}],
                "kings": ["old"],
                "policy": {"min_brier_improvement": 0.01},
            },
            "base_reference": "reference",
            "base_round_dir": "draws",
        }
        result = migrate(cfg, "/operator")
        self.assertTrue(all(not e["enabled"] for e in result["base_pools"]["pools"].values()))
        self.assertEqual(
            result["base_pools"]["pools"]["d3-mini"]["policy"], base_round.DEFAULT_POLICY
        )
        self.assertEqual(result["base_pool_rounds"], {})
        self.assertEqual(cfg["base_tier"]["policy"], {"min_brier_improvement": 0.01})
