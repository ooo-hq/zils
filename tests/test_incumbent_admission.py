"""An explicit incumbent contract admits a pool without changing election decisions."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import zils
from tests.test_base_exam import inputs
from tests.test_regression_suite import election_fixture
from zils import base_exam, base_pilot, base_release, base_round, models, validator


class IncumbentAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.prepare_pool("d3-lite", Path(self.tmp.name))

    def prepare_pool(self, pool, root):
        self.root = root
        self.root.mkdir(exist_ok=True)
        self.model = models.D3_POOL_MODELS[pool]
        self.pool = pool
        (self.root / "reference").mkdir()
        models.write_metadata(self.root / "reference", kind="base", model=self.model)
        self.incumbent = zils.checkpoint_hash(self.root / "reference")
        (self.root / "draws").mkdir()
        self.draws = []
        for name in ("first", "second"):
            path = self.root / "draws" / name
            cases, registry, history = inputs(name)
            base_exam.seal(
                **cases,
                registry=registry,
                blueprint=base_exam.blueprint(cases),
                pool_id=self.pool,
                history=history,
                sealed_at=20.0,
                out=path,
            )
            self.draws.append((base_round.draw(path), path))
        self.write("history.json", inputs()[2])
        self.config = {
            "pool_id": self.pool,
            "model": models.spec(self.model),
            "base_reference": "reference",
            "base_round_dir": "draws",
            "family_history": "history.json",
            "pilot_report": "incumbent-admission.json",
            "chain": {},
            "base_tier": {
                "share": 1.0,
                "incumbent_sha256": self.incumbent,
                "kings": [],
                "draws": [identity for identity, _ in self.draws],
                "policy": dict(base_round.DEFAULT_POLICY),
            },
        }
        self.capacity = {
            "model_sha256": self.incumbent,
            "runtime_sha256": base_round.dataset_hash(base_pilot.runtime_contract(self.model)),
            "settings_sha256": "b" * 64,
            "device": "synthetic-test-only",
            "peak_gpu_mib": 100,
            "available_gpu_mib": 200,
            "completed_at": 10,
            "probability_export_reviewed": True,
        }

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def contract(self, **overrides):
        return base_pilot.incumbent_admission(
            self.config, self.root, self.draws, capacity={**self.capacity, **overrides}
        )

    def test_explicit_contract_admits_without_claiming_pilot_passed(self):
        contract = self.contract()
        self.assertEqual(contract["version"], "zils-base-incumbent-challenge/v1")
        self.assertNotIn("passed", contract)
        self.write(self.config["pilot_report"], contract)
        base_pilot.admit(self.config, self.root, self.draws)
        self.assertEqual(contract["incumbent_sha256"], self.incumbent)

    def test_nano_flash_and_z3_require_their_own_admission_artifacts(self):
        lite_contract, lite_draws = self.contract(), self.draws
        lite_reference = str(self.root / "reference")
        for pool in ("d3-nano", "d3-flash", "d3"):
            with self.subTest(pool=pool):
                self.prepare_pool(pool, Path(self.tmp.name) / pool)
                contract = self.contract()
                self.write(self.config["pilot_report"], contract)
                base_pilot.admit(self.config, self.root, self.draws)
                self.assertEqual(contract["pool_id"], pool)
                self.assertEqual(contract["model"], models.spec(models.D3_POOL_MODELS[pool]))
                with self.assertRaisesRegex(ValueError, "pool"):
                    base_pilot.admit(self.config, self.root, lite_draws)
                with self.assertRaisesRegex(ValueError, "reference"):
                    base_pilot.admit(
                        {**self.config, "base_reference": lite_reference}, self.root, self.draws
                    )
                self.write(self.config["pilot_report"], lite_contract)
                with self.assertRaises(ValueError):
                    base_pilot.admit(self.config, self.root, self.draws)

    def test_wrong_weights_runtime_or_capacity_never_admit(self):
        for override in (
            {"model_sha256": "a" * 64},
            {"runtime_sha256": "a" * 64},
            {"peak_gpu_mib": 201},
            {"peak_gpu_mib": 0},
            {"probability_export_reviewed": False},
            {"device": ""},
            {"completed_at": float("inf")},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.contract(**override)

    def test_stale_or_ambiguous_contract_and_failed_legacy_report_reject(self):
        contract = self.contract()
        for change in (
            {"pool_id": "d3-mini"},
            {"incumbent_sha256": "a" * 64},
            {"draws": []},
            {"passed": True},
            {"policy": {**base_round.DEFAULT_POLICY, "min_score_improvement": 0}},
        ):
            self.write(self.config["pilot_report"], {**contract, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                base_pilot.admit(self.config, self.root, self.draws)
        self.write(self.config["pilot_report"], {"pool_id": self.pool, "passed": False})
        with self.assertRaisesRegex(ValueError, "failed|stale"):
            base_pilot.admit(self.config, self.root, self.draws)

    def test_two_independent_draws_and_real_incumbent_remain_required(self):
        for draws in (self.draws[:1], [self.draws[0], self.draws[0]]):
            with self.assertRaises(ValueError):
                base_pilot.incumbent_admission(
                    self.config, self.root, draws, capacity=self.capacity
                )
        config = copy.deepcopy(self.config)
        config["base_tier"]["incumbent_sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            base_pilot.incumbent_admission(config, self.root, self.draws, capacity=self.capacity)

    def test_validator_and_release_share_admission_contract(self):
        self.write(self.config["pilot_report"], self.contract())
        suite, _ = election_fixture()
        self.write("suite.json", suite)
        paths = {k: self.config[k] for k in base_release.PATHS if k != "suite_manifest"}
        paths["suite_manifest"] = "suite.json"
        entry = {**self.config["base_tier"], "model": models.spec(self.model), "enabled": True}
        config = {
            "chain": {},
            "base_pools": {"share": 0.5, "burn_uid": 0, "pools": {self.pool: entry}},
            "base_pool_rounds": {self.pool: paths},
        }
        self.assertIn(self.pool, validator.pool_rounds(config, self.root))
        active = base_release.active_context(config, self.pool, directory=self.root)
        self.assertEqual(active["incumbent_sha256"], self.incumbent)
        self.assertEqual(active["pilot_report"]["version"], "zils-base-incumbent-challenge/v1")

    def test_closing_checks_installed_contract_and_fresh_draw_before_inference(self):
        self.write(self.config["pilot_report"], self.contract())
        path = self.root / "closing"
        cases, registry, history = inputs("closing")
        base_exam.seal(
            **cases,
            registry=registry,
            blueprint=base_exam.blueprint(cases),
            pool_id=self.pool,
            history=history,
            sealed_at=20.0,
            out=path,
        )
        draw = (base_round.draw(path), path)

        def evaluate(draw, admission):
            config = copy.deepcopy(self.config)
            config["base_tier"]["draws"] = [draw[0]]
            return validator.base_tier_epoch(
                config,
                self.root,
                self.root,
                {},
                SimpleNamespace(),
                [draw],
                closing_admission=admission,
            )

        # Stop at the inference boundary; all admission and closing audits run for real.
        with patch.object(models, "text_runtime", side_effect=RuntimeError("inference reached")):
            with self.assertRaisesRegex(RuntimeError, "inference reached"):
                evaluate(draw, self.config)
            with self.assertRaises(ValueError):
                evaluate(self.draws[0], self.config)
            changed = copy.deepcopy(self.config)
            changed["base_tier"]["incumbent_sha256"] = "a" * 64
            with self.assertRaisesRegex(ValueError, "settings"):
                evaluate(draw, changed)
            self.write(self.config["pilot_report"], {"pool_id": self.pool, "passed": False})
            with self.assertRaisesRegex(ValueError, "failed|stale"):
                evaluate(draw, self.config)
