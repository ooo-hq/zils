import json
import math
import os
import tempfile
import unittest

from zils import serving_share

CFG = {
    "share": 0.1,
    "scores": "serving-scores.json",
    "switch_on_miners": 3,
    "max_age_seconds": 3600,
}
NOW = 1_000_000.0


def entry(score):
    return {
        "rounds": 1,
        "pass_rate": 1,
        "availability": 1,
        "median_latency_s": 1,
        "struck_until": 0,
        "score": score,
    }


class ValidateTest(unittest.TestCase):
    def test_accepts_and_returns_normalized(self):
        self.assertEqual(serving_share.validate(dict(CFG)), CFG)

    def test_rejects_bad_config(self):
        bad = [
            {**CFG, "extra": 1},
            {k: v for k, v in CFG.items() if k != "share"},
            {**CFG, "share": -0.1},
            {**CFG, "share": 1.1},
            {**CFG, "share": math.nan},
            {**CFG, "share": math.inf},
            {**CFG, "scores": ""},
            {**CFG, "scores": 5},
            {**CFG, "switch_on_miners": 0},
            {**CFG, "switch_on_miners": 1.5},
            {**CFG, "max_age_seconds": 59},
            {**CFG, "max_age_seconds": 86401},
            {**CFG, "max_age_seconds": 60.0},
            [],
        ]
        for config in bad:
            with self.subTest(config=config), self.assertRaises(ValueError):
                serving_share.validate(config)


class BudgetTest(unittest.TestCase):
    def pools(self, share):
        return {"share": share, "burn_uid": 0, "pools": {}}

    def config(self, base, feature=None, serving=0.10):
        out = {"base_pools": self.pools(base), "serving_share": {**CFG, "share": serving}}
        if feature is not None:
            out["feature_tracks"] = self.pools(feature)
        return out

    def check(self, config):
        from unittest import mock

        with (
            mock.patch("zils.model_pools.validate", lambda c: c),
            mock.patch("zils.feature_tracks.validate", lambda c: c),
        ):
            serving_share.validate_budget(config)

    def test_noop_without_block(self):
        serving_share.validate_budget({})

    def test_requires_base_pools(self):
        with self.assertRaises(ValueError):
            serving_share.validate_budget({"serving_share": dict(CFG)})

    def test_refuses_legacy_base_tier(self):
        c = self.config(0.63)
        c["base_tier"] = {}
        with self.assertRaises(ValueError):
            serving_share.validate_budget(c)

    def test_refuses_over_budget(self):
        with self.assertRaises(ValueError):
            self.check(self.config(0.63, 0.30))

    def test_accepts_within_budget(self):
        self.check(self.config(0.63))


class RealBudgetTest(unittest.TestCase):
    def setUp(self):
        from tests.promoted_profile import promoted_profile

        self.enterContext(promoted_profile())

    def config(self, base, feature, serving):
        from tests.test_feature_rewards import feature_config, fixture

        config = fixture()[0]
        config["base_pools"] = {**config["base_pools"], "share": base}
        config["feature_tracks"] = {**feature_config(), "share": feature}
        config["serving_share"] = {**CFG, "share": serving}
        return config

    def test_real_validators(self):
        serving_share.validate_budget(self.config(0.63, 0, 0.10))
        with self.assertRaises(ValueError):
            serving_share.validate_budget(self.config(0.63, 0.30, 0.10))


class FreezeTest(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.path = os.path.join(d.name, "scores.json")
        self.cfg = {**CFG, "scores": self.path}

    def write(self, data, age=0):
        with open(self.path, "w") as f:
            f.write(data if isinstance(data, str) else json.dumps(data))
        os.utime(self.path, (NOW - age, NOW - age))

    def test_missing(self):
        self.assertEqual(
            serving_share.freeze(self.cfg, NOW),
            {"written_at": None, "frozen_at": NOW, "scores": {}},
        )

    def test_corrupt_and_non_dict(self):
        for text in ("{nope", "[1]"):
            self.write(text)
            self.assertEqual(serving_share.freeze(self.cfg, NOW)["scores"], {})

    def test_nan_outside_score_is_dropped_and_freeze_is_json_safe(self):
        self.write('{"a": {"m": {"score": 1.0, "pass_rate": NaN}}}')
        frozen = serving_share.freeze(self.cfg, NOW)
        self.assertEqual(frozen["scores"], {"a": {"m": {"score": 1.0}}})
        json.dumps(frozen, allow_nan=False)

    def test_future_mtime_is_not_fresh(self):
        self.write({"a": {"m": entry(1)}}, age=-3600)
        self.assertEqual(serving_share.freeze(self.cfg, NOW)["scores"], {})

    def test_stale(self):
        self.write({"a": {"m": entry(1)}}, age=3601)
        frozen = serving_share.freeze(self.cfg, NOW)
        self.assertEqual(frozen["scores"], {})
        self.assertEqual(frozen["written_at"], NOW - 3601)

    def test_fresh_drops_bad_scores(self):
        self.write(
            '{"a": {"ok": {"score": 1.5}, "neg": {"score": -1}, "nan": {"score": NaN}, "str": {"score": "9"}},'
            ' "b": {"x": {"score": -1}}}'
        )
        frozen = serving_share.freeze(self.cfg, NOW)
        self.assertEqual(frozen["written_at"], NOW)
        self.assertEqual(frozen["scores"], {"a": {"ok": {"score": 1.5}}})


class HostileFileTest(FreezeTest):
    def test_huge_int_score_dropped(self):
        self.write('{"a": {"big": {"score": 1' + "0" * 500 + '}, "ok": {"score": 1}}}')
        self.assertEqual(serving_share.freeze(self.cfg, NOW)["scores"], {"a": {"ok": {"score": 1}}})

    def test_bool_score_dropped(self):
        self.write({"a": {"t": {"score": True}}})
        self.assertEqual(serving_share.freeze(self.cfg, NOW)["scores"], {})

    def test_deep_nesting_freezes_empty(self):
        self.write("[" * 100000)
        self.assertEqual(serving_share.freeze(self.cfg, NOW)["scores"], {})

    def test_oversize_freezes_empty_but_keeps_mtime(self):
        from unittest import mock

        self.write({"a": {"m": entry(1)}})
        with mock.patch.object(serving_share, "MAX_SCORES_BYTES", 10):
            frozen = serving_share.freeze(self.cfg, NOW)
        self.assertEqual((frozen["scores"], frozen["written_at"]), ({}, NOW))


class WeightsTest(unittest.TestCase):
    members = {
        "1": {"hotkey": "a"},
        "2": {"hotkey": "b"},
        "3": {"hotkey": "c"},
        "4": {"hotkey": "d"},
    }

    def frozen(self, scores):
        return {"written_at": 1.0, "frozen_at": 2.0, "scores": scores}

    def test_below_switch_on(self):
        scores = {h: {m: entry(1) for m in "xyz"} for h in "ab"}
        vector, record = serving_share.weights(CFG, self.frozen(scores), self.members)
        self.assertEqual(vector, {})
        self.assertFalse(record["switched_on"])
        self.assertEqual(record["earners"], 2)

    def test_switched_on_sums_and_ignores_unregistered(self):
        scores = {h: {"x": entry(1), "y": entry(2), "z": entry(0.5)} for h in "abc"}
        scores["ghost"] = {"x": entry(9)}
        vector, record = serving_share.weights(CFG, self.frozen(scores), self.members)
        self.assertEqual(vector, {1: 3.5, 2: 3.5, 3: 3.5})
        self.assertTrue(record["switched_on"])
        self.assertEqual(record["earners"], 3)
        self.assertEqual(record["weights"], {"1": 3.5, "2": 3.5, "3": 3.5})
        self.assertEqual(
            (record["share"], record["written_at"], record["frozen_at"]), (0.1, 1.0, 2.0)
        )

    def test_non_finite_total_skipped(self):
        scores = {h: {"x": entry(1)} for h in "abc"}
        scores["d"] = {"x": entry(1e308), "y": entry(1e308)}
        vector, record = serving_share.weights(CFG, self.frozen(scores), self.members)
        self.assertEqual(sorted(vector), [1, 2, 3])
        self.assertEqual(record["earners"], 3)

    def test_zero_score_not_earner(self):
        scores = {"a": {"x": entry(1)}, "b": {"x": entry(1)}, "c": {"x": entry(0)}}
        vector, record = serving_share.weights(CFG, self.frozen(scores), self.members)
        self.assertEqual(vector, {})
        self.assertEqual(record["earners"], 2)


if __name__ == "__main__":
    unittest.main()


class AttachTest(unittest.TestCase):
    def setUp(self):
        from pathlib import Path

        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.work = Path(d.name)
        self.src = os.path.join(d.name, "src.json")
        self.cfg = {"serving_share": {**CFG, "scores": self.src}}
        self.report = {"miners": []}

    def test_unchanged_without_config(self):
        from zils import validator

        self.assertEqual(validator.attach_serving_scores({}, self.work, self.report), self.report)
        self.assertFalse((self.work / "serving-scores.json").exists())

    def test_frozen_on_first_call(self):
        from zils import validator

        with open(self.src, "w") as f:
            json.dump({"hk": {"m": entry(1.0)}}, f)
        first = validator.attach_serving_scores(
            self.cfg, self.work, self.report, os.stat(self.src).st_mtime
        )
        with open(self.src, "w") as f:
            json.dump({"hk": {"m": entry(9.0)}}, f)
        again = validator.attach_serving_scores(self.cfg, self.work, self.report, 5.0)
        self.assertEqual(again, first)
        self.assertEqual(first["serving_share"]["config"], self.cfg["serving_share"])
        self.assertEqual(first["serving_share"]["frozen"]["scores"]["hk"]["m"]["score"], 1.0)
        self.assertNotIn("serving_share", self.report)

    def test_missing_source_attaches_empty(self):
        from zils import validator

        out = validator.attach_serving_scores(self.cfg, self.work, self.report, NOW)
        self.assertEqual(out["serving_share"]["frozen"]["scores"], {})

    def test_publication_requires_frozen_file(self):
        from zils import validator

        (self.work / "report.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "serving-scores.json"):
            validator.publication_report(self.cfg, self.work, self.work)


class FleetBudgetTest(unittest.TestCase):
    def test_fleet_refuses_bad_serving_budget(self):
        import contextlib
        import io
        from pathlib import Path

        from zils import fleet, models

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "base_revision": models.spec(models.KEV)["base_revision"],
                        "serving_share": CFG,
                    }
                )
            )
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                fleet.main(["validator", "--config", str(config)])
            self.assertIn("serving_share requires base_pools", err.getvalue())
