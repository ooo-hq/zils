import json
import math
import os
import tempfile
import unittest

from zils import serving_share

CFG = {"share": 0.1, "scores": "serving-scores.json", "switch_on_miners": 3, "max_age_seconds": 3600}
NOW = 1_000_000.0


def entry(score):
    return {"rounds": 1, "pass_rate": 1, "availability": 1, "median_latency_s": 1, "struck_until": 0, "score": score}


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

        with mock.patch("zils.model_pools.validate", lambda c: c), mock.patch(
            "zils.feature_tracks.validate", lambda c: c
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
            serving_share.freeze(self.cfg, NOW), {"written_at": None, "frozen_at": NOW, "scores": {}}
        )

    def test_corrupt_and_non_dict(self):
        for text in ("{nope", "[1]"):
            self.write(text)
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


class WeightsTest(unittest.TestCase):
    members = {"1": {"hotkey": "a"}, "2": {"hotkey": "b"}, "3": {"hotkey": "c"}, "4": {"hotkey": "d"}}

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
        self.assertEqual((record["share"], record["written_at"], record["frozen_at"]), (0.1, 1.0, 2.0))

    def test_zero_score_not_earner(self):
        scores = {"a": {"x": entry(1)}, "b": {"x": entry(1)}, "c": {"x": entry(0)}}
        vector, record = serving_share.weights(CFG, self.frozen(scores), self.members)
        self.assertEqual(vector, {})
        self.assertEqual(record["earners"], 2)


if __name__ == "__main__":
    unittest.main()
