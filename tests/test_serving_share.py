import json
import math
import os
import tempfile
import unittest
from unittest import mock

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
        vector, _, record = serving_share.weights(CFG, self.frozen(scores), self.members)
        self.assertEqual(vector, {})
        self.assertFalse(record["switched_on"])
        self.assertEqual(record["earners"], 2)

    def test_switched_on_sums_and_ignores_unregistered(self):
        scores = {h: {"x": entry(1), "y": entry(2), "z": entry(0.5)} for h in "abc"}
        scores["ghost"] = {"x": entry(9)}
        vector, _, record = serving_share.weights(CFG, self.frozen(scores), self.members)
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
        vector, _, record = serving_share.weights(CFG, self.frozen(scores), self.members)
        self.assertEqual(sorted(vector), [1, 2, 3])
        self.assertEqual(record["earners"], 3)

    def test_zero_score_not_earner(self):
        scores = {"a": {"x": entry(1)}, "b": {"x": entry(1)}, "c": {"x": entry(0)}}
        vector, _, record = serving_share.weights(CFG, self.frozen(scores), self.members)
        self.assertEqual(vector, {})
        self.assertEqual(record["earners"], 2)


from datetime import UTC, datetime, timedelta  # noqa: E402

from bittensor_wallet import Keypair  # noqa: E402

from zils.runtime import signed  # noqa: E402

KEY = Keypair.create_from_uri("//Evidence")
HK = {n: Keypair.create_from_uri(f"//{n}").ss58_address for n in "ABCDE"}
RCFG = {
    **CFG,
    "volume_share": 0.3,
    "receipts": {"url": "https://g.example/r", "signer": KEY.ss58_address, "window_hours": 24},
}
UNTIL = datetime(2026, 10, 28, 12, tzinfo=UTC)


def iso(t):
    return t.isoformat().replace("+00:00", "Z")


def receipt(hops=(), checks=(), **over):
    key = over.pop("key", KEY)
    payload = {
        "schema": "zils-serving-receipts/v1",
        "signer": KEY.ss58_address,
        "since": iso(UNTIL - timedelta(hours=24)),
        "until": iso(UNTIL),
        "hops": list(hops),
        "checks": list(checks),
        **over,
    }
    return signed(payload, key)


def hop(hk, tokens, model="m"):
    return {
        "hotkey": HK[hk],
        "model": model,
        "weights_sha256": "a" * 64,
        "answered": 1,
        "input_tokens": tokens,
        "failed": {},
    }


def check(hk, **kw):
    return {"hotkey": HK[hk], **{"pass": 0, "marginal": 0, "fail": 0, "mismatch": 0, **kw}}


class ReceiptsConfigTest(unittest.TestCase):
    def test_valid_and_pairing(self):
        self.assertEqual(serving_share.validate(dict(RCFG)), RCFG)
        no_vs = {k: v for k, v in RCFG.items() if k != "volume_share"}
        no_rc = {k: v for k, v in RCFG.items() if k != "receipts"}
        bad_rc = {**RCFG, "receipts": {**RCFG["receipts"], "window_hours": 721}}
        http = {**RCFG, "receipts": {**RCFG["receipts"], "url": "http://x"}}
        for config in (no_vs, no_rc, bad_rc, http, {**RCFG, "volume_share": 1.5}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                serving_share.validate(config)


class VerifyReceiptsTest(unittest.TestCase):
    R = RCFG["receipts"]

    def test_ok(self):
        m = receipt([hop("A", 5)], [check("A", **{"pass": 2})])
        self.assertEqual(
            serving_share.verify_receipts(m, self.R, UNTIL)["hops"][0]["input_tokens"], 5
        )

    def test_refusals(self):
        other = Keypair.create_from_uri("//Other")
        extra = hop("A", 5)
        extra["x"] = 1
        bad = [
            receipt(key=other),
            receipt(schema="nope"),
            receipt(since=iso(UNTIL - timedelta(hours=23))),
            receipt([extra]),
            receipt([hop("A", -1)]),
            receipt([{**hop("A", 1), "weights_sha256": "zz"}]),
            receipt([{**hop("A", 1), "hotkey": "bad"}]),
            receipt([{**hop("A", 1), "failed": {"x": "1"}}]),
            receipt([{**hop("A", 1), "failed": {1: 1}}]),
            receipt(checks=[check("A", fail=-1)]),
            receipt(checks=[{**check("A"), "extra": 1}]),
        ]
        for m in bad:
            with self.subTest(m=m), self.assertRaises(ValueError):
                serving_share.verify_receipts(m, self.R, UNTIL)
        with self.assertRaises(ValueError):
            serving_share.verify_receipts(receipt(), self.R, UNTIL + timedelta(hours=1))


class VolumeWeightsTest(unittest.TestCase):
    members = {str(i + 1): {"hotkey": HK[n]} for i, n in enumerate("ABCD")}

    def run_(self, scores, hops=(), checks=()):
        frozen = {"written_at": 1.0, "frozen_at": 2.0, "scores": scores}
        payload = serving_share.verify_receipts(receipt(hops, checks), RCFG["receipts"], UNTIL)
        return serving_share.weights(RCFG, frozen, self.members, payload)

    def scores(self, names="ABC"):
        return {HK[n]: {"x": entry(1)} for n in names}

    def test_split_by_tokens_across_hops(self):
        _, vol, rec = self.run_(self.scores(), [hop("A", 100), hop("A", 50, "n"), hop("B", 30)])
        self.assertEqual(vol, {1: 150, 2: 30})
        self.assertEqual(rec["volume"], {"1": 150, "2": 30})
        self.assertEqual((rec["volume_share"], rec["struck"]), (0.3, []))

    def test_struck_zero_and_not_counted(self):
        ready, vol, rec = self.run_(
            self.scores(), [hop("A", 9), hop("B", 9)], [check("A", mismatch=1)]
        )
        self.assertEqual((ready, vol), ({}, {}))  # only 2 left, switch-on is 3
        self.assertEqual(rec["struck"], [HK["A"]])
        ready, vol, _ = self.run_(
            self.scores("ABCD"), [hop("A", 9), hop("B", 9)], [check("A", fail=1)]
        )
        self.assertEqual((sorted(ready), vol), ([2, 3, 4], {2: 9}))

    def test_zero_readiness_unregistered_and_no_traffic(self):
        scores = {**self.scores(), HK["D"]: {"x": entry(0)}}
        _, vol, _ = self.run_(scores, [hop("D", 9), hop("E", 9), hop("A", 4)])
        self.assertEqual(vol, {1: 4})
        _, vol, rec = self.run_(self.scores())
        self.assertEqual((vol, rec["volume"]), ({}, {}))

    def test_no_receipts_unchanged(self):
        ready, vol, rec = serving_share.weights(
            CFG, {"written_at": 1.0, "frozen_at": 2.0, "scores": self.scores()}, self.members
        )
        self.assertEqual(vol, {})
        self.assertNotIn("volume", rec)
        self.assertEqual(sorted(ready), [1, 2, 3])


class FreezeReceiptsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = __import__("pathlib").Path(self.tmp.name)
        src = os.path.join(self.tmp.name, "scores.json")
        with open(src, "w") as f:
            json.dump({"hk": {"m": entry(1)}}, f)
        self.cfg = {"serving_share": {**RCFG, "scores": src}}
        self.now = UNTIL.timestamp() + 100
        os.utime(src, (self.now, self.now))

    def attach(self, config=None):
        from zils import validator

        return validator.attach_serving_scores(config or self.cfg, self.work, {}, self.now)

    def test_frozen_once_and_embedded(self):
        message = receipt([hop("A", 5)])
        with mock.patch.object(serving_share, "fetch_receipts", return_value=message) as fetch:
            first = self.attach()
            second = self.attach()
        fetch.assert_called_once()
        self.assertEqual(fetch.call_args.args[1], UNTIL)
        self.assertEqual(first["serving_share"]["receipts"], message)
        self.assertEqual(second, first)
        self.assertTrue((self.work / "serving-receipts.json").is_file())

    def test_fetch_error_writes_nothing(self):
        with mock.patch.object(serving_share, "fetch_receipts", side_effect=OSError("down")):
            with self.assertRaises(OSError):
                self.attach()
        self.assertEqual(os.listdir(self.work), ["scores.json"])

    def test_without_receipts_unchanged(self):
        out = self.attach({"serving_share": {**CFG, "scores": self.cfg["serving_share"]["scores"]}})
        self.assertEqual(set(out["serving_share"]), {"config", "frozen"})
        self.assertFalse((self.work / "serving-receipts.json").exists())

    def test_publication_requires_frozen_receipts(self):
        from zils import validator

        (self.work / "report.json").write_text("{}")
        (self.work / "serving-scores.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "serving-receipts.json"):
            validator.publication_report(self.cfg, self.work, self.work)

    def test_fetch_receipts_verifies(self):
        import io

        def opened(message):
            op = mock.Mock()
            op.open.return_value.__enter__ = lambda s: io.BytesIO(json.dumps(message).encode())
            op.open.return_value.__exit__ = lambda *a: None
            return op

        R = RCFG["receipts"]
        with mock.patch("zils.protocol.opener", return_value=opened(receipt())) as o:
            self.assertEqual(
                serving_share.fetch_receipts(R, UNTIL)["payload"]["signer"], R["signer"]
            )
            url = o.return_value.open.call_args.args[0]
        self.assertIn("until=2026-10-28T12%3A00%3A00Z&hours=24", url)
        with mock.patch("zils.protocol.opener", return_value=opened(receipt(key=KEY, schema="x"))):
            with self.assertRaises(ValueError):
                serving_share.fetch_receipts(R, UNTIL)

    def test_counts_stay_exact_as_floats(self):
        with self.assertRaises(ValueError):
            serving_share.verify_receipts(receipt([hop("A", 2**53)]), RCFG["receipts"], UNTIL)


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
