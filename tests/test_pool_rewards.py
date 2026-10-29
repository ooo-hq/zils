"""Independent pool allocations compose into one conserved incentive vector."""

import copy
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests.promoted_profile import PROMOTED, promoted_profile
from zils import base_round, models, testnet

MEMBERS = {"1": {"hotkey": "a"}, "2": {"hotkey": "b"}}
DRAW = {"manifest_sha256": "a" * 64, "dataset_sha256": "b" * 64}


def pools():
    track = {"incumbent_sha256": "c" * 64, "draws": [DRAW], "enabled": True}
    return {
        "share": 0.7,
        "burn_uid": 0,
        "pools": {
            "d3-mini": {**track, "model": PROMOTED, "share": 0.75, "kings": ["b"]},
            "d3-lite": {**track, "model": "d3-lite-0.8b-v1", "share": 0.25, "kings": ["a"]},
        },
    }


class PoolRewardsTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(promoted_profile())

    def test_independent_allocations_and_shared_king(self):
        from zils import model_pools

        cfg = pools()
        weights, receipt = model_pools.weights(cfg, {}, MEMBERS)
        self.assertEqual(weights, {1: 0.25, 2: 0.75})
        self.assertEqual(receipt["burned"], 0)
        cfg["pools"]["d3-mini"]["kings"] = ["a", "b", "a"]
        weights, _ = model_pools.weights(cfg, {}, MEMBERS)
        self.assertEqual(weights, {1: 0.625, 2: 0.375})

    def test_disabled_or_unowned_pool_burns_without_redistribution(self):
        from zils import model_pools

        for changes in ({"enabled": False}, {"kings": []}, {"kings": ["deregistered"]}):
            with self.subTest(changes=changes):
                cfg = pools()
                cfg["pools"]["d3-lite"].update(changes)
                weights, receipt = model_pools.weights(cfg, {}, MEMBERS)
                self.assertEqual(weights, {0: 0.25, 2: 0.75})
                self.assertEqual(receipt["burned"], 0.25)

    def test_allocation_and_lineage_fail_closed(self):
        from zils import model_pools

        for bad in (-0.1, 1.1, float("nan"), float("inf"), True):
            cfg = pools()
            cfg["pools"]["d3-lite"]["share"] = bad
            with self.subTest(share=bad), self.assertRaises(ValueError):
                model_pools.validate(cfg)
        cfg = pools()
        cfg["pools"]["d3-lite"]["model"] = models.D3
        with self.assertRaisesRegex(ValueError, "pool"):
            model_pools.validate(cfg)
        with self.assertRaises(ValueError):
            model_pools.validate({**pools(), "unknown": True})

    def test_decisions_are_bound_to_pool_profile_and_draw(self):
        from zils import model_pools

        decision = {
            **DRAW,
            "pool_id": "d3-mini",
            "model": models.spec(PROMOTED),
            "policy": dict(base_round.DEFAULT_POLICY),
            "incumbent": {"uid": 65535, "sha256": "c" * 64},
            "candidates": [],
            "pending_winner": None,
        }
        self.assertEqual(
            model_pools.weights(pools(), {"d3-mini": decision}, MEMBERS)[0], {1: 0.25, 2: 0.75}
        )
        for field, value in (
            ("pool_id", "d3-lite"),
            ("model", models.spec(models.D3)),
            ("dataset_sha256", "f" * 64),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                model_pools.weights(pools(), {"d3-mini": {**decision, field: value}}, MEMBERS)
        with self.assertRaises(ValueError):
            model_pools.weights(pools(), {"unknown": decision}, MEMBERS)

    def test_publisher_preserves_mini_vector_and_refuses_changed_config(self):
        from zils import model_pools

        cfg = pools()
        cfg["pools"] = {"d3-mini": {**cfg["pools"]["d3-mini"], "share": 1.0, "kings": ["a"]}}
        config = {
            "chain": {"network": "test", "netuid": 579},
            "validator_hotkey": "validator",
            "members": MEMBERS,
            "base_pools": cfg,
        }
        report = {
            "chain": config["chain"],
            "identities": {"1": "a", "2": "b"},
            "miners": [
                {"uid": 1, "status": "evaluated", "skill": 0.75},
                {"uid": 2, "status": "evaluated", "skill": 0.25},
            ],
            "base_pools": {"config": model_pools.validate(cfg), "decisions": {}},
        }
        state = {
            "members": report["identities"],
            "owner_uid": 0,
            "validator_uid": 0,
            "block": 10,
            "required_version": 1,
            "retry_after_block": 10,
        }
        intents = []
        chain = SimpleNamespace(
            plan=lambda intent, wallet: intents.append(intent) or SimpleNamespace(to_dict=dict)
        )
        wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("zils.testnet.preflight", return_value=state),
        ):
            result = testnet.publish_round(config, tmp, report, chain, wallet)
            self.assertEqual(intents[-1].version_key, testnet.WEIGHTS_VERSION)
            self.assertAlmostEqual(result["requested_weights"]["1"], 0.925)
            self.assertAlmostEqual(result["requested_weights"]["2"], 0.075)
            changed = copy.deepcopy(config)
            changed["base_pools"]["pools"]["d3-mini"]["kings"] = ["b"]
            with self.assertRaisesRegex(ValueError, "config"):
                testnet.publish_round(changed, tmp, report, chain, wallet)
            with self.assertRaisesRegex(ValueError, "base_tier|both"):
                testnet.publish_round({**config, "base_tier": {}}, tmp, report, chain, wallet)
            # Raised on chain after the grace window: this release can no longer write weights.
            state["required_version"] = testnet.WEIGHTS_VERSION + 1
            with self.assertRaisesRegex(ValueError, "requires weights version 2.*zils update"):
                testnet.publish_round(config, tmp, report, chain, wallet)


SERVING_MEMBERS = {"1": {"hotkey": "a"}, "2": {"hotkey": "b"}, "3": {"hotkey": "c"}}


def serving_fixture(scores, switch_on=3, base_share=0.63):
    from zils import model_pools, serving_share

    cfg = {**pools(), "share": base_share}
    serving = {
        "share": 0.10,
        "scores": "/nonexistent/serving-scores.json",
        "switch_on_miners": switch_on,
        "max_age_seconds": 3600,
    }
    config = {
        "chain": {"network": "test", "netuid": 579},
        "validator_hotkey": "validator",
        "members": SERVING_MEMBERS,
        "base_pools": cfg,
        "serving_share": serving,
    }
    report = {
        "chain": config["chain"],
        "identities": {"1": "a", "2": "b", "3": "c"},
        "miners": [
            {"uid": 1, "status": "evaluated", "skill": 0.75},
            {"uid": 2, "status": "evaluated", "skill": 0.25},
        ],
        "base_pools": {"config": model_pools.validate(cfg), "decisions": {}},
        "serving_share": {
            "config": serving_share.validate(serving),
            "frozen": {"written_at": 90.0, "frozen_at": 100.0, "scores": scores},
        },
    }
    return config, report


def publish(config, report, owner_uid=0):
    state = {
        "members": report["identities"],
        "owner_uid": owner_uid,
        "validator_uid": 0,
        "block": 10,
        "required_version": 1,
        "retry_after_block": 10,
    }
    chain = SimpleNamespace(plan=lambda intent, wallet: SimpleNamespace(to_dict=dict))
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
    with tempfile.TemporaryDirectory() as tmp, patch("zils.testnet.preflight", return_value=state):
        return testnet.publish_round(config, tmp, report, chain, wallet)


def digest(value):
    import hashlib
    import json

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ServingShareTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(promoted_profile())

    def test_absent_config_vector_and_record_are_unchanged(self):
        from tests import test_feature_rewards
        from zils import model_pools

        cfg = pools()
        config = {
            "chain": {"network": "test", "netuid": 579},
            "validator_hotkey": "validator",
            "members": MEMBERS,
            "base_pools": cfg,
        }
        report = {
            "chain": config["chain"],
            "identities": {"1": "a", "2": "b"},
            "miners": [
                {"uid": 1, "status": "evaluated", "skill": 0.75},
                {"uid": 2, "status": "evaluated", "skill": 0.25},
            ],
            "base_pools": {"config": model_pools.validate(cfg), "decisions": {}},
        }
        # Snapshots taken from the unmodified publish_round before the serving track existed.
        result = publish(config, report)
        self.assertEqual(result["requested_weights"], {"1": 0.4000000000000001, "2": 0.6})
        self.assertEqual(
            digest(result["tracks"]),
            "40d0a761782310735161e4731aeb87857493bb95b35761f63bcd16ccac87f2e2",
        )
        result = test_feature_rewards.publish(*test_feature_rewards.fixture())
        self.assertEqual(
            result["requested_weights"],
            {"1": 0.4, "2": 0.1, "3": 0.16000000000000003, "4": 0.04000000000000001, "5": 0.3},
        )
        self.assertEqual(
            digest(result["tracks"]),
            "bd626303ee4269cb79e055b5e378f99909b494554ac7ac81eeeba0529f430a5f",
        )

    def test_serving_share_burns_until_switched_on(self):
        scores = {"a": {"m": {"score": 1.0}}, "b": {"m": {"score": 2.0}}}
        result = publish(*serving_fixture(scores))
        weights, tracks = result["requested_weights"], result["tracks"]
        self.assertEqual(set(weights), {"0", "1", "2"})
        self.assertAlmostEqual(weights["0"], 0.10)
        self.assertAlmostEqual(weights["1"], 0.63 * 0.25 + 0.27 * 0.75)
        self.assertAlmostEqual(weights["2"], 0.63 * 0.75 + 0.27 * 0.25)
        self.assertAlmostEqual(tracks["burned"], 0.10)
        self.assertEqual(tracks["serving_share"], 0.10)
        self.assertAlmostEqual(tracks["adapter_share"], 0.27)
        self.assertFalse(tracks["serving"]["switched_on"])
        self.assertEqual(tracks["serving"]["earners"], 2)
        self.assertEqual(tracks["serving"]["weights"], {})
        self.assertNotIn("feature_share", tracks)

    def test_budget_filled_exactly_to_one_still_publishes(self):
        # 1 - 0.9 - 0.1 is -2.8e-17 in floats; the adapter share must clamp to 0, not refuse the round.
        result = publish(*serving_fixture({}, base_share=0.9))
        self.assertEqual(result["tracks"]["adapter_share"], 0.0)
        self.assertAlmostEqual(result["requested_weights"]["0"], 0.10)

    def test_stale_config_over_budget_is_refused_clearly(self):
        with self.assertRaisesRegex(ValueError, "exceed the reward budget"):
            publish(*serving_fixture({}, base_share=0.95))

    def test_switched_on_serving_earners_split_the_share_by_summed_score(self):
        scores = {
            "a": {"m": {"score": 1.0}},
            "b": {"m": {"score": 1.0}, "n": {"score": 1.0}},
            "c": {"m": {"score": 2.0}},
        }
        # Nothing burns, so a non-owner burn uid is fine.
        result = publish(*serving_fixture(scores), owner_uid=7)
        weights, tracks = result["requested_weights"], result["tracks"]
        self.assertEqual(set(weights), {"1", "2", "3"})
        self.assertAlmostEqual(weights["1"], 0.63 * 0.25 + 0.27 * 0.75 + 0.02)
        self.assertAlmostEqual(weights["2"], 0.63 * 0.75 + 0.27 * 0.25 + 0.04)
        self.assertAlmostEqual(weights["3"], 0.04)
        self.assertAlmostEqual(tracks["burned"], 0.0)
        self.assertAlmostEqual(tracks["adapter_share"], 0.27)
        self.assertTrue(tracks["serving"]["switched_on"])
        self.assertEqual(tracks["serving"]["weights"], {"1": 1.0, "2": 2.0, "3": 2.0})

    def test_serving_burn_requires_the_owner_burn_uid(self):
        with self.assertRaisesRegex(ValueError, "burn_uid"):
            publish(*serving_fixture({}), owner_uid=7)

    def test_report_and_config_must_agree_on_serving_share(self):
        config, report = serving_fixture({})
        missing = {k: v for k, v in report.items() if k != "serving_share"}
        with self.assertRaisesRegex(ValueError, "serving"):
            publish(config, missing)
        unconfigured = {k: v for k, v in config.items() if k != "serving_share"}
        with self.assertRaisesRegex(ValueError, "serving"):
            publish(unconfigured, report)
        changed = copy.deepcopy(config)
        changed["serving_share"]["share"] = 0.2
        with self.assertRaisesRegex(ValueError, "serving"):
            publish(changed, report)
        legacy = {k: v for k, v in config.items() if k != "base_pools"}
        legacy_report = {k: v for k, v in report.items() if k != "base_pools"}
        with self.assertRaisesRegex(ValueError, "serving_share requires base_pools"):
            publish(legacy, legacy_report)


def receipts_fixture(hops=(), checks=(), switch_on=3):
    from tests.test_serving_share import HK, KEY, UNTIL, receipt
    from zils import model_pools, serving_share

    scores = {HK[n]: {"m": {"score": 1.0}} for n in "ABC"}
    config, report = serving_fixture(scores, switch_on=switch_on)
    members = {str(i + 1): {"hotkey": HK[n]} for i, n in enumerate("ABC")}
    serving = {
        **config["serving_share"],
        "volume_share": 0.3,
        "receipts": {"url": "https://gateway/r", "signer": KEY.ss58_address, "window_hours": 24},
    }
    config = {**config, "members": members, "serving_share": serving}
    for pool, king in (("d3-mini", "B"), ("d3-lite", "A")):
        config["base_pools"]["pools"][pool]["kings"] = [HK[king]]
    report["base_pools"]["config"] = model_pools.validate(config["base_pools"])
    report = {
        **report,
        "identities": {uid: m["hotkey"] for uid, m in members.items()},
        "serving_share": {
            "config": serving_share.validate(serving),
            # Frozen after the receipts window closed, as attach_serving_scores does.
            "frozen": {**report["serving_share"]["frozen"], "frozen_at": UNTIL.timestamp() + 60},
            "receipts": receipt(hops, checks),
        },
    }
    return config, report


class ServingReceiptsTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(promoted_profile())

    def test_plan_3b_config_without_receipts_is_unchanged(self):
        s1 = {"a": {"m": {"score": 1.0}}, "b": {"m": {"score": 2.0}}}
        s2 = {
            "a": {"m": {"score": 1.0}},
            "b": {"m": {"score": 1.0}, "n": {"score": 1.0}},
            "c": {"m": {"score": 2.0}},
        }
        for scores, owner, want, sha in (
            (
                s1,
                0,
                {"0": 0.1, "1": 0.36, "2": 0.54},
                "b282284f1edd5cab2631a27909b494d5acc986ed29d83d7818e622afb471df1e",
            ),
            (
                s2,
                7,
                {"1": 0.38, "2": 0.5800000000000001, "3": 0.04000000000000001},
                "6d64fba3e72d4375e12192105f55b6d1a8e5be0c47e0b001b061039f8606d277",
            ),
        ):
            result = publish(*serving_fixture(scores), owner_uid=owner)
            self.assertEqual(result["requested_weights"], want)
            self.assertEqual(digest(result["tracks"]), sha)

    def test_no_traffic_burns_the_volume_share(self):
        result = publish(*receipts_fixture())
        w, t = result["requested_weights"], result["tracks"]
        self.assertAlmostEqual(w["0"], 0.03)
        self.assertAlmostEqual(w["3"], 0.07 / 3)
        self.assertAlmostEqual(w["1"], 0.36 + 0.07 / 3)
        self.assertAlmostEqual(t["burned"], 0.03)
        self.assertAlmostEqual(t["serving_volume_share"], 0.03)
        self.assertEqual(t["serving"]["volume"], {})

    def test_volume_splits_by_tokens(self):
        from tests.test_serving_share import hop

        result = publish(*receipts_fixture([hop("A", 100), hop("A", 100, "n"), hop("B", 600)]))
        w, t = result["requested_weights"], result["tracks"]
        self.assertAlmostEqual(w["1"], 0.36 + 0.07 / 3 + 0.0075)
        self.assertAlmostEqual(w["2"], 0.54 + 0.07 / 3 + 0.0225)
        self.assertAlmostEqual(w["3"], 0.07 / 3)
        self.assertNotIn("0", w)
        self.assertAlmostEqual(t["burned"], 0.0)
        self.assertEqual(t["serving"]["volume"], {"1": 200, "2": 600})

    def test_struck_miner_gets_nothing_from_either_track(self):
        from tests.test_serving_share import check, hop

        config, report = receipts_fixture(
            [hop("A", 100), hop("B", 100)], [check("A", mismatch=1)], switch_on=2
        )
        w = publish(config, report)["requested_weights"]
        self.assertAlmostEqual(w["1"], 0.36)  # base and adapter only
        self.assertAlmostEqual(w["2"], 0.54 + 0.035 + 0.03)
        self.assertAlmostEqual(w["3"], 0.035)

    def test_burn_requires_owner_uid(self):
        with self.assertRaisesRegex(ValueError, "burn_uid"):
            publish(*receipts_fixture(), owner_uid=7)

    def test_receipts_window_must_have_closed_before_the_scores_froze(self):
        # A correctly signed window from the future (or one not on a whole hour) is not this round's.
        from datetime import timedelta

        from tests.test_serving_share import UNTIL, iso, receipt

        config, report = receipts_fixture()
        later = UNTIL + timedelta(hours=2)
        report["serving_share"]["receipts"] = receipt(
            since=iso(later - timedelta(hours=24)), until=iso(later)
        )
        with self.assertRaisesRegex(ValueError, "window"):
            publish(config, report)
        config, report = receipts_fixture()
        odd = UNTIL - timedelta(minutes=30)
        report["serving_share"]["receipts"] = receipt(
            since=iso(odd - timedelta(hours=24)), until=iso(odd)
        )
        with self.assertRaisesRegex(ValueError, "window"):
            publish(config, report)

    def test_receipts_must_verify_and_match_config(self):
        from bittensor_wallet import Keypair

        from tests.test_serving_share import receipt

        config, report = receipts_fixture()
        report["serving_share"]["receipts"] = receipt(key=Keypair.create_from_uri("//Other"))
        with self.assertRaises(ValueError):
            publish(config, report)
        config, report = receipts_fixture()
        del report["serving_share"]["receipts"]
        with self.assertRaisesRegex(ValueError, "serving"):
            publish(config, report)


if __name__ == "__main__":
    unittest.main()
