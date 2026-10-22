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


if __name__ == "__main__":
    unittest.main()
