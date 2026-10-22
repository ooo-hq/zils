"""Feature emissions conserve the budget and cannot bypass frozen round evidence."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.promoted_profile import PROMOTED, promoted_profile
from zils import box, feature_tracks, models, testnet

MEMBERS = {str(i): {"hotkey": f"miner-{i}"} for i in range(1, 7)}
DRAWS = [
    {"manifest_sha256": "a" * 64, "dataset_sha256": "b" * 64},
    {"manifest_sha256": "c" * 64, "dataset_sha256": "d" * 64},
]


def feature_config():
    return {
        "share": 0.2,
        "burn_uid": 0,
        "tracks": {
            f"box/{pool}": {
                "model": models.spec(model),
                "enabled": True,
                "share": share,
                "incumbent_sha256": "e" * 64,
                "kings": [f"miner-{uid}"],
                "draws": copy.deepcopy(DRAWS),
                "policy": {},
            }
            for pool, model, share, uid in (
                ("d3-mini", PROMOTED, 0.8, 3),
                ("d3-lite", "d3-lite-0.8b-v1", 0.2, 4),
            )
        },
    }


def fixture():
    base = {
        "share": 0.5,
        "burn_uid": 0,
        "pools": {
            pool: {
                "model": models.spec(model),
                "enabled": True,
                "share": share,
                "incumbent_sha256": "f" * 64,
                "draws": DRAWS,
                "kings": [f"miner-{uid}"],
            }
            for pool, model, share, uid in (
                ("d3-mini", PROMOTED, 0.8, 1),
                ("d3-lite", "d3-lite-0.8b-v1", 0.2, 2),
            )
        },
    }
    from zils import model_pools

    features = feature_tracks.validate(feature_config())
    decisions = {
        key: {
            "track_id": key,
            "model": entry["model"],
            "incumbent_sha256": entry["incumbent_sha256"],
            "draws": entry["draws"],
            "scope": "synthetic-only",
            "decoder": "box-edges-101-presence/v1",
            "candidates": [],
            "evaluated_at": 100.0,
            "policy": box.validate_policy(),
        }
        for key, entry in features["tracks"].items()
    }
    config = {
        "chain": {"network": "test", "netuid": 579},
        "validator_hotkey": "validator",
        "members": MEMBERS,
        "base_pools": base,
        "feature_tracks": features,
    }
    report = {
        "chain": config["chain"],
        "identities": {k: v["hotkey"] for k, v in MEMBERS.items()},
        "miners": [{"uid": 5, "status": "evaluated", "skill": 1.0}],
        "base_pools": {"config": model_pools.validate(base), "decisions": {}},
        "feature_tracks": {"config": features, "decisions": decisions},
    }
    return config, report


def publish(config, report):
    state = {
        "members": report["identities"],
        "owner_uid": 0,
        "validator_uid": 0,
        "block": 10,
        "required_version": 1,
        "retry_after_block": 10,
    }
    chain = SimpleNamespace(plan=lambda intent, wallet: SimpleNamespace(to_dict=lambda: {}))
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
    with tempfile.TemporaryDirectory() as tmp, patch("zils.testnet.preflight", return_value=state):
        return testnet.publish_round(config, tmp, report, chain, wallet)


class FeatureRewardsTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(promoted_profile())

    def test_manual_publisher_requires_and_reuses_all_frozen_tracks(self):
        from zils.validator import publication_report

        config, report = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            raw = {k: v for k, v in report.items() if k not in ("base_pools", "feature_tracks")}
            (work / "report.json").write_text(json.dumps(raw))
            (work / "base-pools.json").write_text(json.dumps(report["base_pools"]))
            with self.assertRaises(ValueError):
                publication_report(config, work, work)
            (work / "feature-tracks.json").write_text(json.dumps(report["feature_tracks"]))
            self.assertEqual(publication_report(config, work, work), report)

    def test_approved_split_reaches_all_five_competitions(self):
        config, report = fixture()
        result = publish(config, report)
        expected = {"1": 0.40, "2": 0.10, "3": 0.16, "4": 0.04, "5": 0.30}
        self.assertEqual(set(result["requested_weights"]), set(expected))
        for uid, value in expected.items():
            self.assertAlmostEqual(result["requested_weights"][uid], value)

    def test_missing_or_changed_feature_evidence_cannot_publish(self):
        config, report = fixture()
        for mutation in ("missing", "config", "parent", "draw"):
            bad = copy.deepcopy(report)
            if mutation == "missing":
                del bad["feature_tracks"]
            elif mutation == "config":
                bad["feature_tracks"]["config"]["share"] = 0.1
            elif mutation == "parent":
                bad["feature_tracks"]["decisions"]["box/d3-mini"]["model"] = models.spec(models.D3)
            else:
                bad["feature_tracks"]["decisions"]["box/d3-mini"]["draws"] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                publish(config, bad)

    def test_empty_feature_tracks_burn_and_preserve_customer_share(self):
        config, report = fixture()
        for entry in config["feature_tracks"]["tracks"].values():
            entry["kings"] = []
        report["feature_tracks"]["config"] = copy.deepcopy(config["feature_tracks"])
        result = publish(config, report)
        self.assertAlmostEqual(result["requested_weights"]["0"], 0.2)
        self.assertAlmostEqual(result["requested_weights"]["5"], 0.3)

    def test_invalid_outer_split_and_legacy_features_rejected(self):
        config, report = fixture()
        config["feature_tracks"]["share"] = 0.6
        report["feature_tracks"]["config"]["share"] = 0.6
        with self.assertRaises(ValueError):
            publish(config, report)
        config, report = fixture()
        del config["base_pools"]
        with self.assertRaises(ValueError):
            publish(config, report)

    def test_last_five_distinct_registered_feature_kings(self):
        config, report = fixture()
        config["feature_tracks"]["tracks"]["box/d3-mini"]["kings"] = [
            "miner-3",
            "miner-3",
            "gone",
            "miner-4",
            "miner-6",
            "miner-2",
            "miner-1",
        ]
        report["feature_tracks"]["config"] = copy.deepcopy(config["feature_tracks"])
        result = publish(config, report)["requested_weights"]
        self.assertAlmostEqual(result["3"], 0.04)
        self.assertAlmostEqual(result["4"], 0.08)
        self.assertAlmostEqual(result["6"], 0.04)
        self.assertAlmostEqual(result["2"], 0.14)
        self.assertAlmostEqual(result["1"], 0.4)


if __name__ == "__main__":
    unittest.main()
