"""Crown proposals require private, immutable two-draw evidence and current ownership."""

import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts import crown_feature
from tests import test_feature_round
from tests.test_feature_commitments import adapter
from tests.test_feature_rewards import fixture as reward_fixture
from zils import feature_commitments as bundles, models


class FeatureCrownTest(unittest.TestCase):
    def setUp(self):
        self.runner = test_feature_round.FeatureRoundTest()
        self.runner.setUp()
        self.addCleanup(self.runner.doCleanups)
        self.root = self.runner.root
        self.track = "box/d3-mini"
        self.config = test_feature_round.fixture(self.root)
        self.config |= {
            "chain": {"network": "test", "netuid": 579},
            "base_pools": reward_fixture()[0]["base_pools"],
            "adapter_evidence": {"unchanged": True},
        }
        self.config["feature_tracks"]["tracks"][self.track]["kings"] = ["previous-king"]
        self.snapshot, self.candidate, fetch = self.runner.candidate()
        self.snapshot |= {"network": "test", "netuid": 579, "block": 20}
        with (
            patch(
                "zils.feature_commitments.parent",
                return_value=bundles.validate_bundle(self.candidate),
            ),
            patch("zils.feature_commitments.fetch", side_effect=fetch),
        ):
            self.runner.run_epoch(self.config, 1000, self.snapshot)
            result, _, _ = self.runner.run_epoch(self.config, 44200, self.snapshot)
        self.decision = result[self.track]
        self.config_path = self.root / "config.json"
        self.decision_path = self.root / "confirmed.json"
        self.snapshot_path = self.root / "snapshot.json"
        self.out = self.root / "proposed.json"
        self.write_inputs()

    def write_inputs(self):
        for path, value in (
            (self.config_path, self.config),
            (self.decision_path, self.decision),
            (self.snapshot_path, self.snapshot),
        ):
            path.write_text(json.dumps(value))

    def propose(self):
        return crown_feature.propose(
            self.config_path,
            self.track,
            self.decision_path,
            self.candidate,
            self.snapshot_path,
            self.out,
        )

    def cache(self, bundle, draw=1):
        track = self.config["feature_tracks"]["tracks"][self.track]
        binding = track["draws"][draw]
        role = "incumbent-" if bundle == track["incumbent_sha256"] else ""
        return (
            self.root
            / "feature-tracks"
            / self.track
            / "store"
            / bundle
            / (role + binding["manifest_sha256"] + "-scores.json")
        )

    def test_cli_writes_only_the_reviewed_track_changes_and_preserves_source(self):
        original = self.config_path.read_bytes()
        with redirect_stdout(io.StringIO()) as output:
            crown_feature.main(
                [
                    "--config",
                    str(self.config_path),
                    "--track",
                    self.track,
                    "--decision",
                    str(self.decision_path),
                    "--candidate",
                    str(self.candidate),
                    "--snapshot",
                    str(self.snapshot_path),
                    "--out",
                    str(self.out),
                ]
            )
        proposal = json.loads(self.out.read_text())
        expected = copy.deepcopy(self.config)
        expected["feature_tracks"]["tracks"][self.track] |= {
            "incumbent_sha256": bundles.bundle_hash(self.candidate),
            "kings": ["miner-one", "previous-king"],
        }
        expected["feature_track_rounds"][self.track]["reference"] = str(self.candidate.resolve())
        self.assertEqual(proposal, expected)
        self.assertEqual(self.config_path.read_bytes(), original)
        self.assertFalse(json.loads(output.getvalue())["installed"])
        with self.assertRaises(FileExistsError):
            self.propose()
        self.assertEqual(json.loads(self.out.read_text()), expected)

    def test_fabricated_confirmed_winner_and_wrong_parent_fail(self):
        original = copy.deepcopy(self.decision)
        self.decision["confirmed_winner"]["uid"] = 2
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.propose()
        self.assertFalse(self.out.exists())
        self.decision = original
        self.write_inputs()
        source = adapter(self.root / "wrong-parent", model=models.D3)
        self.candidate = self.root / "wrong-bundle"
        bundles.pack(source, self.candidate, self.track)
        with self.assertRaises(ValueError):
            self.propose()
        self.assertFalse(self.out.exists())

    def test_crown_reevaluates_new_baseline_and_preserves_candidate_evidence(self):
        bundle = bundles.bundle_hash(self.candidate)
        original = {
            self.cache(bundle, draw): self.cache(bundle, draw).read_bytes() for draw in (0, 1)
        }
        proposal = self.propose()
        result, calls, _ = self.runner.run_epoch(proposal, 44201)
        self.assertEqual(calls, [(65535, "first"), (65535, "second")])
        self.assertEqual(result[self.track]["incumbent_sha256"], bundle)
        self.assertIsNone(result[self.track]["confirmed_winner"])
        self.assertEqual(len(result[self.track]["qualification"]), 2)
        _, calls, _ = self.runner.run_epoch(proposal, 44202)
        self.assertEqual(calls, [])
        self.assertEqual({path: path.read_bytes() for path in original}, original)

    def test_changed_config_and_cross_network_snapshot_fail(self):
        self.config["feature_tracks"]["tracks"][self.track]["policy"]["min_iou_gain"] = 0.9
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.propose()
        self.config["feature_tracks"]["tracks"][self.track]["policy"].pop("min_iou_gain")
        self.snapshot["netuid"] = 580
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.propose()
        self.assertFalse(self.out.exists())

    def test_missing_or_rebound_cache_cannot_crown(self):
        for bundle in (
            bundles.bundle_hash(self.candidate),
            self.config["feature_tracks"]["tracks"][self.track]["incumbent_sha256"],
        ):
            for draw in (0, 1):
                path = self.cache(bundle, draw)
                original = path.read_text()
                path.unlink()
                with self.subTest(bundle=bundle, draw=draw), self.assertRaises(OSError):
                    self.propose()
                path.write_text(original)
        path = self.cache(bundles.bundle_hash(self.candidate))
        original = path.read_text()
        for field, value in (
            ("bundle_sha256", "f" * 64),
            ("checkpoint_sha256", "f" * 64),
            ("decoder", "wrong"),
            ("hotkey", "replacement"),
            ("model", models.spec(models.D3)),
            ("dataset_sha256", "f" * 64),
            ("draw", self.config["feature_tracks"]["tracks"][self.track]["draws"][0]),
            ("evaluated_at", None),
        ):
            with self.subTest(field=field):
                cached = json.loads(original)
                cached[field] = value
                path.write_text(json.dumps(cached))
                with self.assertRaises(ValueError):
                    self.propose()
        cached = json.loads(original)
        cached["miners"][0]["runtime"]["model"] = models.spec(models.D3)
        path.write_text(json.dumps(cached))
        with self.assertRaises(ValueError):
            self.propose()
        self.assertFalse(self.out.exists())

    def test_recomputed_ineligible_predictions_override_claimed_confirmation(self):
        path = self.cache(bundles.bundle_hash(self.candidate))
        cached = json.loads(path.read_text())
        for row in cached["miners"][0]["predictions"]:
            if not row["id"].endswith(":presence"):
                row["probabilities"] = {key: float(key == "0") for key in row["probabilities"]}
        path.write_text(json.dumps(cached))
        with self.assertRaisesRegex(ValueError, "qualif|confirmed"):
            self.propose()
        self.assertFalse(self.out.exists())

    def test_actual_cache_times_enforce_confirmation_delay(self):
        path = self.cache(bundles.bundle_hash(self.candidate))
        cached = json.loads(path.read_text())
        cached["evaluated_at"] = 44199
        path.write_text(json.dumps(cached))
        with self.assertRaisesRegex(ValueError, "12 hours"):
            self.propose()
        self.assertFalse(self.out.exists())

    def test_uid_reassignment_and_changed_claim_cannot_inherit_the_crown(self):
        self.snapshot["members"]["1"] = "replacement"
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.propose()
        # Changing public receipts too still cannot replace the private cache owner.
        self.decision = json.loads(json.dumps(self.decision).replace("miner-one", "replacement"))
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.propose()
        self.snapshot["members"]["1"] = "miner-one"
        self.decision = json.loads(json.dumps(self.decision).replace("replacement", "miner-one"))
        self.snapshot["commitments"]["1"]["data"] = bundles.format_commitment(
            "f" * 64, "a" * 40, "owner/model"
        )
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.propose()
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
