"""Sealed feature exams, private cache binding and delayed promotion evidence."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

import zils
from scripts.box_dataset import generate
from tests.test_feature_commitments import adapter
from zils import base_round, feature_commitments, feature_round, models


def fixture(root):
    source = adapter(root / "base", kind="base")
    reference = root / "reference"
    feature_commitments.pack(source, reference, "box/d3-mini", allow_base=True)
    for name, seed in (("first", 311), ("second", 312)):
        generate(root / name, 6, seed)
    return {
        "feature_tracks": {
            "share": 0.2,
            "burn_uid": 0,
            "tracks": {
                "box/d3-mini": {
                    "model": models.spec(models.D3_ZILS_R5),
                    "enabled": True,
                    "share": 1,
                    "incumbent_sha256": feature_commitments.bundle_hash(reference),
                    "kings": [],
                    "draws": [feature_round.draw(root / name) for name in ("first", "second")],
                    "policy": {"bootstrap_resamples": 100},
                }
            },
        },
        "feature_track_rounds": {
            "box/d3-mini": {"reference": "reference", "draw_dirs": ["first", "second"]}
        },
    }


class FeatureRoundTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_audit_rejects_changed_expansion_and_assets_even_with_renamed_labels(self):
        config = fixture(self.root)
        path = self.root / "first"
        manifest, cases = feature_round.audit(path)
        self.assertEqual(
            feature_round.draw(path), config["feature_tracks"]["tracks"]["box/d3-mini"]["draws"][0]
        )
        image = path / "images" / cases[0]["image"]
        original = image.read_bytes()
        image.write_bytes(original + b"changed")
        with self.assertRaises(ValueError):
            feature_round.audit(path)
        image.write_bytes(original)
        expanded = path / "expanded.jsonl"
        rows = [json.loads(line) for line in expanded.read_text().splitlines()]
        rows[0]["state"]["truth"] = cases[0]["box"]
        expanded.write_text("".join(json.dumps(row) + "\n" for row in rows))
        from zils.runtime import digest

        manifest["files"]["expanded.jsonl"] = digest(expanded)
        (path / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "expansion"):
            feature_round.audit(path)

    def test_audit_rejects_symlinked_images_and_unsupported_manifest(self):
        fixture(self.root)
        path = self.root / "first"
        manifest, cases = feature_round.audit(path)
        image = path / "images" / cases[0]["image"]
        target = self.root / "photo.png"
        image.rename(target)
        image.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "regular|symlink"):
            feature_round.audit(path)
        image.unlink()
        target.rename(image)
        manifest["scope"] = "real-images"
        (path / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            feature_round.audit(path)

    def test_rounds_bind_parent_reference_and_disjoint_images(self):
        config = fixture(self.root)
        self.assertEqual(set(feature_round.rounds(config, self.root)), {"box/d3-mini"})
        bad = copy.deepcopy(config)
        bad["feature_tracks"]["tracks"]["box/d3-mini"]["incumbent_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "incumbent"):
            feature_round.rounds(bad, self.root)
        shutil.rmtree(self.root / "second")
        shutil.copytree(self.root / "first", self.root / "second")
        with self.assertRaises(ValueError):
            feature_round.rounds(config, self.root)

    def run_epoch(self, config, now, snapshot=None, *, fail=None, reject_uid=None):
        work = self.root / f"work-{now}"
        work.mkdir(exist_ok=True)
        calls = []

        def evaluate(args, attempt, name, submissions, cases, **kwargs):
            if fail:
                raise RuntimeError(fail)
            rows = [json.loads(line) for line in Path(cases).read_text().splitlines()]
            self.assertEqual(kwargs["model"], models.D3_ZILS_R5)
            self.assertEqual(kwargs["images"], Path(cases).parent / "images")
            result = []
            for entry in submissions:
                if entry["uid"] == reject_uid:
                    result.append(
                        {
                            "uid": entry["uid"],
                            "sha256": entry["sha256"],
                            "status": "rejected",
                            "error": "private runner error",
                        }
                    )
                    calls.append((entry["uid"], Path(cases).parent.name))
                    continue
                good = entry["uid"] != 65535
                predictions = []
                for row in rows:
                    keys = zils.options(row["question"])
                    selected = row["label"] if good or len(keys) == 2 else "0"
                    predictions.append(
                        {
                            "id": row["id"],
                            "elapsed_ms": 1,
                            "probabilities": {k: float(k == selected) for k in keys},
                        }
                    )
                result.append(
                    {
                        "uid": entry["uid"],
                        "sha256": entry["sha256"],
                        "status": "evaluated",
                        "runtime": {"model": models.spec(models.D3_ZILS_R5)},
                        "predictions": predictions,
                    }
                )
                calls.append((entry["uid"], Path(cases).parent.name))
            return {"dataset_sha256": base_round.dataset_hash(rows), "miners": result}

        with (
            patch("zils.validator.run_evaluate", side_effect=evaluate),
            patch("zils.feature_round.time.time", return_value=now),
        ):
            result = feature_round.epoch(
                config, self.root, work, snapshot or {}, SimpleNamespace(device="cpu")
            )
        return result, calls, work

    def candidate(self):
        source = adapter(self.root / "adapter")
        bundle = self.root / "candidate"
        feature_commitments.pack(source, bundle, "box/d3-mini")
        sha = feature_commitments.bundle_hash(bundle)
        snapshot = {
            "members": {"1": "miner-one"},
            "commitments": {
                "1": {
                    "data": feature_commitments.format_commitment(sha, "a" * 40, "owner/model"),
                    "block": 2,
                }
            },
        }

        def fetch(claim, destination, **kwargs):
            shutil.copytree(bundle, destination)
            return {"manifest": feature_commitments.validate_bundle(destination)}

        return snapshot, bundle, fetch

    def test_empty_intake_qualifies_both_draws_and_resumes_without_gpu(self):
        config = fixture(self.root)
        first, calls, work = self.run_epoch(config, 1000)
        self.assertEqual(calls, [(65535, "first"), (65535, "second")])
        receipt = first["box/d3-mini"]
        self.assertEqual(len(receipt["qualification"]), 2)
        self.assertIsNone(receipt["confirmed_winner"])
        self.assertEqual(receipt["candidates"], [])
        second, calls, _ = self.run_epoch(config, 1001)
        self.assertEqual(calls, [])
        self.assertEqual(second["box/d3-mini"]["evaluated_at"], receipt["evaluated_at"])
        public = feature_round.attach(config, self.root, work, {})
        encoded = json.dumps(public)
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn("predictions", encoded)
        self.assertNotIn("group_id", encoded)
        self.assertNotIn("target", encoded)

    def test_candidate_confirmation_waits_twelve_hours_and_reuses_original_times(self):
        config = fixture(self.root)
        snapshot, bundle, fetch = self.candidate()
        with (
            patch(
                "zils.feature_commitments.parent",
                return_value=feature_commitments.validate_bundle(bundle),
            ),
            patch("zils.feature_commitments.fetch", side_effect=fetch),
        ):
            first, calls, _ = self.run_epoch(config, 1000, snapshot)
            self.assertEqual(set(calls), {(65535, "first"), (65535, "second"), (1, "first")})
            self.assertIsNotNone(first["box/d3-mini"]["pending_winner"])
            self.assertIsNone(first["box/d3-mini"]["confirmed_winner"])
            second, calls, _ = self.run_epoch(config, 1001, snapshot)
            self.assertEqual(calls, [])
            self.assertIsNone(second["box/d3-mini"]["confirmed_winner"])
            third, calls, work = self.run_epoch(config, 44200, snapshot)
            self.assertEqual(calls, [(1, "second")])
            receipt = third["box/d3-mini"]
            self.assertEqual(receipt["confirmed_winner"]["uid"], 1)
            self.assertEqual(receipt["confirmed_winner"]["hotkey"], "miner-one")
            self.assertEqual(receipt["confirmation"]["evaluated_at"], 44200)
            self.assertEqual(receipt["candidates"][0]["evaluated_at"], 1000)
            self.assertEqual(config["feature_tracks"]["tracks"]["box/d3-mini"]["kings"], [])
            frozen = feature_round.attach(config, self.root, work, {})
            changed = copy.deepcopy(config)
            changed["feature_tracks"]["tracks"]["box/d3-mini"]["kings"] = ["new-king"]
            with self.assertRaisesRegex(ValueError, "configuration"):
                feature_round.attach(changed, self.root, work, {})
            self.assertEqual(feature_round.attach(config, self.root, work, {}), frozen)

    def test_runtime_failure_cannot_reuse_a_stale_decision(self):
        config = fixture(self.root)
        with self.assertRaises(RuntimeError):
            self.run_epoch(config, 1000, fail="GPU unavailable")
        with self.assertRaises(ValueError):
            feature_round.attach(config, self.root, self.root / "work-1000", {})

    def test_changed_cache_binding_is_a_hard_failure(self):
        config = fixture(self.root)
        self.run_epoch(config, 1000)
        cached = next((self.root / "feature-tracks").rglob("*-scores.json"))
        saved = json.loads(cached.read_text())
        saved["decoder"] = "wrong"
        cached.write_text(json.dumps(saved))
        with self.assertRaises(RuntimeError):
            self.run_epoch(config, 1001)

    def test_audit_failure_removes_previous_epoch_decision(self):
        config = fixture(self.root)
        self.run_epoch(config, 1000)
        next((self.root / "first" / "images").iterdir()).unlink()
        with self.assertRaises(ValueError):
            self.run_epoch(config, 1001)
        self.assertFalse((self.root / "feature-tracks/box/d3-mini/decision.json").exists())

    def test_confirmation_rechecks_bindings_times_and_metric_gates(self):
        config = fixture(self.root)
        snapshot, bundle, fetch = self.candidate()
        with (
            patch(
                "zils.feature_commitments.parent",
                return_value=feature_commitments.validate_bundle(bundle),
            ),
            patch("zils.feature_commitments.fetch", side_effect=fetch),
        ):
            first, _, _ = self.run_epoch(config, 1000, snapshot)
            second, _, _ = self.run_epoch(config, 44200, snapshot)
        first = first["box/d3-mini"]
        second = second["box/d3-mini"]["confirmation"]
        for mutate in (
            lambda row: row.update(decoder="wrong"),
            lambda row: row.update(incumbent_sha256="0" * 64),
            lambda row: row.update(evaluated_draw=first["evaluated_draw"]),
            lambda row: row["candidates"][0].update(evaluated_at=44199),
            lambda row: row["candidates"][0].update(evaluated_at=float("nan")),
        ):
            changed = copy.deepcopy(second)
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                feature_round.confirm(first, changed)
        changed = copy.deepcopy(second)
        changed["candidates"][0]["candidate"]["false_positive_rate"] = 1
        self.assertIsNone(feature_round.confirm(first, changed))

    def test_invalid_decision_is_not_frozen_before_it_can_be_repaired(self):
        config = fixture(self.root)
        _, _, work = self.run_epoch(config, 1000)
        path = self.root / "feature-tracks/box/d3-mini/decision.json"
        value = json.loads(path.read_text())
        value["decoder"] = "wrong"
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            feature_round.attach(config, self.root, work, {})
        self.assertFalse((work / "feature-tracks.json").exists())

    def test_uid_reassignment_cannot_inherit_first_claim_qualification(self):
        config = fixture(self.root)
        snapshot, bundle, fetch = self.candidate()
        with (
            patch(
                "zils.feature_commitments.parent",
                return_value=feature_commitments.validate_bundle(bundle),
            ),
            patch("zils.feature_commitments.fetch", side_effect=fetch),
        ):
            self.run_epoch(config, 1000, snapshot)
            snapshot["members"]["1"] = "new-hotkey"
            result, calls, _ = self.run_epoch(config, 44200, snapshot)
        receipt = result["box/d3-mini"]
        self.assertEqual(calls, [])
        self.assertEqual(receipt["candidates"], [])
        self.assertIsNone(receipt["confirmed_winner"])
        self.assertIn("1", receipt["rejected"])

    def test_failed_incumbent_result_is_retried_and_never_cached(self):
        config = fixture(self.root)
        with self.assertRaises(RuntimeError):
            self.run_epoch(config, 1000, reject_uid=65535)
        self.assertEqual(list((self.root / "feature-tracks").rglob("*-scores.json")), [])
        _, calls, _ = self.run_epoch(config, 1001)
        self.assertEqual(calls, [(65535, "first"), (65535, "second")])

    def test_parent_transport_failure_is_hard_and_recovers_on_retry(self):
        config = fixture(self.root)
        snapshot, bundle, fetch = self.candidate()
        with patch(
            "zils.feature_commitments.parent",
            side_effect=ConnectionError("private upstream outage"),
        ):
            with self.assertRaises(ConnectionError):
                self.run_epoch(config, 1000, snapshot)
        with (
            patch(
                "zils.feature_commitments.parent",
                return_value=feature_commitments.validate_bundle(bundle),
            ),
            patch("zils.feature_commitments.fetch", side_effect=fetch),
        ):
            result, calls, _ = self.run_epoch(config, 1001, snapshot)
        self.assertEqual(result["box/d3-mini"]["pending_winner"]["uid"], 1)
        self.assertIn((1, "first"), calls)

    def test_fetch_transport_failure_is_hard_but_missing_repo_is_candidate_rejection(self):
        config = fixture(self.root)
        snapshot, bundle, _ = self.candidate()
        with (
            patch(
                "zils.feature_commitments.parent",
                return_value=feature_commitments.validate_bundle(bundle),
            ),
            patch(
                "zils.feature_commitments.fetch",
                side_effect=ConnectionError("private download outage"),
            ),
        ):
            with self.assertRaises(RuntimeError):
                self.run_epoch(config, 1000, snapshot)
        with patch(
            "zils.feature_commitments.parent",
            side_effect=HTTPError("https://private/repo", 404, "missing", {}, None),
        ):
            result, _, work = self.run_epoch(config, 1001, snapshot)
        self.assertIsNone(result["box/d3-mini"]["pending_winner"])
        self.assertEqual(
            json.loads((work / "feature-intake.json").read_text())["rejected"],
            {"1": "invalid_feature_commitment"},
        )

    def test_permanent_http_client_errors_reject_only_the_candidate(self):
        config = fixture(self.root)
        snapshot, bundle, _ = self.candidate()
        manifest = feature_commitments.validate_bundle(bundle)
        for index, code in enumerate((400, 401, 403, 404, 410, 422)):
            for phase in ("parent", "fetch"):
                with self.subTest(code=code, phase=phase):
                    error = HTTPError("https://private/repo", code, "client error", {}, None)
                    with (
                        patch(
                            "zils.feature_commitments.parent",
                            **(
                                {"side_effect": error}
                                if phase == "parent"
                                else {"return_value": manifest}
                            ),
                        ),
                        patch("zils.feature_commitments.fetch", side_effect=error),
                    ):
                        result, _, work = self.run_epoch(
                            config, 2000 + index * 2 + (phase == "fetch"), snapshot
                        )
                    receipt = result["box/d3-mini"]
                    self.assertEqual(receipt["candidates"], [])
                    self.assertEqual(len(receipt["qualification"]), 2)
                    rejection = (
                        json.loads((work / "feature-intake.json").read_text())["rejected"]
                        if phase == "parent"
                        else receipt["rejected"]
                    )
                    self.assertEqual(
                        rejection["1"],
                        "invalid_feature_commitment"
                        if phase == "parent"
                        else "invalid_feature_bundle",
                    )

    def test_transient_http_errors_abort_parent_and_fetch(self):
        config = fixture(self.root)
        snapshot, bundle, _ = self.candidate()
        manifest = feature_commitments.validate_bundle(bundle)
        for index, code in enumerate((408, 429, 500, 503)):
            for phase in ("parent", "fetch"):
                with self.subTest(code=code, phase=phase):
                    error = HTTPError("https://private/repo", code, "transient error", {}, None)
                    with (
                        patch(
                            "zils.feature_commitments.parent",
                            **(
                                {"side_effect": error}
                                if phase == "parent"
                                else {"return_value": manifest}
                            ),
                        ),
                        patch("zils.feature_commitments.fetch", side_effect=error),
                    ):
                        with self.assertRaises((HTTPError, RuntimeError)):
                            self.run_epoch(config, 3000 + index * 2 + (phase == "fetch"), snapshot)
                    self.assertFalse(
                        (self.root / "feature-tracks/box/d3-mini/decision.json").exists()
                    )

    def test_crowned_adapter_uses_incumbent_cache_without_mutating_candidate_evidence(self):
        config = fixture(self.root)
        snapshot, bundle, fetch = self.candidate()
        with (
            patch(
                "zils.feature_commitments.parent",
                return_value=feature_commitments.validate_bundle(bundle),
            ),
            patch("zils.feature_commitments.fetch", side_effect=fetch),
        ):
            self.run_epoch(config, 1000, snapshot)
            qualified, _, _ = self.run_epoch(config, 44200, snapshot)
        winner = qualified["box/d3-mini"]["confirmed_winner"]
        store = self.root / "feature-tracks/box/d3-mini/store" / winner["sha256"]
        original = {path: path.read_bytes() for path in store.glob("*-scores.json")}
        self.assertEqual(len(original), 2)
        config["feature_tracks"]["tracks"]["box/d3-mini"].update(
            incumbent_sha256=winner["sha256"], kings=[winner["hotkey"]]
        )
        config["feature_track_rounds"]["box/d3-mini"]["reference"] = "candidate"
        result, calls, _ = self.run_epoch(config, 45000)
        self.assertEqual(calls, [(65535, "first"), (65535, "second")])
        self.assertEqual(result["box/d3-mini"]["incumbent_sha256"], winner["sha256"])
        self.assertEqual({path: path.read_bytes() for path in original}, original)
        self.assertEqual(len(list(store.glob("incumbent-*-scores.json"))), 2)
        _, calls, _ = self.run_epoch(config, 45001)
        self.assertEqual(calls, [])

    def alias_claims(self, copy_block):
        _, original, _ = self.candidate()
        alias = self.root / "alias"
        shutil.copytree(original, alias)
        manifest = feature_commitments.validate_bundle(alias)
        (alias / "feature.json").write_text(json.dumps(manifest))
        paths = {feature_commitments.bundle_hash(path): path for path in (original, alias)}
        snapshot = {
            "members": {"1": "copy-hotkey", "2": "original-hotkey"},
            "commitments": {
                str(uid): {
                    "data": feature_commitments.format_commitment(
                        feature_commitments.bundle_hash(path), "a" * 40, "owner/model"
                    ),
                    "block": block,
                }
                for uid, path, block in ((1, alias, copy_block), (2, original, 2))
            },
        }

        def parent(claim):
            return feature_commitments.validate_bundle(paths[claim["sha256"]])

        def fetch(claim, destination, **kwargs):
            shutil.copytree(paths[claim["sha256"]], destination)
            return {"manifest": feature_commitments.validate_bundle(destination)}

        return snapshot, manifest["checkpoint_sha256"], parent, fetch

    def test_checkpoint_alias_owner_is_earliest_block_not_lowest_uid(self):
        config = fixture(self.root)
        snapshot, checkpoint, parent, fetch = self.alias_claims(copy_block=3)
        with (
            patch("zils.feature_commitments.parent", side_effect=parent),
            patch("zils.feature_commitments.fetch", side_effect=fetch),
        ):
            result, calls, _ = self.run_epoch(config, 1000, snapshot)
        receipt = result["box/d3-mini"]
        self.assertEqual(receipt["pending_winner"]["uid"], 2)
        self.assertNotIn((1, "first"), calls)
        ownership = self.root / "feature-tracks/box/d3-mini/store/claims" / (checkpoint + ".json")
        self.assertEqual(
            json.loads(ownership.read_text()), {"hotkey": "original-hotkey", "block": 2}
        )

    def test_checkpoint_alias_ties_reject_all_without_recording_an_owner(self):
        config = fixture(self.root)
        snapshot, checkpoint, parent, fetch = self.alias_claims(copy_block=2)
        with (
            patch("zils.feature_commitments.parent", side_effect=parent),
            patch("zils.feature_commitments.fetch", side_effect=fetch),
        ):
            result, calls, _ = self.run_epoch(config, 1000, snapshot)
        receipt = result["box/d3-mini"]
        self.assertEqual(receipt["candidates"], [])
        self.assertEqual(set(receipt["rejected"]), {"1", "2"})
        self.assertEqual(calls, [(65535, "first"), (65535, "second")])
        ownership = self.root / "feature-tracks/box/d3-mini/store/claims" / (checkpoint + ".json")
        self.assertFalse(ownership.exists())

    def test_failed_original_download_cannot_persist_the_copiers_ownership(self):
        config = fixture(self.root)
        snapshot, checkpoint, parent, fetch = self.alias_claims(copy_block=3)
        original_hash = snapshot["commitments"]["2"]["data"].split(":")[1]

        def unavailable(claim, destination, **kwargs):
            if claim["sha256"] == original_hash:
                raise ConnectionError("original download unavailable")
            return fetch(claim, destination, **kwargs)

        with (
            patch("zils.feature_commitments.parent", side_effect=parent),
            patch("zils.feature_commitments.fetch", side_effect=unavailable),
        ):
            with self.assertRaises(RuntimeError):
                self.run_epoch(config, 1000, snapshot)
        ownership = self.root / "feature-tracks/box/d3-mini/store/claims" / (checkpoint + ".json")
        self.assertFalse(ownership.exists())
        with (
            patch("zils.feature_commitments.parent", side_effect=parent),
            patch("zils.feature_commitments.fetch", side_effect=fetch),
        ):
            result, _, _ = self.run_epoch(config, 1001, snapshot)
        self.assertEqual(result["box/d3-mini"]["pending_winner"]["uid"], 2)


if __name__ == "__main__":
    unittest.main()
