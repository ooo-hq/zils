"""Immutable box adapters stay separate from base claims and reject untrusted artifacts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import zils
from tests.promoted_profile import PROMOTED, promoted_profile
from tests.test_base_tier_intake import fake_hub, revision
from zils import commitments, feature_commitments as features, models, protocol


def adapter(root, model=PROMOTED, kind="adapter"):
    root.mkdir()
    models.write_metadata(root, model=model, kind=kind)
    if kind == "adapter":
        profile = models.spec(model)
        count = models.pins(model).get("target_modules", 152)
        config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "peft_type": "LORA",
            "base_model_name_or_path": profile["base"],
            "revision": profile["base_revision"],
            "modules_to_save": None,
            "use_dora": False,
            "use_rslora": False,
            "target_modules": [f"language_model.layers.{i}.self_attn.q_proj" for i in range(count)],
        }
        (root / "adapter_config.json").write_text(json.dumps(config))
        (root / "adapter_model.safetensors").write_bytes(b"tensor fixture")
    return root


class FeatureCommitmentsTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(promoted_profile())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = adapter(self.root / "adapter")

    def pack(self, name="bundle"):
        output = self.root / name
        manifest = features.pack(self.source, output, "box/d3-mini")
        return output, manifest

    def test_pack_binds_manifest_and_artifact_bytes_but_excludes_unrelated_files(self):
        (self.source / "training.log").write_text("local only")
        output, manifest = self.pack()
        self.assertEqual(
            manifest,
            {
                "version": "zils-feature-adapter/v1",
                "track_id": "box/d3-mini",
                "model": models.spec(PROMOTED),
                "decoder": "box-edges-101-presence/v1",
                "checkpoint_sha256": zils.checkpoint_hash(self.source),
            },
        )
        self.assertEqual(
            set(p.name for p in output.iterdir()),
            {
                "feature.json",
                "model.json",
                "adapter_config.json",
                "adapter_model.safetensors",
            },
        )
        original = features.bundle_hash(output)
        (output / "README.md").write_text("not an artifact")
        self.assertEqual(features.bundle_hash(output), original)
        (output / "feature.json").write_text(json.dumps(manifest, indent=4))
        self.assertNotEqual(features.bundle_hash(output), original)
        (output / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            features.validate_bundle(output)

    def test_pack_preserves_existing_destination(self):
        output, _ = self.pack()
        with self.assertRaises(FileExistsError):
            features.pack(self.source, output, "box/d3-mini")
        features.validate_bundle(output)

    def test_wrong_track_parent_decoder_and_extra_manifest_fields_fail(self):
        output, original = self.pack()
        for changes in (
            {"track_id": "box/d3-lite"},
            {"track_id": "new/d3-mini"},
            {"model": models.spec(models.D3)},
            {"decoder": "custom-code"},
            {"version": "next"},
            {"code": "run.py"},
            {"checkpoint_sha256": "bad"},
        ):
            with self.subTest(changes=changes):
                (output / "feature.json").write_text(json.dumps(original | changes))
                with self.assertRaises(ValueError):
                    features.validate_bundle(output)

    def test_rejects_symlinks_and_nonunit_temperature(self):
        output, _ = self.pack()
        artifact = output / "adapter_model.safetensors"
        artifact.unlink()
        artifact.symlink_to(self.source / artifact.name)
        with self.assertRaisesRegex(ValueError, "symlink"):
            features.validate_bundle(output)
        with self.assertRaisesRegex(ValueError, "symlink"):
            features.pack(output, self.root / "bad", "box/d3-mini")
        models.write_metadata(self.source, model=PROMOTED, temperature=0.2)
        with self.assertRaisesRegex(ValueError, "temperature"):
            features.pack(self.source, self.root / "calibrated", "box/d3-mini")

    def test_only_operator_can_pack_and_validate_a_base_reference(self):
        source = adapter(self.root / "base", kind="base")
        with self.assertRaisesRegex(ValueError, "adapter"):
            features.pack(source, self.root / "not-allowed", "box/d3-mini")
        output = self.root / "reference"
        features.pack(source, output, "box/d3-mini", allow_base=True)
        self.assertEqual(features.validate_bundle(output, allow_base=True)["model"]["id"], PROMOTED)
        self.assertEqual(len(features.bundle_hash(output)), 64)
        with self.assertRaisesRegex(ValueError, "adapter"):
            features.validate_bundle(output)

    def test_adapter_recipe_cannot_enable_code_or_change_language_lora_scope(self):
        path = self.source / "adapter_config.json"
        original = json.loads(path.read_text())
        for changes in (
            {"r": 64},
            {"peft_type": "PREFIX_TUNING"},
            {"use_dora": True},
            {"modules_to_save": ["lm_head"]},
            {"target_modules": ["visual.q_proj"]},
            {"auto_mapping": {"parent_library": "evil", "base_model_class": "Exploit"}},
            {"rank_pattern": {"q_proj": 64}},
            {"unknown_execution_hook": "run.py"},
        ):
            with self.subTest(changes=changes):
                path.write_text(json.dumps(original | changes))
                with self.assertRaises(ValueError):
                    features.pack(self.source, self.root / "rejected", "box/d3-mini")

    def test_claim_namespace_and_earliest_duplicate_rules(self):
        sha, rev = "a" * 64, "b" * 40
        text = features.format_commitment(sha, rev, "owner/adapter")
        self.assertTrue(text.startswith("zilsbox1:"))
        with self.assertRaises(ValueError):
            commitments.parse_commitment(text)
        snapshot = {
            "commitments": {
                "1": {"block": 10, "data": text},
                "2": {"block": 11, "data": text},
                "3": {"block": 12, "data": commitments.format_commitment(sha, rev, "owner/base")},
                "4": {"block": 12, "data": "random"},
                "5": {"block": 12, "data": "zilsbox1:bad"},
            }
        }
        rows = features.candidates(snapshot)
        self.assertEqual(rows[1]["sha256"], sha)
        self.assertIn("first by uid 1", rows[2]["error"])
        self.assertEqual(set(rows), {1, 2, 5})
        snapshot["commitments"]["2"]["block"] = 10
        self.assertIn("several hotkeys", features.candidates(snapshot)[1]["error"])
        for bad in (text.replace(rev, "main"), text.replace("owner/adapter", "../bad")):
            with self.assertRaises(ValueError):
                features.parse_commitment(bad)

    def test_pinned_fetch_checks_track_parent_and_complete_hash(self):
        output, manifest = self.pack()
        requested = []
        claim = {
            "sha256": features.bundle_hash(output),
            "repo": "owner/box",
            "revision": revision("box"),
        }
        repos = {claim["repo"]: {claim["revision"]: output}}
        with protocol.local_server(fake_hub(repos, requested)) as endpoint:
            self.assertEqual(features.parent(claim, endpoint=endpoint), manifest)
            self.assertEqual(requested, [("owner/box", "feature.json")])
            for track, model in (("box/d3-lite", PROMOTED), ("box/d3-mini", models.D3)):
                with self.assertRaises(ValueError):
                    features.fetch(
                        claim,
                        self.root / "rejected",
                        track_id=track,
                        model=model,
                        endpoint=endpoint,
                    )
            self.assertFalse((self.root / "rejected").exists())
            target = self.root / "download"
            receipt = features.fetch(
                claim, target, track_id="box/d3-mini", model=PROMOTED, endpoint=endpoint
            )
            self.assertEqual(receipt["manifest"], manifest)
            self.assertEqual(features.bundle_hash(target), claim["sha256"])
            with self.assertRaisesRegex(ValueError, "commitment"):
                features.fetch(
                    claim | {"sha256": "f" * 64},
                    self.root / "copy",
                    track_id="box/d3-mini",
                    model=PROMOTED,
                    endpoint=endpoint,
                )

    def test_download_and_manifest_are_bounded(self):
        output, _ = self.pack()
        claim = {
            "sha256": features.bundle_hash(output),
            "repo": "owner/box",
            "revision": revision("box"),
        }
        repos = {claim["repo"]: {claim["revision"]: output}}
        with protocol.local_server(fake_hub(repos, [])) as endpoint:
            with patch("zils.models.MAX_ADAPTER_BYTES", 10):
                with self.assertRaisesRegex(ValueError, "budget"):
                    features.fetch(
                        claim,
                        self.root / "too-big",
                        track_id="box/d3-mini",
                        model=PROMOTED,
                        endpoint=endpoint,
                    )
            (output / "feature.json").write_bytes(b" " * 16385)
            with self.assertRaisesRegex(ValueError, "16 KiB"):
                features.parent(claim, endpoint=endpoint)

    def test_malformed_manifest_types_fail_as_invalid_input(self):
        output, original = self.pack()
        for value in (None, [], original | {"model": []}, original | {"model": {"id": []}}):
            with self.subTest(value=value):
                (output / "feature.json").write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    features.validate_bundle(output)

    def test_metadata_size_and_download_deadline_fail_before_acceptance(self):
        output, _ = self.pack()
        claim = {
            "sha256": features.bundle_hash(output),
            "repo": "owner/box",
            "revision": revision("box"),
        }
        repos = {claim["repo"]: {claim["revision"]: output}}
        requested = []
        with protocol.local_server(fake_hub(repos, requested)) as endpoint:
            with patch("zils.commitments.DOWNLOAD_SECONDS", 0):
                with self.assertRaisesRegex(ValueError, "deadline"):
                    features.fetch(
                        claim,
                        self.root / "expired",
                        track_id="box/d3-mini",
                        model=PROMOTED,
                        endpoint=endpoint,
                    )
            (output / "adapter_config.json").write_bytes(b" " * 16385)
            requested.clear()
            with self.assertRaisesRegex(ValueError, "metadata"):
                features.fetch(
                    claim,
                    self.root / "metadata",
                    track_id="box/d3-mini",
                    model=PROMOTED,
                    endpoint=endpoint,
                )
            self.assertEqual(requested, [("owner/box", "feature.json")])


if __name__ == "__main__":
    unittest.main()
