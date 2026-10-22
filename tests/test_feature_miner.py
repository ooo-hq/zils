"""Feature submissions require explicit writes and verify the immutable remote bundle."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.promoted_profile import PROMOTED, promoted_profile
from tests.test_base_tier_intake import fake_hub, revision
from tests.test_commit_candidate import Chain
from tests.test_feature_commitments import adapter
from zils import feature_commitments as features, feature_miner, protocol, testnet


class FeatureMinerTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(promoted_profile())
        from bittensor.wallet import Wallet

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / "bundle"
        self.source = adapter(self.root / "adapter")
        features.pack(self.source, self.bundle, "box/d3-mini")
        self.sha256 = features.bundle_hash(self.bundle)
        self.keys = Wallet(path=str(self.root / "wallets"), name="fixture", hotkey="miner")
        self.hotkey = self.keys.regenerate_hotkey(
            seed=bytes([7]) * 32, suppress=True
        ).hotkey.ss58_address
        self.chain = Chain({self.hotkey: 1})
        self.repos, self.uploads, self.requested = {}, [], []
        endpoint = self.enterContext(protocol.local_server(fake_hub(self.repos, self.requested)))
        self.enterContext(patch("zils.commitments.HF_ENDPOINT", endpoint))
        self.enterContext(patch("zils.miner.load_config", return_value={}))

        @contextlib.contextmanager
        def connect(config, *, raw_calls=False):
            testnet.check_config(config)
            yield self.chain

        self.enterContext(patch("zils.testnet.connect", connect))
        repos, uploads, root = self.repos, self.uploads, self.root

        class HfApi:
            def create_repo(self, repo_id, *, repo_type, private, exist_ok):
                assert (repo_type, private, exist_ok) == ("model", False, True)

            def upload_folder(self, *, repo_id, repo_type, folder_path, allow_patterns, **kwargs):
                rev = revision(f"upload-{len(uploads)}")
                target = root / "hub" / rev
                target.mkdir(parents=True)
                for name in allow_patterns:
                    shutil.copyfile(Path(folder_path) / name, target / name)
                repos.setdefault(repo_id, {})[rev] = target
                uploads.append((repo_id, sorted(allow_patterns)))
                return SimpleNamespace(oid=rev)

        self.enterContext(patch("huggingface_hub.HfApi", HfApi))

    def run_cli(self, *extra):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                result = feature_miner.main(
                    [
                        "commit",
                        "--candidate",
                        str(self.bundle),
                        "--repo",
                        "alice/box",
                        "--wallet-path",
                        str(self.root / "wallets"),
                        "--wallet-name",
                        "fixture",
                        "--wallet-hotkey",
                        "miner",
                        *extra,
                    ]
                )
                code = 0
            except SystemExit as stopped:
                code, result = stopped.code, None
        return code, output.getvalue(), result

    def test_dry_run_has_no_hub_or_chain_mutation_and_yes_returns_verified_receipt(self):
        code, out, result = self.run_cli()
        self.assertEqual(code, 0, out)
        self.assertIsNone(result)
        self.assertEqual((self.repos, self.uploads, self.chain.calls), ({}, [], []))
        self.assertIn("dry run", out)
        self.assertIn("one active commitment per hotkey", out)
        code, out, result = self.run_cli("--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(
            self.uploads,
            [
                (
                    "alice/box",
                    [
                        "adapter_config.json",
                        "adapter_model.safetensors",
                        "feature.json",
                        "model.json",
                    ],
                )
            ],
        )
        text = features.format_commitment(self.sha256, revision("upload-0"), "alice/box")
        self.assertEqual(
            testnet.read_commitment(self.chain, 579, self.hotkey, self.chain.block),
            {"block": 301, "data": text},
        )
        self.assertEqual(result["commitment"], text)
        self.assertEqual(result["block"], 301)
        self.assertEqual(result["track_id"], "box/d3-mini")
        self.assertEqual(result["bundle_sha256"], self.sha256)
        code, out, again = self.run_cli("--revision", revision("upload-0"), "--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(again, result)
        self.assertEqual(len(self.chain.calls), 1)

    def test_remote_manifest_tampering_cannot_write_even_when_sizes_match(self):
        rev = revision("remote")
        remote = self.root / "remote"
        shutil.copytree(self.bundle, remote)
        path = remote / "feature.json"
        manifest = json.loads(path.read_text())
        manifest["checkpoint_sha256"] = "f" * 64
        path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        self.assertEqual(path.stat().st_size, (self.bundle / "feature.json").stat().st_size)
        self.repos["alice/box"] = {rev: remote}
        code, out, _ = self.run_cli("--revision", rev, "--yes")
        self.assertEqual(code, 1, out)
        self.assertEqual(self.chain.calls, [])
        self.assertIn("checkpoint hash", out)

    def test_write_requires_registration_and_rejects_branch_or_base_reference(self):
        code, out, _ = self.run_cli("--revision", "main", "--yes")
        self.assertEqual(code, 1, out)
        self.chain.uids = {}
        code, out, _ = self.run_cli("--yes")
        self.assertEqual(code, 1, out)
        self.assertIn("not registered", out)
        source = adapter(self.root / "base", kind="base")
        self.bundle = self.root / "reference"
        features.pack(source, self.bundle, "box/d3-mini", allow_base=True)
        code, out, _ = self.run_cli("--yes")
        self.assertEqual(code, 1, out)
        self.assertIn("adapter", out)
        self.assertEqual((self.uploads, self.chain.calls), ([], []))

    def test_post_write_readback_failure_is_not_a_success_receipt(self):
        with patch("zils.testnet.write_commitment", return_value=None):
            code, out, result = self.run_cli("--yes")
        self.assertEqual(code, 1, out)
        self.assertIsNone(result)
        self.assertIn("chain does not show", out)

    def test_pack_cli_writes_a_usable_bundle(self):
        target = self.root / "cli-bundle"
        with contextlib.redirect_stdout(io.StringIO()):
            feature_miner.main(
                [
                    "pack",
                    "--checkpoint",
                    str(self.source),
                    "--out",
                    str(target),
                    "--track",
                    "box/d3-mini",
                ]
            )
        self.assertEqual(features.validate_bundle(target)["model"]["id"], PROMOTED)


if __name__ == "__main__":
    unittest.main()
