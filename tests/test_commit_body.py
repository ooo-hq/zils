"""`zils-miner commit-body`: a soup body dry run, ownership file, validation and the chain write."""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_base_tier_intake import fake_hub, revision
from tests.test_bodies import NAMES, checkpoint, weights
from tests.test_commit_candidate import Chain
from zils import bodies, miner, protocol as wire, testnet

REPO = "alice/qwen-decision-body"


@unittest.skipUnless(
    importlib.util.find_spec("bittensor") and importlib.util.find_spec("kev"), "use .venv-kev"
)
class CommitBodyTest(unittest.TestCase):
    def setUp(self):
        from bittensor.wallet import Wallet

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = root = Path(tmp.name)
        self.wallet = {"path": str(root / "wallets"), "name": "fixture", "hotkey": "miner"}
        self.hotkey = (
            Wallet(**self.wallet)
            .regenerate_hotkey(seed=bytes([7]) * 32, suppress=True)
            .hotkey.ss58_address
        )
        base = weights(0)
        noise = weights(1)
        self.chassis = checkpoint(root / "chassis", base, extras=False, license=None, owner=None)
        self.base = checkpoint(root / "qwen", base, prefix="model.", license=None, owner=None)
        body = {n: (base[n].float() + 0.01 * noise[n].float()).bfloat16() for n in NAMES}
        self.body = checkpoint(root / "body", body, prefix="model.", owner=self.hotkey)
        self.sha256 = bodies.body_hash(self.body, NAMES)
        self.rev = revision("body-upload")
        self.text = bodies.format_commitment(self.sha256, self.rev, REPO)
        self.chain = Chain({self.hotkey: 1})
        self.repos = {REPO: {self.rev: self.body}}
        hub = self.enterContext(wire.local_server(fake_hub(self.repos, [])))
        self.enterContext(patch("zils.commitments.HF_ENDPOINT", hub))
        self.config = root / "zils" / "miner.json"
        self.enterContext(patch.dict(os.environ, {"ZILS_MINER_CONFIG": str(self.config)}))
        self.connects = []

        @contextlib.contextmanager
        def connect(config, *, raw_calls=False):
            testnet.check_config(config)
            self.connects.append(raw_calls)
            yield self.chain

        self.enterContext(patch("zils.testnet.connect", connect))

    def run_miner(self, *argv, local=True, revision=True):
        flags = [
            "--wallet-name",
            self.wallet["name"],
            "--wallet-hotkey",
            self.wallet["hotkey"],
            "--wallet-path",
            self.wallet["path"],
            "--repo",
            REPO,
            "--revision",
            self.rev,
            "--chassis",
            str(self.chassis),
            "--base",
            str(self.base),
        ]
        if local:
            flags += ["--path", str(self.body)]
        if not revision:
            flags.remove("--revision")
            flags.remove(self.rev)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                miner.main(["commit-body", *flags, *argv])
                code = 0
            except SystemExit as stopped:
                code = stopped.code
        return code, output.getvalue()

    def test_dry_run_prints_hash_and_commitment_without_writing(self):
        code, out = self.run_miner()
        self.assertEqual(code, 0, out)
        self.assertIn(f"body {self.sha256}", out)
        self.assertIn(f"commitment {self.text}", out)
        self.assertIn("dry run: nothing written", out)
        self.assertIn("one commitment slot", out)
        self.assertEqual(self.chain.calls, [])
        self.assertFalse(self.config.exists())

    def test_downloads_the_revision_when_no_path_is_given(self):
        def snapshot_download(*, repo_id, revision, allow_patterns, **kwargs):
            self.assertEqual((repo_id, revision), (REPO, self.rev))
            self.assertIn("zils-owner.json", allow_patterns)
            return str(self.body)

        with patch("huggingface_hub.snapshot_download", snapshot_download):
            code, out = self.run_miner(local=False)
        self.assertEqual(code, 0, out)
        self.assertIn(f"commitment {self.text}", out)

    def test_accepts_symlinked_snapshots_for_the_body_chassis_and_base(self):
        from tests.test_bodies import snapshot_like

        for folder in (self.chassis, self.base, self.body):
            snapshot_like(folder)

        def snapshot_download(*, repo_id, revision, allow_patterns, **kwargs):
            return str(self.body)

        with patch("huggingface_hub.snapshot_download", snapshot_download):
            code, out = self.run_miner(local=False)
        self.assertEqual(code, 0, out)
        self.assertIn(f"commitment {self.text}", out)

    def check_only(self, *argv):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                miner.main(
                    [
                        "commit-body",
                        "--check",
                        "--chassis",
                        str(self.chassis),
                        "--base",
                        str(self.base),
                        *argv,
                    ]
                )
                code = 0
            except SystemExit as stopped:
                code = stopped.code
        return code, output.getvalue()

    def test_check_runs_the_local_checks_without_a_wallet_repo_or_owner_file(self):
        (self.body / "zils-owner.json").unlink()
        code, out = self.check_only("--path", str(self.body))
        self.assertEqual(code, 0, out)
        self.assertIn(f"body {self.sha256} passes the local checks", out)
        self.assertIn("--write-owner", out)  # the next step
        self.assertFalse((self.body / "zils-owner.json").exists())
        self.assertEqual(self.connects, [])

    def test_check_explains_a_text_only_merge(self):
        from tests.test_bodies import weights

        text = {
            n.removeprefix("language_model."): t
            for n, t in weights(0).items()
            if n.startswith("language_model.")
        }
        folder = checkpoint(self.root / "text-only", text, prefix="model.", owner=None)
        code, out = self.check_only("--path", str(folder))
        self.assertEqual(code, 1, out)
        self.assertIn("text-only checkpoint", out)
        self.assertNotIn("Traceback", out)

    def test_check_needs_a_path_and_refuses_to_write(self):
        code, out = self.check_only()
        self.assertEqual(code, 1, out)
        self.assertIn("--check needs --path", out)
        for flag in ("--yes", "--write-owner"):
            code, out = self.check_only("--path", str(self.body), flag)
            self.assertEqual(code, 1, out)
            self.assertIn("--check only reads", out)

    def test_refuses_without_an_ownership_file(self):
        (self.body / "zils-owner.json").unlink()
        code, out = self.run_miner("--yes")
        self.assertEqual(code, 1, out)
        self.assertIn("zils-owner.json", out)
        self.assertIn("--write-owner", out)
        self.assertEqual(self.chain.calls, [])

    def test_refuses_an_ownership_file_naming_another_hotkey(self):
        (self.body / "zils-owner.json").write_text(json.dumps({"hotkey": "5Other"}))
        code, out = self.run_miner("--yes")
        self.assertEqual(code, 1, out)
        self.assertIn("not the committing hotkey", out)
        self.assertEqual(self.chain.calls, [])

    def test_write_owner_writes_the_file_and_asks_for_an_upload(self):
        (self.body / "zils-owner.json").unlink()
        code, out = self.run_miner("--write-owner")
        self.assertEqual(code, 0, out)
        owner = json.loads((self.body / "zils-owner.json").read_text())
        self.assertEqual(owner, {"hotkey": self.hotkey})
        self.assertIn("upload", out)
        self.assertIn(f"body {self.sha256}", out)
        self.assertNotIn("zilsb2:", out)
        self.assertEqual(self.chain.calls, [])
        (self.body / "zils-owner.json").unlink()
        code, out = self.run_miner("--write-owner", revision=False)
        self.assertEqual(code, 0, out)
        self.assertTrue((self.body / "zils-owner.json").is_file())
        code, out = self.run_miner("--write-owner", local=False)
        self.assertEqual(code, 1)
        self.assertIn("--write-owner needs --path", out)
        code, out = self.run_miner("--write-owner", "--yes")
        self.assertEqual(code, 1)

    def test_refuses_a_body_failing_validation(self):
        readme = self.body / "README.md"
        readme.write_text(readme.read_text().replace("apache-2.0", "gpl-3.0"))
        code, out = self.run_miner("--yes")
        self.assertEqual(code, 1, out)
        self.assertIn("license", out)
        self.assertEqual(self.chain.calls, [])
        readme.write_text(readme.read_text().replace("gpl-3.0", "apache-2.0"))
        far = checkpoint(self.root / "far", weights(5), prefix="model.", owner=self.hotkey)
        code, out = self.run_miner("--yes", "--path", str(far))
        self.assertEqual(code, 1, out)
        self.assertIn("change from the base", out)
        self.assertEqual(self.chain.calls, [])

    def test_yes_writes_the_commitment_once(self):
        self.config.parent.mkdir(parents=True)
        saved = {"repo": "alice/d3-candidate", "revision": "c" * 40}
        self.config.write_text(json.dumps(saved))
        with patch("zils.testnet.write_commitment", wraps=testnet.write_commitment) as write:
            code, out = self.run_miner("--yes")
        self.assertEqual(code, 0, out)
        write.assert_called_once()
        self.assertEqual(write.call_args.args[3], self.text)
        self.assertEqual(self.connects[-1], True)
        stored = testnet.read_commitment(self.chain, 579, self.hotkey, self.chain.block)
        self.assertEqual(stored["data"], self.text)
        self.assertIn("committed at block 301", out)
        # The saved base-candidate setup is not touched by a body commit.
        self.assertEqual(json.loads(self.config.read_text()), saved)
        code, out = self.run_miner("--yes")
        self.assertEqual(code, 0, out)
        self.assertIn("already committed at block 301; nothing to do", out)
        self.assertEqual(len(self.chain.calls), 1)

    def test_needs_a_revision_and_never_takes_the_saved_candidate_one(self):
        self.config.parent.mkdir(parents=True)
        self.config.write_text(json.dumps({"repo": "alice/d3", "revision": self.rev}))
        code, out = self.run_miner("--yes", revision=False)
        self.assertEqual(code, 1, out)
        self.assertIn("--revision is needed", out)
        self.assertEqual(self.chain.calls, [])

    def test_yes_refuses_a_local_folder_that_differs_from_the_revision(self):
        other = checkpoint(self.root / "other", weights(0), prefix="model.", owner=self.hotkey)
        self.repos[REPO][self.rev] = other
        code, out = self.run_miner("--yes")
        self.assertEqual(code, 1, out)
        self.assertIn("differs from the local body", out)
        self.assertEqual(self.chain.calls, [])

    def test_warns_before_replacing_another_commitment_and_refuses_unregistered(self):
        with patch(
            "zils.miner.hotkey_state",
            return_value={"uid": 1, "commitment": {"block": 10, "data": "zils1:old"}},
        ):
            code, out = self.run_miner()
        self.assertEqual(code, 0, out)
        self.assertIn("replaces zils1:old", out)
        self.chain.uids = {}
        code, out = self.run_miner("--yes")
        self.assertEqual(code, 1)
        self.assertIn("is not registered on testnet subnet 579", out)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(
    importlib.util.find_spec("bittensor") and importlib.util.find_spec("kev"), "use .venv-kev"
)
class CommitBodyPoolTest(unittest.TestCase):
    """`commit-body --pool`: lite, flash and z3 bodies, with their own parent, chassis and cap."""

    setUp = CommitBodyTest.setUp
    run_miner = CommitBodyTest.run_miner

    def test_a_lite_body_commits_as_zilsb3_with_the_lite_pool(self):
        code, out = self.run_miner("--pool", "d3-lite")
        self.assertEqual(code, 0, out)
        text = bodies.format_pool_commitment("d3-lite", self.sha256, self.rev, REPO)
        self.assertTrue(text.startswith("zilsb3:d3-lite:"))
        self.assertIn(f"commitment {text}", out)
        self.assertIn("pool d3-lite", out)
        code, out = self.run_miner("--pool", "d3", "--yes")
        self.assertEqual(code, 0, out)
        stored = testnet.read_commitment(self.chain, 579, self.hotkey, self.chain.block)
        self.assertEqual(
            stored["data"], bodies.format_pool_commitment("d3", self.sha256, self.rev, REPO)
        )

    def test_mini_stays_the_default_and_keeps_zilsb2(self):
        code, out = self.run_miner("--pool", "d3-mini")
        self.assertEqual(code, 0, out)
        self.assertIn(f"commitment {self.text}", out)

    def test_each_pool_downloads_its_own_chassis_and_parent(self):
        from zils import d3, models

        seen = []

        def pinned(download=False, model=models.D3):
            seen.append(("chassis", model))
            return self.chassis

        def snapshot_download(*, repo_id, revision, allow_patterns, **kwargs):
            seen.append(("parent", repo_id, revision))
            return str(self.base)

        for pool, model in (
            ("d3-lite", "d3-lite-0.8b-v1"),
            ("d3-flash", "d3-flash-8.4b-v1"),
            ("d3", "d3-ultra-26b-v1"),
        ):
            seen.clear()
            args = type("Args", (), {"chassis": None, "base": None, "pool": pool})()
            with (
                patch.object(d3, "pinned_base_path", pinned),
                patch("huggingface_hub.snapshot_download", snapshot_download),
            ):
                miner.body_references(args)
            self.assertEqual(seen, [("chassis", model), ("parent", *bodies.parent(pool))])

    def test_the_download_respects_the_pools_byte_cap(self):
        listed = {
            "zils-owner.json": {"size": 40, "sha256": None},
            "model.safetensors": {"size": 3 * 1024**3, "sha256": None},
        }
        with patch("zils.commitments.revision_files", return_value=listed):
            with self.assertRaisesRegex(ValueError, "over"):
                miner.download_body(REPO, self.rev, self.hotkey, pool="d3-lite")
            with (
                patch("huggingface_hub.snapshot_download", return_value=str(self.body)),
            ):
                miner.download_body(REPO, self.rev, self.hotkey, pool="d3-flash")
