"""`zils commit-body`: a soup body dry run, ownership file, validation and the chain write."""

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
