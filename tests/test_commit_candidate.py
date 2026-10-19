"""`zils setup`, `doctor` and `commit-candidate` against a fake chain and a fake Hugging Face."""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import zils
from tests.test_base_tier_intake import candidate, fake_hub, revision
from zils import commitments, miner, models, protocol as wire, testnet


class Chain:
    """Uids, Keys and CommitmentOf, with set_commitment applied at the next block."""

    network, endpoint = "test", testnet.ENDPOINT

    def __init__(self, uids):
        self.block, self.uids, self.records, self.calls = 300, uids, {}, []

    def query(self, item, params, *, block=None):
        assert params[0] == 579 and block == self.block
        if item[1] == "Uids":
            return self.uids.get(params[1])
        if item[1] == "Keys":
            return {uid: hotkey for hotkey, uid in self.uids.items()}.get(params[1])
        return self.records.get(params[1])

    def submit_call(self, call, wallet, *, signer, **kwargs):
        from bittensor.result import ExtrinsicResult

        assert (call.module, call.function, signer) == ("Commitments", "set_commitment", "hotkey")
        assert call.params["netuid"] == 579 and kwargs["wait_for_finalization"]
        self.calls.append(call)
        self.block += 1
        self.records[wallet.hotkey.ss58_address] = {
            "block": self.block,
            "deposit": 0,
            "info": call.params["info"],
        }
        return ExtrinsicResult(True)


@unittest.skipUnless(
    importlib.util.find_spec("bittensor") and importlib.util.find_spec("kev"), "use .venv-kev"
)
class CommitCandidateTest(unittest.TestCase):
    def setUp(self):
        from bittensor.wallet import Wallet

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.root = root
        self.candidate = candidate(root / "candidate", 0.85)
        self.sha256 = zils.checkpoint_hash(self.candidate)
        self.wallet = {"path": str(root / "wallets"), "name": "fixture", "hotkey": "miner"}
        self.hotkey = (
            Wallet(**self.wallet)
            .regenerate_hotkey(seed=bytes([7]) * 32, suppress=True)
            .hotkey.ss58_address
        )
        self.chain = Chain({self.hotkey: 1})
        self.repos, self.requested = {}, []
        hub = self.enterContext(wire.local_server(fake_hub(self.repos, self.requested)))
        self.enterContext(patch("zils.commitments.HF_ENDPOINT", hub))
        self.enterContext(patch("zils.d3.validate_candidate", Mock()))
        self.connects = []
        self.config = root / "zils" / "miner.json"
        self.enterContext(patch.dict(os.environ, {"ZILS_MINER_CONFIG": str(self.config)}))
        self.d3 = root / "venv-d3" / "python"
        self.d3.parent.mkdir()
        self.d3.touch()
        self.enterContext(patch("zils.miner.d3_python", lambda: self.d3))
        self.runtime = SimpleNamespace(returncode=0, stderr="")
        self.enterContext(patch("zils.miner.subprocess.run", lambda *a, **k: self.runtime))
        self.enterContext(patch("huggingface_hub.get_token", lambda: "hf_fixture"))

        @contextlib.contextmanager
        def connect(config, *, raw_calls=False):
            testnet.check_config(config)
            self.connects.append(raw_calls)
            yield self.chain

        self.enterContext(patch("zils.testnet.connect", connect))
        repos, uploads = self.repos, []
        self.uploads = uploads

        class HfApi:
            def create_repo(self, repo_id, *, repo_type, private, exist_ok):
                assert repo_type == "model" and private is False and exist_ok

            def upload_folder(self, *, repo_id, repo_type, folder_path, allow_patterns, **kwargs):
                rev = revision(f"upload-{len(uploads)}")
                target = root / "hub" / rev
                target.mkdir(parents=True)
                for name in allow_patterns:
                    shutil.copy(Path(folder_path) / name, target / name)
                repos.setdefault(repo_id, {})[rev] = target
                uploads.append((repo_id, sorted(allow_patterns)))
                return SimpleNamespace(oid=rev)

        self.enterContext(patch("huggingface_hub.HfApi", HfApi))

    def run_miner(self, *argv, wallet=True):
        flags = wallet * [
            "--wallet-name",
            self.wallet["name"],
            "--wallet-hotkey",
            self.wallet["hotkey"],
            "--wallet-path",
            self.wallet["path"],
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                miner.main([argv[0], *flags, *argv[1:]])
                code = 0
            except SystemExit as stopped:
                code = stopped.code
        return code, output.getvalue()

    def test_dry_run_upload_commit_and_no_second_commit(self):
        repo = "alice/d3-mini-base-candidate-round-three"
        candidate_flags = ("--candidate", str(self.candidate), "--repo", repo)
        code, out = self.run_miner("commit-candidate", *candidate_flags)
        self.assertEqual(code, 0, out)
        self.assertIn("dry run: nothing uploaded or written", out)
        self.assertIn(f"commitment zils1:{self.sha256}:", out)
        self.assertEqual((self.uploads, self.chain.calls), ([], []))

        code, out = self.run_miner("commit-candidate", *candidate_flags, "--yes")
        self.assertEqual(code, 0, out)
        rev = revision("upload-0")
        self.assertEqual(self.uploads, [(repo, sorted(models.D3_CANDIDATE_FILES))])
        self.assertIn(f"uploaded {repo} at {rev}", out)
        self.assertIn("committed at block 301", out)
        self.assertEqual(self.connects[-1], True)
        text = commitments.format_commitment(self.sha256, rev, repo)
        self.assertGreater(len(text), 128)
        self.assertEqual(len(self.chain.calls[0].params["info"]["fields"]), 2)
        stored = testnet.read_commitment(self.chain, 579, self.hotkey, self.chain.block)
        self.assertEqual(stored, {"block": 301, "data": text})
        # What a validator snapshot then reads is this miner's candidate.
        snapshot = {"commitments": {"1": stored}}
        self.assertEqual(commitments.candidates(snapshot, "f" * 64)[1]["revision"], rev)

        # Committing the same thing again would move the block; the command refuses to.
        code, out = self.run_miner("commit-candidate", *candidate_flags, "--revision", rev, "--yes")
        self.assertEqual(code, 0, out)
        self.assertIn("already committed at block 301; nothing to do", out)
        self.assertEqual(len(self.chain.calls), 1)

    def test_refuses_bad_revision_branch_mismatch_and_unregistered(self):
        flags = ("--candidate", str(self.candidate), "--repo", "alice/d3", "--yes")
        code, out = self.run_miner("commit-candidate", *flags, "--revision", "main")
        self.assertEqual(code, 1)
        self.assertIn("40-hex commit sha", out)
        code, out = self.run_miner("commit-candidate", *flags, "--revision", revision("none"))
        self.assertEqual(code, 1)
        self.assertIn("HTTP Error 404", out)
        other = candidate(self.root / "other", 0.5)
        self.repos["alice/d3"] = {revision("other"): other}
        code, out = self.run_miner("commit-candidate", *flags, "--revision", revision("other"))
        self.assertEqual(code, 1)
        self.assertIn("from the local candidate", out)
        self.assertEqual(self.chain.calls, [])
        self.chain.uids = {}
        code, out = self.run_miner("commit-candidate", *flags)
        self.assertEqual(code, 1)
        self.assertIn("is not registered on testnet subnet 579", out)

    def test_doctor_stops_at_the_first_failing_step_with_its_fix(self):
        code, out = self.run_miner("doctor")
        self.assertEqual(code, 1)
        self.assertEqual(out.count("✓ "), 4)
        self.assertIn("✗ candidate\n  no --candidate", out)
        self.assertIn(f"{self.d3} -m zils.d3 train", out)

        code, out = self.run_miner("doctor", "--candidate", str(self.candidate))
        self.assertEqual(out.count("✓ "), 5)
        self.assertIn("✗ hugging face\n  no --repo", out)

        flags = ("--candidate", str(self.candidate), "--repo", "alice/d3")
        code, out = self.run_miner("doctor", *flags)
        self.assertIn("✗ hugging face\n  not uploaded yet", out)
        self.assertIn("zils commit-candidate", out)
        self.assertIn("--repo alice/d3 \\\n    --yes", out)
        with patch("huggingface_hub.get_token", lambda: None):
            code, out = self.run_miner("doctor", *flags)
        self.assertIn("✗ hugging face\n  not logged in to Hugging Face", out)

        self.run_miner("commit-candidate", *flags, "--yes")
        rev = revision("upload-0")
        code, out = self.run_miner("doctor", *flags, "--revision", revision("none"))
        self.assertIn("✗ hugging face\n  HTTP Error 404", out)

        code, out = self.run_miner("doctor", *flags, "--revision", rev)
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("✓ "), 7)
        self.assertIn(f"Ready. uid 1 committed {self.sha256[:12]} at block 301", out)

        self.chain.records.clear()
        code, out = self.run_miner("doctor", *flags, "--revision", rev)
        self.assertIn("✗ commitment\n  the chain holds no commitment", out)
        self.assertIn(f"--revision {rev} \\\n    --yes", out)

        self.chain.uids = {}
        code, out = self.run_miner("doctor", *flags, "--revision", rev)
        self.assertEqual(out.count("✓ "), 3)
        self.assertIn("✗ registration\n  hotkey", out)
        self.assertIn("tx burned-register --netuid 579", out)

        code, out = self.run_miner("doctor", "--wallet-hotkey", "absent")
        self.assertIn("✗ wallet hotkey\n  no hotkey file", out)

        real = miner.importlib.import_module

        def without_hub(name, *rest):
            if name == "huggingface_hub":
                raise ImportError("No module named huggingface_hub")
            return real(name, *rest)

        with patch("zils.miner.importlib.import_module", without_hub):
            code, out = self.run_miner("doctor")
        self.assertEqual(out.count("✓ "), 0)
        self.assertIn("✗ requirements\n  missing huggingface_hub", out)

        self.runtime = SimpleNamespace(returncode=1, stderr="Traceback\nOSError: d3 requires X")
        code, out = self.run_miner("doctor")
        self.assertEqual(out.count("✓ "), 1)
        self.assertIn("✗ d3 runtime\n  " + f"{self.d3}: OSError: d3 requires X", out)
        self.d3.unlink()
        code, out = self.run_miner("doctor")
        self.assertIn("✗ d3 runtime\n  no d3 training runtime", out)
        self.assertIn("install.sh | sh", out)

    def test_setup_saves_answers_that_later_commands_default_to(self):
        code, out = self.run_miner("doctor", wallet=False)
        self.assertEqual(code, 1)
        self.assertIn("no wallet configured; run zils setup", out)

        repo = "alice/d3"
        answers = iter([self.wallet["path"], "", "", str(self.candidate), repo])
        with patch("builtins.input", lambda prompt: next(answers)):
            code, out = self.run_miner("setup", wallet=False)
        self.assertIn("wallets there: fixture", out)
        self.assertIn("hotkeys: miner", out)
        self.assertIn("✗ hugging face\n  not uploaded yet", out)
        saved = json.loads(self.config.read_text())
        self.assertEqual(saved["wallet_name"], "fixture")
        self.assertEqual(saved["wallet_hotkey"], "miner")
        self.assertEqual(saved["candidate"], str(self.candidate.resolve()))

        code, out = self.run_miner("commit-candidate", "--yes", wallet=False)
        self.assertEqual(code, 0, out)
        rev = revision("upload-0")
        self.assertEqual(json.loads(self.config.read_text())["revision"], rev)
        code, out = self.run_miner("doctor", wallet=False)
        self.assertEqual(code, 0, out)
        self.assertIn("Ready. uid 1", out)

        # Retraining into a new folder drops the stale revision, so doctor asks for an upload.
        other = candidate(self.root / "retrained", 0.5)
        answers = iter(["", "", "", str(other), ""])
        with patch("builtins.input", lambda prompt: next(answers)):
            code, out = self.run_miner("setup", wallet=False)
        self.assertNotIn("revision", json.loads(self.config.read_text()))
        self.assertIn("✗ hugging face\n  not uploaded yet", out)


if __name__ == "__main__":
    unittest.main()
