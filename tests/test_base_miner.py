import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from zils import base_miner


class BaseMinerTest(unittest.TestCase):
    def test_training_failure_releases_gpu_and_removes_serving_yield(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gpu.lock").touch()
            config = {
                "state": str(root / "state"),
                "interval_seconds": 86400,
                "compute_lock": str(root / "gpu.lock"),
                "yield_file": str(root / "yield"),
                "hotkey": "miner",
                "model": "d3-mini-4.5b-v1",
                "data_sha256": "a" * 64,
                "runtime_python": "python",
                "data": "/public/train",
                "reference": "/base",
                "max_training_seconds": 10,
            }
            with (
                patch("zils.base_miner.validate", return_value=(None, {})),
                patch("huggingface_hub.HfApi"),
                patch("zils.runtime.gpu_ready", return_value=True),
                patch(
                    "zils.base_miner.shutil.disk_usage",
                    return_value=SimpleNamespace(free=100 * 1024**3),
                ),
                patch.dict("os.environ", {"HF_TOKEN": "secret"}),
                patch(
                    "zils.base_miner.subprocess.run",
                    side_effect=subprocess.TimeoutExpired("train", 10),
                ) as run,
            ):
                with self.assertRaises(subprocess.TimeoutExpired):
                    base_miner.run(config)
                self.assertNotIn("HF_TOKEN", run.call_args.kwargs["env"])
            self.assertFalse((root / "yield").exists())
            self.assertEqual(
                json.loads((root / "state/attempt.json").read_text())["status"], "training"
            )

    def test_mainnet_is_refused(self):
        with self.assertRaisesRegex(ValueError, "testnet"):
            base_miner.validate({"netuid": 1})

    def test_uncertain_attempt_blocks_new_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            (state / "attempt.json").write_text(json.dumps({"status": "uploading"}))
            with self.assertRaisesRegex(RuntimeError, "review"):
                base_miner.next_attempt(state, now=100000, interval=86400)

    def test_cadence_survives_service_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            (state / "attempt.json").write_text(
                json.dumps({"status": "committed", "finished_at": 100})
            )
            self.assertIsNone(base_miner.next_attempt(state, now=200, interval=86400))
            self.assertEqual(base_miner.next_attempt(state, now=90000, interval=86400), "90000")

    def test_chain_response_loss_is_reconciled_without_second_write(self):
        expected = "zils1:expected"
        before = {"commitment": {"data": "old", "block": 1}}
        after = {"commitment": {"data": expected, "block": 2}}
        with (
            patch("zils.base_miner.chain_state", side_effect=[before, after]),
            patch("zils.base_miner.write_chain", side_effect=TimeoutError) as write,
        ):
            self.assertEqual(base_miner.commit_once({}, None, "hotkey", expected)["block"], 2)
        write.assert_called_once()

    def test_uncertain_chain_does_not_retry(self):
        with (
            patch("zils.base_miner.chain_state", return_value={"commitment": None}),
            patch("zils.base_miner.write_chain", side_effect=TimeoutError) as write,
        ):
            with self.assertRaisesRegex(RuntimeError, "review"):
                base_miner.commit_once({}, None, "hotkey", "expected")
        write.assert_called_once()

    def test_existing_commitment_is_not_written_again(self):
        with (
            patch(
                "zils.base_miner.chain_state",
                return_value={"commitment": {"data": "same", "block": 1}},
            ),
            patch("zils.base_miner.write_chain") as write,
        ):
            self.assertEqual(base_miner.commit_once({}, None, "hotkey", "same")["block"], 1)
        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
