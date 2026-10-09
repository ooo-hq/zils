"""Signed presence and lease supervision without hardware or production jobs."""

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bittensor_wallet import Keypair

from tests.test_queue import Store
from zils import coordinator, miner_presence, models, queue_protocol
from zils.cloud import APIError


class PresenceTests(unittest.TestCase):
    def test_signed_presence_replay_spoofing_and_unknown_fields(self):
        class PresenceStore(Store):
            def rpc(self, name, args):
                if name == "zils_worker_heartbeat":
                    self.observed = args
                    return "2026-10-09T00:00:00Z"
                return super().rpc(name, args)

        store = PresenceStore()
        key = Keypair.create_from_seed("0x" + "19" * 32)
        store.tables["fez_training_workers"].append({"hotkey": key.ss58_address, "enabled": True})
        audience, path = "https://queue.example", "/v1/workers/heartbeat"
        service = coordinator.Service(store, audience)
        with patch("zils.miner_presence.gpu_ready", return_value=True):
            body = miner_presence.presence_payload(
                {models.JEVK5: "reference"}, "cuda", minimum_mib=1024
            )
        message = queue_protocol.sign(key, audience, path, body)
        self.assertIn("received_at", service.worker(path, message))
        self.assertEqual(store.observed["p_hotkey"], key.ss58_address)
        with self.assertRaises(APIError):
            service.worker(path, message)
        message = queue_protocol.sign(key, audience, path, body)
        message["payload"]["hotkey"] = Keypair.create_from_seed("0x" + "29" * 32).ss58_address
        with self.assertRaises(APIError):
            service.worker(path, message)
        for extra in (
            {"quality_band": 100},
            {"received_at": "2099-01-01T00:00:00Z"},
            {"hotkey": "other"},
        ):
            with self.assertRaises(APIError):
                service.worker(path, queue_protocol.sign(key, audience, path, {**body, **extra}))
        for invalid in ({"profiles": []}, {"profiles": [{**body["profiles"][0], "ready": 1}]}):
            with self.assertRaises(APIError):
                service.worker(path, queue_protocol.sign(key, audience, path, invalid))

    def test_evaluation_evidence_binds_attempt_and_separates_acceptance(self):
        from zils.miner_grading import evaluation_observations

        job = {"id": "job", "job_sha256": "a" * 64}
        attempts = [{"hotkey": "worker", "lease_token": "lease", "sha256": "b" * 64}]
        assignments = [
            {
                "hotkey": "worker",
                "uid": 1,
                "lease_token": "lease",
                "sha256": "b" * 64,
                "state": "submitted",
            }
        ]
        report = {
            "job_sha256": "a" * 64,
            "submitted": {1: "b" * 64},
            "baseline": {"status": "evaluated"},
            "miners": [{"uid": 1, "status": "evaluated", "sha256": "b" * 64}],
            "delivery": {"status": "no_qualifying_model"},
        }
        evidence = evaluation_observations(job, attempts, assignments, report)
        self.assertEqual(evidence[0]["outcome"], "valid")
        report["miners"][0]["status"] = "rejected"
        self.assertEqual(
            evaluation_observations(job, attempts, assignments, report)[0]["outcome"],
            "pending_review",
        )
        report["job_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            evaluation_observations(job, attempts, assignments, report)

    def test_presence_opt_in_busy_and_stoppable(self):
        client = Mock()
        refs = {models.JEVK5: "reference"}
        with miner_presence.presence_loop(client, refs, "cuda"):
            pass
        client.call.assert_not_called()
        arrived = threading.Event()
        client.call.side_effect = lambda *args: arrived.set()
        with patch("zils.miner_presence.gpu_ready", return_value=True):
            with miner_presence.presence_loop(
                client, refs, "cuda", enabled=True, minimum_mib=1024, interval=0.02
            ) as busy:
                self.assertTrue(arrived.wait(2))
                busy.set()
                arrived.clear()
                self.assertTrue(arrived.wait(2))
                self.assertFalse(client.call.call_args.args[1]["profiles"][0]["ready"])
            calls = client.call.call_count
            time.sleep(0.05)
            self.assertEqual(client.call.call_count, calls)

    def test_text_training_forwards_lease_and_stops_child(self):
        import zils
        from miner.worker import train_candidate
        from zils import benchmark

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            reference.mkdir()
            models.write_metadata(reference, kind="base")
            data = root / "miner-training.jsonl"
            data.write_text("{}\n")
            config = {
                "base_revision": models.spec(models.JEVK5)["base_revision"],
                "initial_sha256": zils.checkpoint_hash(reference),
                "training_sha256": benchmark.file_hash(data),
                "hotkey": "worker",
                "training_authority": "queue",
                "uid": 1,
            }
            job = {**config, "round_id": "a" * 32}
            calls = []

            def revoked():
                calls.append(True)
                raise APIError(409, "revoked")

            def fake_child(command, log, device, **kwargs):
                from zils.runtime import _run_child

                _run_child(
                    [sys.executable, "-c", "import time; time.sleep(20)"],
                    log,
                    {},
                    25,
                    kwargs.get("check_lease"),
                )

            started = time.monotonic()
            with (
                patch("miner.worker.run_child", side_effect=fake_child),
                self.assertRaises(APIError),
            ):
                train_candidate(config, root, job, sys.executable, "cuda", check_lease=revoked)
            self.assertTrue(calls)
            self.assertLess(time.monotonic() - started, 4)
            self.assertFalse(list(root.glob("state/jobs/*/candidate.json")))

    def test_mps_memory_guard_uses_native_probe_and_fails_closed(self):
        from zils.runtime import gpu_ready

        with patch(
            "zils.runtime.subprocess.run",
            return_value=Mock(
                stdout="Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100000.\nPages inactive: 100000.\nPages speculative: 0.\n"
            ),
        ) as probe:
            self.assertTrue(gpu_ready("mps", minimum_mib=1024))
            self.assertFalse(gpu_ready("mps", minimum_mib=4096))
            self.assertEqual(probe.call_args.args[0], ["/usr/bin/vm_stat"])
        with patch("zils.runtime.subprocess.run", side_effect=OSError):
            self.assertFalse(gpu_ready("mps", minimum_mib=1024))
