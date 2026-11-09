"""A queue miner keeps a customer's training rows only while it works on the job (customer terms: deleted after the
round). Whatever the ending (submitted, lease lost or cancelled, GPU deferred, crash), the rows and photos go; the
trained checkpoint stays for the validator."""

import hashlib
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import zils
from tests.test_fleet_profiles import checkpoint
from zils import models
from zils.runtime import CapacityUnavailable

TRAINING = b'{"state": {"text": "a customer row"}, "questions": {}}\n'


class Setup:
    def __init__(self, root):
        self.root, self.calls, self.downloads = root, [], 0
        self.reference = checkpoint(root / "reference", models.D3)
        self.job_id = str(uuid.uuid4())
        self.assignment = {
            "job_id": self.job_id,
            "lease_token": str(uuid.uuid4()),
            "uid": 1,
            "model": models.spec(models.D3),
            "base_revision": models.spec(models.D3)["base_revision"],
            "initial_sha256": zils.checkpoint_hash(self.reference),
            "training_sha256": hashlib.sha256(TRAINING).hexdigest(),
            "job_sha256": "c" * 64,
            "training": {"url": "https://example.test/training"},
            "min_free_mib": 14000,
            "max_seconds": 900,
        }
        self.submit_error = None
        self.claim = lambda: self.assignment

    @property
    def job(self):
        return self.root / "state" / self.job_id

    def client(self):
        setup = self

        class Client:
            url = "https://example.test"
            key = SimpleNamespace(ss58_address="hotkey")

            def call(self, name, payload=None):
                setup.calls.append(name)
                if name == "claim":
                    return {"assignment": setup.claim()}
                if name == "uploads":
                    names = models.artifact_files(setup.job / "candidate")
                    return {"uploads": {n: {"uploaded": True} for n in names}}
                if name == "submit" and setup.submit_error:
                    error, setup.submit_error = setup.submit_error, None
                    raise error
                return {}

        return Client()

    def download(self, url, path, limit):
        self.downloads += 1
        path.write_bytes(TRAINING)

    def trains(self, outcome=None):
        """A trainer that sees the rows (and photos folder), then returns a checkpoint or raises `outcome`."""

        def train(config, directory, job, runtime, device, **options):
            assert (directory / "miner-training.jsonl").read_bytes() == TRAINING
            (directory / "images").mkdir(exist_ok=True)
            (directory / "images" / "photo.png").write_bytes(b"png")
            if outcome is not None:
                raise outcome
            candidate = directory / "candidate"
            if not candidate.exists():
                checkpoint(candidate, models.D3)
            return {"checkpoint": str(candidate), "sha256": "d" * 64}

        return train

    def run(self, train):
        from miner import queue

        with (
            patch("miner.queue.gpu_ready", return_value=True),
            patch("miner.queue.download", self.download),
            patch("miner.queue.train_candidate", train),
            patch("miner.queue.lease_heartbeat") as heartbeat,
        ):
            heartbeat.return_value.__enter__.return_value = lambda: None
            return queue.run_once(
                self.client(), self.root / "state", self.reference, sys.executable, "cuda"
            )

    def customer_files(self):
        return sorted(
            p.name
            for p in self.job.rglob("*")
            if p.name == "miner-training.jsonl" or "images" in p.parts
        )


class DeletionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.setup = Setup(Path(temp.name))

    def test_a_submitted_job_keeps_its_checkpoint_but_not_the_customer_rows(self):
        self.assertTrue(self.setup.run(self.setup.trains()))
        self.assertIn("submit", self.setup.calls)
        self.assertEqual(self.setup.customer_files(), [])
        self.assertTrue(
            (self.setup.job / "candidate" / "model.json").exists()
        )  # still there for the validator
        self.assertTrue((self.setup.job / "assignment.json").exists())

    def test_a_lost_or_cancelled_lease_deletes_the_rows(self):
        with self.assertRaises(RuntimeError):
            self.setup.run(self.setup.trains(RuntimeError("lease lost: the job was cancelled")))
        self.assertIn("fail", self.setup.calls)
        self.assertEqual(self.setup.customer_files(), [])

    def test_a_job_handed_back_for_lack_of_gpu_deletes_the_rows(self):
        self.assertFalse(self.setup.run(self.setup.trains(CapacityUnavailable())))
        self.assertIn("defer", self.setup.calls)
        self.assertEqual(self.setup.customer_files(), [])

    def test_rows_left_by_a_crash_are_deleted_before_the_next_claim(self):
        self.setup.job.mkdir(parents=True)
        (self.setup.job / "miner-training.jsonl").write_bytes(TRAINING)
        (self.setup.job / "images").mkdir()
        (self.setup.job / "images" / "photo.png").write_bytes(b"png")
        (self.setup.job / "assignment.json").write_text("{}")
        self.setup.claim = lambda: None  # no new job: the sweep still runs
        self.assertFalse(self.setup.run(self.setup.trains()))
        self.assertEqual(self.setup.customer_files(), [])
        self.assertTrue((self.setup.job / "assignment.json").exists())

    def test_a_retry_after_a_lost_submit_response_downloads_the_rows_again(self):
        self.setup.submit_error = ConnectionError("response lost")
        with self.assertRaises(ConnectionError):
            self.setup.run(self.setup.trains())
        self.assertEqual(self.setup.customer_files(), [])
        self.assertTrue(self.setup.run(self.setup.trains()))  # same lease, same hash check
        self.assertEqual(self.setup.downloads, 2)
        self.assertEqual(self.setup.calls.count("submit"), 2)
        self.assertEqual(self.setup.customer_files(), [])


class FleetBundleTests(unittest.TestCase):
    def test_a_bundle_whose_rows_were_deleted_says_how_to_get_new_ones(self):
        from miner.worker import train_candidate

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            reference = checkpoint(bundle / "reference", models.D3)
            config = {
                "base_revision": models.spec(models.D3)["base_revision"],
                "initial_sha256": zils.checkpoint_hash(reference),
                "training_sha256": hashlib.sha256(TRAINING).hexdigest(),
                "job_id": "job",
                "job_sha256": "c" * 64,
                "hotkey": "hotkey",
                "validator_hotkey": "validator",
                "uid": 1,
            }
            job = {
                **{
                    k: config[k]
                    for k in (
                        "base_revision",
                        "initial_sha256",
                        "training_sha256",
                        "job_id",
                        "job_sha256",
                    )
                },
                "round_id": "a" * 32,
            }
            with self.assertRaisesRegex(
                ValueError, "deleted after its last round.*zils.fleet init"
            ):
                train_candidate(config, bundle, job, sys.executable, "cuda")


if __name__ == "__main__":
    unittest.main()
