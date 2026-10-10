"""One paid comparison, identical held-out data, and no changes to release provenance."""

import copy
import json
import shutil
import tempfile
import unittest
import uuid
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from tests.test_adapter_releases import fixture
from zils import jev_comparison as jev
from zils.coordinator import public_job


class Store:
    def __init__(self, job, source):
        self.job, self.source = job, source
        self.claims = 0

    def rows(self, table, query):
        return [copy.deepcopy(self.job)]

    def patch(self, table, query, values):
        previous = self.job.get("jev_comparison")
        if "jev_comparison=is.null" in query and previous:
            return []
        if (
            "jev_comparison->>run_id=eq." in query
            and f"run_id=eq.{previous['run_id']}" not in query
        ):
            return []
        self.claims += 1
        self.job.update(copy.deepcopy(values))
        return [copy.deepcopy(self.job)]

    def download(self, bucket, path, destination, limit):
        shutil.copyfile(self.source / "data" / Path(path).name, destination)


class ComparisonTest(unittest.TestCase):
    def setup_job(self, root):
        job = fixture(root / "source")
        job["result"]["workflow"] = {"state": "ready", "model_id": "private-model"}
        job["jev_comparison"] = {
            "status": "pending",
            "model": jev.MODEL,
            "run_id": str(uuid.uuid4()),
            "authorized_at": jev.now().isoformat(),
        }
        store = Store(job, root / "source")
        return job, store, jev.Comparisons(store, root / "cache", "test-key")

    def prediction(self, case, key):
        return {
            "id": case["id"],
            "probabilities": {"true": 0.9, "false": 0.1},
            "elapsed_ms": 10,
            "input_tokens": 200,
        }

    def test_completed_comparison_is_cached_and_preserves_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, store, worker = self.setup_job(Path(tmp))
            original = copy.deepcopy(job)
            original.pop("jev_comparison")
            with patch.object(jev, "predict", side_effect=self.prediction) as predict:
                self.assertTrue(worker.tick())
                self.assertFalse(worker.tick())
                self.assertEqual(predict.call_count, 2)
            result = job.pop("jev_comparison")
            self.assertEqual(job, original)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["accuracy"], 1)
            self.assertEqual(result["count"], 2)
            self.assertEqual(result["input_tokens"], 400)
            self.assertEqual(result["checkpoint_sha256"], job["result"]["delivery"]["sha256"])
            job["jev_comparison"] = result
            self.assertEqual(public_job(job)["jev_comparison"], result)

    def test_changed_evidence_and_images_never_reach_jev(self):
        for mode in ("hash", "count", "candidate", "image", "unapproved"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                job, _, worker = self.setup_job(root)
                if mode == "hash":
                    (root / "source/data/test.jsonl").write_text("{}\n")
                elif mode == "count":
                    job["result"]["miners"][0]["cases"] = 99
                elif mode == "candidate":
                    job["result"]["delivery"]["uid"] = 99
                elif mode == "image":
                    job["model_profile"] = jev.models.spec(jev.models.IMAJEV)
                    job["manifest"]["model"] = job["model_profile"]
                else:
                    job["result"]["delivery"]["status"] = "no_qualifying_model"
                with patch.object(jev, "predict") as predict:
                    worker.tick()
                    predict.assert_not_called()
                if mode not in ("image", "unapproved"):
                    self.assertEqual(job["jev_comparison"]["status"], "failed")

    def test_failed_and_interrupted_calls_are_not_recharged(self):
        for error in (requests.Timeout("secret request text"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                job, _, worker = self.setup_job(root)
                with patch.object(
                    jev, "predict", side_effect=[self.prediction({"id": "test-0"}, ""), error]
                ) as predict:
                    try:
                        worker.tick()
                    except KeyboardInterrupt:
                        job["jev_comparison"]["lease_until"] = (
                            jev.now() - timedelta(minutes=1)
                        ).isoformat()
                    worker.tick()
                    self.assertEqual(predict.call_count, 2)
                self.assertEqual(job["jev_comparison"]["status"], "failed")
                self.assertNotIn("accuracy", job["jev_comparison"])
                self.assertNotIn("secret", json.dumps(public_job(job)))

    def test_live_lease_or_lost_claim_cannot_duplicate_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, store, worker = self.setup_job(Path(tmp))
            with (
                patch.object(store, "patch", return_value=[]),
                patch.object(jev, "predict") as predict,
            ):
                self.assertFalse(worker.tick())
                predict.assert_not_called()
            job["jev_comparison"] = {
                "status": "running",
                "lease_until": (jev.now() + timedelta(minutes=1)).isoformat(),
            }
            with patch.object(jev, "predict") as predict:
                self.assertFalse(worker.tick())
                predict.assert_not_called()

    def test_old_submissions_without_permission_are_not_enrolled(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, _, worker = self.setup_job(Path(tmp))
            job.pop("jev_comparison")
            with patch.object(jev, "predict") as predict:
                self.assertFalse(worker.tick())
                predict.assert_not_called()

    def test_new_submissions_record_explicit_comparison_permission(self):
        from tests.test_jobs import POLICY
        from tests.test_queue import Store as QueueStore
        from zils.coordinator import Service

        for consent in (None, False, True):
            store = QueueStore()
            store.url = "http://127.0.0.1:8910"
            service = Service(store, "http://127.0.0.1:8910", jev.models.JEVK5)
            payload = {"name": "support", "acceptance": POLICY, "allow_training_data_export": True}
            if consent is not None:
                payload["allow_jev_comparison"] = consent
            result = service.customer("POST", "/v1/jobs", "owner-token", payload)
            comparison = result["job"]["jev_comparison"]
            if consent is True:
                self.assertEqual(comparison["status"], "pending")
                self.assertIn("authorized_at", comparison)
            else:
                self.assertIsNone(comparison)

    def test_provider_contract_and_no_answer_leakage(self):
        for kind, question, answer, label in (
            ("noul", {"instructions": "Is it urgent?"}, {"noul": 0.8}, "true"),
            (
                "choice",
                {"instructions": "Choose a team", "criteria": {"a": None, "b": None}},
                {"choice": "a", "probabilities": {"a": 0.8, "b": 0.2}},
                "a",
            ),
            (
                "score",
                {"instructions": "Rate urgency", "criteria": ["low", "high"]},
                {"probabilities": {"0": 0.2, "1": 0.8}},
                "1",
            ),
        ):
            case = {
                "id": "test",
                "family": "routing",
                "group_id": "group",
                "state": "Help",
                "question": {"type": kind, **question},
                "label": label,
            }
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.status_code = 200
            response.json.return_value = {
                "model": jev.MODEL,
                "answers": {"decision": {"type": kind, **answer}},
                "usage": {"input_tokens": 42},
            }
            with patch.object(jev.requests, "post", return_value=response) as post:
                row = jev.predict(case, "test-secret")
                self.assertEqual(row["input_tokens"], 42)
                payload = post.call_args.kwargs["json"]
                self.assertEqual(
                    payload,
                    {
                        "model": jev.MODEL,
                        "state": case["state"],
                        "questions": {"decision": case["question"]},
                    },
                )
                response.json.return_value["model"] = "different-model"
                with self.assertRaises(ValueError):
                    jev.predict(case, "test-secret")


if __name__ == "__main__":
    unittest.main()
