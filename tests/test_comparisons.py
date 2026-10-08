import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.test_queue import OTHER, OWNER, Store
from zils import comparisons as c, coordinator, models
from zils.cloud import APIError
from zils.comparison_metrics import summarize

ID = "33333333-3333-4333-8333-333333333333"
SHA = "a" * 64
CONSENT = {"allow_typesafe_export": True, "unseen_examples": True, "consent_version": c.CONSENT}


def case(index):
    return {
        "id": str(index),
        "group_id": str(index),
        "family": "routing",
        "state": str(index),
        "question": {"type": "choice", "instructions": "Pick", "criteria": {"a": "A", "b": "B"}},
        "label": "a",
    }


def prediction(row, correct):
    return {
        "id": row["id"],
        "probabilities": {"a": 0.9 if correct else 0.1, "b": 0.1 if correct else 0.9},
        "elapsed_ms": 1,
    }


class ComparisonStore(Store):
    def __init__(self):
        super().__init__()
        self.tables[c.TABLE] = []

    def matches(self, row, query):
        if "lease_until=lt." in query:
            stamp = query.split("lease_until=lt.")[1].split("&")[0]
            if not row.get("lease_until") or datetime.fromisoformat(
                row["lease_until"]
            ) >= datetime.fromisoformat(stamp):
                return False
        return super().matches(row, query)

    def request(self, method, path, body=None, headers=None):
        if method != "POST" or path != f"/rest/v1/{c.TABLE}?on_conflict=job_id":
            raise AssertionError("unexpected request")
        if not any(row["job_id"] == body["job_id"] for row in self.tables[c.TABLE]):
            self.tables[c.TABLE].append(
                {**copy.deepcopy(body), "status": "uploading", "created_at": "2026-10-07T00:00:00Z"}
            )

    def signed(self, bucket, path, *, upload=False):
        return {
            "url": "https://example.invalid/storage/v1/object/" + path,
            "method": "PUT",
            "headers": {"Content-Type": "application/octet-stream", "x-upsert": "false"},
        }

    def download(self, bucket, path, destination, limit):
        data = self.objects[bucket, path]
        if len(data) > limit:
            raise ValueError("large file")
        Path(destination).write_bytes(data)
        return len(data)


class Comparisons(unittest.TestCase):
    def setUp(self):
        self.store = ComparisonStore()
        self.job = {
            "id": ID,
            "owner_id": OWNER,
            "status": "completed",
            "release_prefix": "release",
            "job_sha256": SHA,
            "manifest": {"model": models.spec(models.JEVK5)},
            "result": {"delivery": {"status": "accepted", "sha256": SHA}},
        }
        self.store.tables[coordinator.JOBS] = [self.job]
        self.service = coordinator.Service(
            self.store, "https://training.example.invalid", models.JEVK5
        )
        self.enable = patch.dict(
            "os.environ", {"ZILS_JEV_COMPARISON_ENABLED": "1", "TYPESAFE_API_KEY": "test-key"}
        )
        self.enable.start()
        self.addCleanup(self.enable.stop)

    def call(self, method="POST", action="", body=None, token="owner-token"):
        return self.service.customer(
            method, f"/v1/jobs/{ID}/comparison" + action, token, body or {}
        )

    def test_ownership_checked_before_comparison_reads(self):
        for method in ("GET", "POST"):
            with self.assertRaises(APIError) as caught:
                self.call(method, body=CONSENT, token="other-token")
            self.assertEqual(caught.exception.status, 404)
        self.assertNotEqual(OWNER, OTHER)

    def test_requires_all_consent_fields_and_accepted_model(self):
        for key in CONSENT:
            with self.assertRaises(APIError):
                self.call(body={k: v for k, v in CONSENT.items() if k != key})
        self.assertEqual(self.store.tables[c.TABLE], [])
        self.job["result"]["delivery"]["status"] = "no_qualifying_model"
        with self.assertRaises(APIError):
            self.call(body=CONSENT)

    def test_duplicate_create_and_submit_preserve_frozen_run(self):
        created = self.call(body=CONSENT)
        self.assertEqual(created["upload"]["headers"]["x-upsert"], "false")
        self.store.objects[c.DATA_BUCKET, c.input_path(ID)] = b"saved"
        self.call(body=CONSENT)
        self.assertEqual(len(self.store.tables[c.TABLE]), 1)
        self.assertEqual(self.call(action="/submit")["comparison"]["status"], "queued")
        row = self.store.tables[c.TABLE][0]
        row.update(status="completed", result={"preserved": True})
        self.assertEqual(self.call(action="/submit")["comparison"]["result"], {"preserved": True})
        self.assertNotIn("upload", self.call(body=CONSENT))
        self.assertNotIn("owner_id", self.call("GET")["comparison"])
        self.assertNotIn("lease_token", self.call("GET")["comparison"])

    def test_missing_upload_and_disabled_service_cannot_queue(self):
        self.call(body=CONSENT)
        with self.assertRaises(APIError):
            self.call(action="/submit")
        with patch.dict("os.environ", {"ZILS_JEV_COMPARISON_ENABLED": "0"}):
            self.assertFalse(self.call("GET")["available"])
            with self.assertRaises(APIError):
                self.call(body=CONSENT)

    def test_overlap_ids_groups_prompts_and_duplicate_prompts_rejected(self):
        original = case(0)
        for key in ("id", "group_id", "state"):
            fresh = {**case(1), key: original[key]}
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_fresh([fresh], [original])
        with self.assertRaises(ValueError):
            c.validate_fresh([case(1), {**case(2), "state": "1"}], [])
        c.validate_fresh([case(1)], [original])

    def test_overlap_stops_worker_before_external_request(self):
        row = {
            **self.call(body=CONSENT)["comparison"],
            "job_id": ID,
            "owner_id": OWNER,
            "consent_version": c.CONSENT,
        }
        payload = (json.dumps(case(1)) + "\n").encode()
        self.store.objects[c.DATA_BUCKET, c.input_path(ID)] = payload
        self.job["manifest"]["files"] = {"train.jsonl": hashlib.sha256(payload).hexdigest()}
        self.store.objects[c.DATA_BUCKET, coordinator.prepared_path(self.job, "train.jsonl")] = (
            payload
        )
        processor = SimpleNamespace(service=self.service, store=self.store)
        with tempfile.TemporaryDirectory() as root, patch.object(c, "jev_prediction") as external:
            with self.assertRaises(ValueError):
                c.evaluate(processor, row, Path(root))
            external.assert_not_called()

    def test_processor_claims_once_and_never_retries_paid_failures(self):
        self.call(body=CONSENT)
        row = self.store.tables[c.TABLE][0]
        row["status"] = "queued"
        with tempfile.TemporaryDirectory() as root:
            processor = SimpleNamespace(
                root=Path(root), store=self.store, args=SimpleNamespace(device="cpu")
            )
            with patch.object(c, "evaluate", side_effect=ValueError("private prompt")) as evaluate:
                self.assertTrue(c.process_one(processor))
                self.assertFalse(c.process_one(processor))
                evaluate.assert_called_once()
            self.assertEqual(row["status"], "failed")
            self.assertNotIn("private prompt", row["error"])
            self.assertIsNone(row.get("result"))
            row.update(
                status="running",
                lease_until=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            )
            with patch.object(c, "evaluate") as evaluate:
                self.assertFalse(c.process_one(processor))
                evaluate.assert_not_called()
            self.assertEqual(row["status"], "failed")

    def test_complete_result_is_saved_without_changing_training_acceptance(self):
        self.call(body=CONSENT)
        row = self.store.tables[c.TABLE][0]
        row["status"] = "queued"
        before = copy.deepcopy(self.job)
        with (
            tempfile.TemporaryDirectory() as root,
            patch.object(
                c,
                "evaluate",
                return_value={"result": {"verdict": "less_accurate"}, "input_sha256": "b" * 64},
            ),
        ):
            processor = SimpleNamespace(
                root=Path(root), store=self.store, args=SimpleNamespace(device="cpu")
            )
            self.assertTrue(c.process_one(processor))
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["result"]["verdict"], "less_accurate")
        self.assertEqual(self.job, before)

    def test_provider_pins_version_and_sends_only_inputs(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code, response.content = 200, b"{}"
        response.json.return_value = {
            "model": c.JEV,
            "answers": {
                "decision": {
                    "type": "choice",
                    "choice": "b",
                    "probabilities": {"a": 0.49, "b": 0.5},
                }
            },
        }
        with patch.object(c.requests, "post", return_value=response) as post:
            result = c.jev_prediction(case(1), "test-key")
            payload = post.call_args.kwargs["json"]
            self.assertEqual(set(payload), {"model", "state", "questions"})
            self.assertEqual(payload["questions"]["decision"], case(1)["question"])
            self.assertFalse(post.call_args.kwargs["allow_redirects"])
            self.assertEqual(result["choice"], "b")
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1)
            response.json.return_value["model"] = "jev-latest"
            with self.assertRaises(ValueError):
                c.jev_prediction(case(1), "test-key")

    def test_fresh_test_uses_published_frozen_weights_and_saves_complete_evidence(self):
        from tests.test_adapter_releases import fixture
        from zils.cloud import MODEL_BUCKET

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            job = fixture(source)
            self.store.tables[coordinator.JOBS] = [job]
            for file in (source / "data").iterdir():
                self.store.objects[c.DATA_BUCKET, coordinator.prepared_path(job, file.name)] = (
                    file.read_bytes()
                )
            for name in (*models.JEVK5_FILES, "release.json"):
                self.store.objects[MODEL_BUCKET, job["release_prefix"] + "/" + name] = (
                    source / name
                ).read_bytes()
            cases = [case(index) for index in range(40)]
            data = "".join(json.dumps(row) + "\n" for row in cases).encode()
            self.store.objects[c.DATA_BUCKET, c.input_path(job["id"])] = data
            c.customer(self.service, job, "POST", None, CONSENT)
            c.customer(self.service, job, "POST", "submit", {})
            processor = SimpleNamespace(
                root=root / "work",
                store=self.store,
                service=self.service,
                args=SimpleNamespace(device="cpu", runtime_python="fixture"),
            )

            def inference(command, log, device, timeout):
                submission = json.loads(
                    Path(command[command.index("--submissions") + 1]).read_text()
                )[0]
                self.assertEqual(submission["sha256"], job["result"]["delivery"]["sha256"])
                self.assertEqual(models.temperature(submission["checkpoint"]), 0.75)
                Path(command[command.index("--report") + 1]).write_text(
                    json.dumps(
                        {
                            "miners": [
                                {
                                    "status": "evaluated",
                                    "sha256": submission["sha256"],
                                    "predictions": [prediction(row, True) for row in cases],
                                }
                            ]
                        }
                    )
                )

            with (
                patch("zils.jevk5.validate_inputs"),
                patch.object(c, "run_child", side_effect=inference),
                patch.object(
                    c, "jev_prediction", side_effect=lambda row, key: prediction(row, False)
                ) as external,
            ):
                self.assertTrue(c.process_one(processor))
                self.assertEqual(external.call_count, 40)
            row = self.store.tables[c.TABLE][0]
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["input_sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(row["result"]["trained"]["correct"], 40)
            self.assertEqual(row["result"]["jev"]["correct"], 0)
            self.assertEqual(len(row["result"]["rows"]), 40)


class Evidence(unittest.TestCase):
    def test_wins_losses_and_small_sample(self):
        cases = [case(i) for i in range(40)]
        trained = [prediction(row, True) for row in cases]
        jev = [prediction(row, False) for row in cases]
        self.assertEqual(summarize(cases, trained, jev)["verdict"], "more_accurate")
        self.assertEqual(summarize(cases, jev, trained)["verdict"], "less_accurate")
        self.assertEqual(summarize(cases, trained, trained)["verdict"], "no_clear_difference")
        self.assertEqual(
            summarize(cases[:16], trained[:16], jev[:16])["verdict"], "no_clear_difference"
        )
        for row in cases:
            row["group_id"] = "one-conversation"
        self.assertEqual(summarize(cases, trained, jev)["verdict"], "no_clear_difference")

    def test_partial_invalid_and_returned_choice(self):
        cases = [case(1)]
        good = [prediction(cases[0], True)]
        with self.assertRaises(ValueError):
            summarize(cases, good, [])
        # Rounded probability ties do not replace a provider's actual returned choice.
        tied = [{"id": "1", "probabilities": {"a": 0.5, "b": 0.5}, "elapsed_ms": 1, "choice": "b"}]
        result = summarize(cases, good, tied)
        self.assertEqual(result["jev"]["correct"], 0)
        self.assertEqual(result["wins"], 1)
        tied[0]["choice"] = "invented"
        with self.assertRaises(ValueError):
            summarize(cases, good, tied)


if __name__ == "__main__":
    unittest.main()
