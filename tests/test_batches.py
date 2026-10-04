"""Bulk input validation and public tenant boundaries."""

import importlib
import importlib.util
import json
import unittest
import uuid

from fez.cloud import APIError
from fez.decisions import DecisionError
from tests.test_api import BODY, FINGERPRINT, RELEASE


class BatchTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("fez.batches"), "durable bulk module missing")
        return importlib.import_module("fez.batches")

    def test_mixed_input_preserves_ids_and_rejects_duplicates_before_execution(self):
        m = self.module()
        catalog = {"zils-shared": {"id": RELEASE, "fingerprint": FINGERPRINT}}
        rows = [
            {"custom_id": "good", "body": BODY},
            {"custom_id": "bad", "body": {**BODY, "questions": {}}},
            {"custom_id": "private", "body": {**BODY, "model": "private"}},
        ]
        data = "\n".join(json.dumps(x) for x in rows).encode()
        result = m.parse_input(data, catalog)
        self.assertEqual([x["custom_id"] for x in result], ["good", "bad", "private"])
        self.assertEqual(result[0]["frozen"], catalog["zils-shared"])
        self.assertIsNone(result[0]["result"])
        self.assertEqual(result[1]["result"]["error"]["status"], 422)
        self.assertEqual(result[2]["result"]["error"]["status"], 404)
        duplicate = data + b"\n" + json.dumps(rows[0]).encode()
        with self.assertRaises(DecisionError) as cm:
            m.parse_input(duplicate, catalog)
        self.assertEqual(cm.exception.code, "duplicate_custom_id")
        for malformed in (b"", b"{}", b"{bad", b'{"custom_id":"x","custom_id":"y","body":{}}'):
            with self.assertRaises(DecisionError):
                m.parse_input(malformed, catalog)
        with self.assertRaises(DecisionError):
            m.parse_input(data, catalog, max_records=2)

    def test_polling_never_queries_without_owner_and_never_returns_leases(self):
        m = self.module()
        owner, batch = str(uuid.uuid4()), str(uuid.uuid4())

        class DB:
            def rows(self, table, query):
                self.query = query
                if "owner_id=eq." + owner not in query:
                    return []
                return [
                    {
                        "id": batch,
                        "owner_id": owner,
                        "status": "running",
                        "lease_token": "secret",
                        "catalog": {"private": "hidden"},
                    }
                ]

        db = DB()
        service = m.Batches(db)
        result = service.get(owner, batch)
        self.assertEqual(result["status"], "running")
        self.assertNotIn("secret", str(result))
        self.assertNotIn("hidden", str(result))
        with self.assertRaises(DecisionError) as cm:
            service.get(str(uuid.uuid4()), batch)
        self.assertEqual(cm.exception.status, 404)

    def test_postgres_incompatible_unicode_is_rejected_before_work(self):
        m = self.module()
        catalog = {"zils-shared": {"id": RELEASE, "fingerprint": FINGERPRINT}}
        cases = [
            {**BODY, "state": "text\x00"},
            {**BODY, "questions": {"bad\x00": {"type": "noul"}}},
            {
                **BODY,
                "questions": {"q": {"type": "choice", "criteria": {"a\x00": None, "b": None}}},
            },
            {
                **BODY,
                "questions": {"q": {"type": "score", "criteria": ["low", {"text": "bad\x00"}]}},
            },
        ]
        rows = [{"custom_id": str(i), "body": case} for i, case in enumerate(cases)]
        parsed = m.parse_input("\n".join(json.dumps(x) for x in rows).encode(), catalog)
        self.assertTrue(all(x["result"]["error"]["status"] == 422 for x in parsed))
        with self.assertRaises(DecisionError):
            m.parse_input(json.dumps({"custom_id": "bad\x00", "body": BODY}).encode(), catalog)

    def test_create_retry_recovers_only_confirmed_completed_uploads(self):
        m = self.module()
        owner, batch = str(uuid.uuid4()), str(uuid.uuid4())

        class StorageConflict:
            def __init__(self, status, uploaded):
                self.status, self.uploaded = status, uploaded

            def rpc(self, name, args):
                return {"id": batch, "status": "uploading", "input_path": "input.jsonl"}

            def signed(self, bucket, path, *, upload=False):
                raise APIError(self.status, "Storage rejected the signing request")

            def exists(self, bucket, path):
                return self.uploaded

        for status, uploaded, recovered in [
            (409, True, True),
            (409, False, False),
            (503, True, False),
        ]:
            with self.subTest(status=status, uploaded=uploaded):
                service = m.Batches(StorageConflict(status, uploaded))
                if recovered:
                    code, result = service.dispatch(
                        "POST", "/v1/batches", owner, {"idempotency_key": "retry"}, None
                    )
                    self.assertEqual(
                        (code, result["id"], result["status"]), (200, batch, "uploading")
                    )
                    self.assertNotIn("upload", result)
                else:
                    with self.assertRaises(APIError) as cm:
                        service.dispatch(
                            "POST", "/v1/batches", owner, {"idempotency_key": "retry"}, None
                        )
                    self.assertEqual(cm.exception.status, status)
