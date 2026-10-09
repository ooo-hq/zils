"""Actual SDK/HTTP transfers backed by the real fenced SQL catalog."""

import tempfile
import unittest
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import boto3
import requests
from botocore.config import Config

from tests.api_database import literal
from tests.spaces_fixture import S3, LostReplyDB, database
from zils.cloud import APIError, upload
from zils.spaces import SpacesStorage
from zils.storage import ObjectCatalog


class SpacesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = database()
        cls.db = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def setUp(self):
        self.s3 = S3()
        self.addCleanup(self.s3.close)
        self.client = boto3.client(
            "s3",
            endpoint_url=self.s3.origin,
            region_name="us-east-1",
            aws_access_key_id="fixture",
            aws_secret_access_key="fixture-secret",
            config=Config(
                signature_version="s3v4",
                retries={"total_max_attempts": 1},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )
        self.addCleanup(self.client.close)
        self.catalog = ObjectCatalog(self.db)
        self.store = SpacesStorage(self.catalog, self.client, "test-storage")
        self.path = f"{uuid.uuid4()}/candidate.bin"
        self.bucket = "fez-training-models"

    def put(self, grant, data=b"abc"):
        response = requests.put(grant["url"], data=data, headers=grant["headers"], timeout=5)
        try:
            return response.status_code
        finally:
            response.close()

    def test_old_grant_cannot_replace_completed_bytes(self):
        grant = self.store.signed(self.bucket, self.path, upload=True)
        self.assertEqual(parse_qs(urlsplit(grant["url"]).query)["partNumber"], ["1"])
        self.assertEqual(grant["headers"], {"Content-Type": "application/octet-stream"})
        self.assertEqual(grant["provider"], "spaces")
        self.assertIn("expires_at", grant)
        self.assertFalse(self.store.exists(self.bucket, self.path))
        self.assertEqual(self.put(grant, b"first"), 200)
        self.assertEqual(self.put(grant, b"final"), 200)
        self.assertTrue(self.store.exists(self.bucket, self.path))
        self.assertEqual(self.put(grant, b"wrong"), 404)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "bytes"
            self.store.download(self.bucket, self.path, dest, 100)
            self.assertEqual(dest.read_bytes(), b"final")
        with self.assertRaises(APIError) as ctx:
            self.store.signed(self.bucket, self.path, upload=True)
        self.assertEqual(ctx.exception.status, 409)

    def test_lost_completion_and_commit_replies_recover_without_replacement(self):
        for action in (None, "commit", "bind"):
            with self.subTest(action=action):
                path = f"{uuid.uuid4()}/candidate.bin"
                db = LostReplyDB(self.db, action)
                store = SpacesStorage(ObjectCatalog(db), self.client, "test-storage")
                try:
                    grant = store.signed(self.bucket, path, upload=True)
                except APIError:
                    grant = store.signed(self.bucket, path, upload=True)
                self.put(grant)
                self.s3.lose_completion = action is None
                try:
                    ready = store.exists(self.bucket, path)
                except APIError:
                    ready = store.exists(self.bucket, path)
                self.assertTrue(ready)
                self.assertEqual(len([x for x in self.s3.objects if x.endswith(path)]), 1)

    def test_ambiguous_creation_never_reuses_physical_key(self):
        self.s3.lose_creation = True
        with self.assertRaises(APIError):
            self.store.signed(self.bucket, self.path, upload=True)
        first = self.catalog.get(self.bucket, self.path)
        self.s3.lose_creation = False
        with self.assertRaises(APIError):
            self.store.signed(self.bucket, self.path, upload=True)
        self.db.sql(
            f"update zils_storage_objects set lease_until=now()-interval '1 second' where generation={literal(first['generation'])}"
        )
        grant = self.store.signed(self.bucket, self.path, upload=True)
        self.assertNotEqual(
            first["physical_key"], self.catalog.get(self.bucket, self.path)["physical_key"]
        )
        self.assertEqual(len(self.s3.uploads), 2)
        self.put(grant)
        self.assertTrue(self.store.exists(self.bucket, self.path))

    def test_forbidden_list_is_unavailable_not_missing(self):
        grant = self.store.signed(self.bucket, self.path, upload=True)
        self.put(grant)
        self.s3.deny_lists = True
        with self.assertRaises(APIError) as ctx:
            self.store.exists(self.bucket, self.path)
        self.assertEqual(ctx.exception.status, 503)

    def test_invalid_parts_cannot_become_ready(self):
        for kind in ("empty", "oversize", "second", "metadata"):
            with self.subTest(kind=kind):
                path = f"{uuid.uuid4()}/bad.bin"
                grant = self.store.signed(self.bucket, path, upload=True)
                self.put(grant, b"" if kind == "empty" else b"abc")
                row = self.catalog.get(self.bucket, path)
                uid = row["upload_id"]
                if kind == "oversize":
                    self.db.sql(
                        f"update zils_storage_objects set max_bytes=2 where generation={literal(row['generation'])}"
                    )
                if kind == "second":
                    self.s3.uploads[uid]["parts"][2] = self.s3.uploads[uid]["parts"][1]
                if kind == "metadata":
                    self.s3.uploads[uid]["generation"] = str(uuid.uuid4())
                with self.assertRaises(APIError):
                    self.store.exists(self.bucket, path)
                self.assertNotEqual(self.catalog.get(self.bucket, path)["state"], "ready")

    def test_server_upload_is_readable_and_delete_hides_it_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.write_bytes(b"trusted server content")
            self.store.upload(self.bucket, self.path, source)
            self.assertTrue(self.store.exists(self.bucket, self.path))
            row = self.catalog.get(self.bucket, self.path)
            self.store.remove(self.bucket, [self.path])
            self.assertFalse(self.store.exists(self.bucket, self.path))
            with self.assertRaises(APIError):
                self.store.signed(self.bucket, self.path)
            self.db.sql(
                f"update zils_storage_objects set cleanup_after=now()-interval '1 second',lease_until=now()-interval '1 second',grant_expires_at=now()-interval '1 second' where generation={literal(row['generation'])}"
            )
            self.store.reap()
            self.assertNotIn(row["physical_key"], self.s3.objects)
            self.assertEqual(self.catalog.get(self.bucket, self.path)["state"], "deleted")

    def test_existing_miner_upload_helper_sends_the_part_bytes(self):
        grant = self.store.signed(self.bucket, self.path, upload=True)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "candidate"
            source.write_bytes(b"miner checkpoint")
            upload(grant["url"], source, grant["headers"], grant["method"])
        self.assertTrue(self.store.exists(self.bucket, self.path))

    def test_cleanup_reaps_old_unbound_upload_but_preserves_current_generation(self):
        self.s3.lose_creation = True
        with self.assertRaises(APIError):
            self.store.signed(self.bucket, self.path, upload=True)
        first = self.catalog.get(self.bucket, self.path)
        old_uid = next(iter(self.s3.uploads))
        self.s3.uploads[old_uid]["created"] = "2000-01-01T00:00:00+00:00"
        self.s3.lose_creation = False
        self.db.sql(
            f"update zils_storage_objects set lease_until=now()-interval '1 second' where generation={literal(first['generation'])}"
        )
        grant = self.store.signed(self.bucket, self.path, upload=True)
        current_uid = self.catalog.get(self.bucket, self.path)["upload_id"]
        self.s3.uploads[current_uid]["created"] = "2000-01-01T00:00:00+00:00"
        self.db.sql(
            f"update zils_storage_retired set safe_after=now()-interval '1 second' where generation={literal(first['generation'])}"
        )
        self.store.reap()
        self.assertNotIn(old_uid, self.s3.uploads)
        self.assertIn(current_uid, self.s3.uploads)
        self.put(grant)
        self.assertTrue(self.store.exists(self.bucket, self.path))
