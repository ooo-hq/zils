"""Provider changes must preserve legacy files, private reads and deleted identities."""

import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import boto3
from botocore.config import Config

from tests.spaces_fixture import S3, database
from zils.cloud import APIError, Supabase
from zils.spaces import SpacesStorage
from zils.storage import ObjectCatalog, StorageRouter


class LegacyFiles:
    def __init__(self):
        self.files = {}

    def stat(self, bucket, path):
        data = self.files.get((bucket, path))
        return None if data is None else {"size": len(data)}

    def exists(self, bucket, path):
        return (bucket, path) in self.files

    def signed(self, bucket, path, *, upload=False):
        if upload and self.exists(bucket, path):
            raise APIError(409, "Immutable file exists")
        return {
            "url": f"https://legacy.example/{bucket}/{path}",
            "provider": "supabase",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
            "method": "PUT",
            "headers": {"Content-Type": "application/octet-stream", "x-upsert": "false"},
        }

    def upload(self, bucket, path, source):
        if self.exists(bucket, path):
            raise APIError(409, "Immutable file exists")
        self.files[bucket, path] = Path(source).read_bytes()

    def download(self, bucket, path, destination, limit, **kwargs):
        data = self.files[bucket, path]
        if len(data) > limit:
            raise ValueError("over limit")
        Path(destination).write_bytes(data)
        return len(data)

    def remove(self, bucket, paths):
        for path in paths:
            self.files.pop((bucket, path), None)


class StorageRouterTest(unittest.TestCase):
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
        self.legacy = LegacyFiles()
        self.catalog = ObjectCatalog(self.db)
        self.spaces = SpacesStorage(self.catalog, self.client, "test-storage")
        self.router = StorageRouter(self.db, self.legacy, self.spaces, self.catalog, "spaces")
        self.bucket, self.path = "fez-training-models", f"{uuid.uuid4()}/candidate.bin"

    def test_legacy_reads_and_writes_never_shadow_a_completed_object(self):
        self.legacy.files[self.bucket, self.path] = b"existing model"
        self.assertTrue(self.router.exists(self.bucket, self.path))
        self.assertEqual(self.router.signed(self.bucket, self.path)["provider"], "supabase")
        with self.assertRaises(APIError):
            self.router.signed(self.bucket, self.path, upload=True)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model"
            self.router.download(self.bucket, self.path, dest, 100)
            self.assertEqual(dest.read_bytes(), b"existing model")
        self.assertEqual(self.s3.uploads, {})

    def test_rollback_keeps_spaces_files_readable_and_records_new_legacy_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model"
            source.write_bytes(b"spaces model")
            self.router.upload(self.bucket, self.path, source)
            rollback = StorageRouter(self.db, self.legacy, self.spaces, self.catalog, "supabase")
            self.assertEqual(rollback.signed(self.bucket, self.path)["provider"], "spaces")
            destination = Path(tmp) / "downloaded"
            rollback.download(self.bucket, self.path, destination, 100)
            self.assertEqual(destination.read_bytes(), b"spaces model")
            fresh = f"{uuid.uuid4()}/new-model"
            rollback.upload(self.bucket, fresh, source)
            row = self.catalog.get(self.bucket, fresh)
            self.assertEqual((row["state"], row["provider"]), ("ready", "supabase"))

    def test_tombstone_blocks_legacy_fallback_even_if_source_copy_remains(self):
        self.legacy.files[self.bucket, self.path] = b"old copy"
        self.router.remove(self.bucket, [self.path])
        self.legacy.files[self.bucket, self.path] = b"retained source"
        self.assertFalse(self.router.exists(self.bucket, self.path))
        with self.assertRaises(APIError) as error:
            self.router.signed(self.bucket, self.path)
        self.assertEqual(error.exception.status, 404)

    def test_catalog_failure_never_falls_back_to_legacy(self):
        self.legacy.files[self.bucket, self.path] = b"old copy"
        with patch.object(self.db, "rpc", side_effect=APIError(503, "database unavailable")):
            with self.assertRaises(APIError):
                self.router.signed(self.bucket, self.path)

    def test_catalog_off_keeps_auth_construction_independent_of_spaces_credentials(self):
        with patch.dict(
            os.environ, {"ZILS_STORAGE_CATALOG": "off", "ZILS_STORAGE_WRITE_PROVIDER": "spaces"}
        ):
            db = Supabase("https://example.supabase.co", "server-key")
            self.assertEqual(db.url, "https://example.supabase.co")
            with self.assertRaises(ValueError):
                db.signed(self.bucket, self.path, upload=True)

    def test_pending_migration_reads_source_until_copy_verifies(self):
        self.legacy.files[self.bucket, self.path] = b"retained legacy bytes"
        token = str(uuid.uuid4())
        self.catalog.change(self.bucket, self.path, "register_legacy", token, {"size_bytes": 21})
        self.catalog.change(
            self.bucket,
            self.path,
            "begin_copy",
            token,
            {"physical_bucket": "test-storage", "sha256": "a" * 64},
        )
        self.assertEqual(self.router.signed(self.bucket, self.path)["provider"], "supabase")
        self.assertTrue(self.router.exists(self.bucket, self.path))


class ImageExpiryTest(unittest.TestCase):
    def test_image_grant_records_explicit_expiry_without_reading_a_jwt(self):
        from zils.image_store import ImageStore

        owner, asset = str(uuid.uuid4()), str(uuid.uuid4())
        expiry = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

        class DB:
            recorded = None

            def rpc(self, name, values):
                if name == "zils_image_grant":
                    self.recorded = values["p_expires"]
                    return None
                return {
                    "id": asset,
                    "owner_id": owner,
                    "state": "uploading",
                    "expires_at": expiry,
                    "source_path": f"{owner}/{asset}/source",
                }

            def signed(self, *args, **kwargs):
                return {
                    "url": "https://private.nyc3.digitaloceanspaces.com/photo?uploadId=one&partNumber=1",
                    "expires_at": expiry,
                    "provider": "spaces",
                    "method": "PUT",
                    "headers": {},
                }

        db = DB()
        result = ImageStore(db).create(owner, "prediction", None, "photo.png", 3, "a" * 64)
        self.assertEqual(result["asset"]["id"], asset)
        self.assertEqual(db.recorded, expiry)
