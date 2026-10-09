"""Migration copies keep legacy bytes readable until independently verified."""

import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from scripts.migrate_storage import fingerprint, inventory, migrate_one, write_report
from tests import test_spaces as fixture
from tests.api_database import literal
from tests.spaces_fixture import LostReplyDB
from zils.cloud import APIError
from zils.spaces import SpacesStorage
from zils.storage import ObjectCatalog, StorageRouter


class Legacy:
    def __init__(self, raw=b"original model"):
        self.raw = raw
        self.downloads = 0
        self.removed = []

    def stat(self, bucket, path):
        return {"size": len(self.raw)}

    def download(self, bucket, path, target, limit, **kwargs):
        self.downloads += 1
        target.write_bytes(self.raw)

    def remove(self, bucket, paths):
        self.removed.extend(paths)


class InventoryDB:
    def __init__(self):
        self.calls = []
        self.job = str(uuid.uuid4())
        self.status = "completed"

    def request(self, method, path, body):
        self.calls.append(body)
        if not path.endswith("/fez-training-data"):
            return []
        if not body["prefix"]:
            return [{"name": self.job, "id": None, "metadata": None}]
        if body["prefix"] == self.job:
            return [{"name": "inputs", "id": None, "metadata": None}]
        rows = [
            {"name": f"{x}.jsonl", "id": str(uuid.uuid4()), "metadata": {"size": 3}}
            for x in ("calibration", "test", "train")
        ]
        return rows[body["offset"] : body["offset"] + body["limit"]]

    def rows(self, table, query):
        return [{"id": self.job, "status": self.status}]


class MigrationTests(unittest.TestCase):
    setUpClass = classmethod(fixture.SpacesTest.setUpClass.__func__)
    tearDownClass = classmethod(fixture.SpacesTest.tearDownClass.__func__)
    setUp = fixture.SpacesTest.setUp

    def test_migration_retains_outstanding_legacy_upload_expiry(self):
        legacy = Legacy()
        row = self.catalog.change(
            self.bucket,
            self.path,
            "register_legacy",
            str(uuid.uuid4()),
            {"size_bytes": len(legacy.raw)},
        )
        self.db.sql(
            f"update zils_storage_objects set grant_expires_at=now()+interval '2 hours' where generation={literal(row['generation'])}"
        )
        expiry = self.catalog.get(self.bucket, self.path)["grant_expires_at"]
        self.assertEqual(self.copy(legacy)["copy_state"], "verified")
        copied = self.catalog.get(self.bucket, self.path)
        self.assertEqual(copied["grant_expires_at"], expiry)
        deleting = self.catalog.change(self.bucket, self.path, "delete", str(uuid.uuid4()))
        from zils.storage import timestamp

        self.assertGreater(timestamp(deleting["cleanup_after"]), timestamp(expiry))

    def test_deferred_upload_pages_do_not_starve_deletions_or_later_work(self):
        # Insert expired but ineligible uploads ahead of a later cleanup target.
        prefix = str(uuid.uuid4())
        self.db.sql(
            f"insert into zils_storage_objects(bucket,path,provider,physical_bucket,physical_key,state,token,max_bytes,updated_at,lease_until,grant_expires_at) select 'fez-training-data',{literal(prefix)}||'/'||n,'spaces','test-storage','deferred-'||{literal(prefix)}||'/'||n,'allocating',gen_random_uuid(),100,now()-interval '3 days',now()-interval '3 days',now()-interval '3 days' from generate_series(1,105) n"
        )
        row = self.catalog.change(
            self.bucket, self.path, "register_legacy", str(uuid.uuid4()), {"size_bytes": 3}
        )
        self.catalog.change(self.bucket, self.path, "delete", str(uuid.uuid4()))
        self.db.sql(
            f"update zils_storage_objects set cleanup_after=now()-interval '1 second',lease_until=now()-interval '1 second',grant_expires_at=now()-interval '1 second' where generation={literal(row['generation'])}"
        )
        legacy = Legacy()
        router = StorageRouter(self.db, legacy, self.store, self.catalog, "spaces")
        visited = []
        router.reap(eligible=lambda row: visited.append(row["path"]) or False)
        self.assertEqual(self.catalog.get(self.bucket, self.path)["state"], "deleted")
        self.assertGreaterEqual(len([p for p in visited if p.startswith(prefix)]), 105)

    def test_abandoned_upload_cleanup_is_dry_run_and_fenced(self):
        self.store.signed(self.bucket, self.path, upload=True)
        row = self.catalog.get(self.bucket, self.path)
        self.db.sql(
            f"update zils_storage_objects set updated_at=now()-interval '2 days', lease_until=now()-interval '2 days', grant_expires_at=now()-interval '2 days' where generation={literal(row['generation'])}"
        )
        self.store.reap(dry_run=True, eligible=lambda row: True)
        self.assertEqual(self.catalog.get(self.bucket, self.path)["generation"], row["generation"])
        self.store.reap(eligible=lambda row: False)
        self.assertIn(row["upload_id"], self.s3.uploads)
        self.store.reap(eligible=lambda row: True)
        self.assertNotIn(row["upload_id"], self.s3.uploads)
        self.assertNotEqual(
            self.catalog.get(self.bucket, self.path)["generation"], row["generation"]
        )
        self.assertEqual(
            self.store.signed(self.bucket, self.path, upload=True)["provider"], "spaces"
        )

    def item(self, legacy):
        return {
            "bucket": self.bucket,
            "path": self.path,
            "source_size": len(legacy.raw),
            "source_sha256": None,
            "eligibility": "eligible",
        }

    def copy(self, legacy, store=None, item=None):
        with patch("scripts.migrate_storage.eligibility", return_value=("eligible", "completed")):
            return migrate_one(
                self.db, legacy, store or self.store, item or self.item(legacy), apply=True
            )

    def test_verified_copy_and_rerun_and_rollback(self):
        legacy = Legacy()
        result = self.copy(legacy)
        self.assertEqual(result["copy_state"], "verified")
        row = self.catalog.get(self.bucket, self.path)
        self.assertEqual(row["sha256"], hashlib.sha256(legacy.raw).hexdigest())
        self.assertFalse(row["legacy_readable"])
        self.assertTrue(row["legacy_copy"])
        self.assertEqual(self.copy(legacy)["copy_state"], "verified")
        self.assertEqual(len(self.s3.objects), 1)
        router = StorageRouter(self.db, legacy, self.store, self.catalog, "supabase")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "read"
            router.download(self.bucket, self.path, target, 100)
            self.assertEqual(target.read_bytes(), legacy.raw)
        self.assertEqual(legacy.removed, [])

    def test_mismatch_does_not_publish(self):
        legacy = Legacy()
        original = self.store._call

        def changed(method, **values):
            result = original(method, **values)
            if method == "complete_multipart_upload":
                self.s3.objects[values["Key"]]["file"].write_bytes(b"corrupt model!")
            return result

        with patch.object(self.store, "_call", changed):
            result = self.copy(legacy)
        self.assertNotEqual(result["copy_state"], "verified")
        row = self.catalog.get(self.bucket, self.path)
        self.assertTrue(row["legacy_readable"])
        self.assertNotEqual(row["state"], "ready")

    def test_lost_commit_reply_reconciles_without_duplicate(self):
        legacy = Legacy()
        store = SpacesStorage(
            ObjectCatalog(LostReplyDB(self.db, "commit")), self.client, "test-storage"
        )
        self.copy(legacy, store)
        self.assertEqual(self.copy(legacy, store)["copy_state"], "verified")
        self.assertEqual(len(self.s3.objects), 1)

    def test_transfer_reply_loss_can_retry_without_source_deletion(self):
        legacy = Legacy()
        with patch.object(legacy, "download", side_effect=APIError(503, "lost")):
            self.assertEqual(self.copy(legacy)["copy_state"], "retry")
        self.assertIsNone(self.catalog.get(self.bucket, self.path))
        self.s3.lose_completion = True
        self.assertEqual(self.copy(legacy)["copy_state"], "verified")
        self.assertEqual(legacy.removed, [])

    def test_changed_inventory_identity_cannot_copy_new_upload(self):
        legacy = Legacy()
        item = self.item(legacy) | {"source_id": str(uuid.uuid4())}
        self.assertEqual(self.copy(legacy, item=item)["copy_state"], "failed")
        self.assertEqual(legacy.downloads, 0)

    def test_authorized_removal_also_removes_retained_photo_copy(self):
        legacy = Legacy()
        self.copy(legacy)
        StorageRouter(self.db, legacy, self.store, self.catalog, "spaces").remove(
            self.bucket, [self.path]
        )
        self.assertEqual(legacy.removed, [self.path])
        self.assertEqual(self.catalog.get(self.bucket, self.path)["state"], "deleting")

    def test_expired_deletion_cleans_legacy_after_all_grants(self):
        legacy = Legacy()
        self.copy(legacy)
        router = StorageRouter(self.db, legacy, self.store, self.catalog, "spaces")
        router.remove(self.bucket, [self.path])
        row = self.catalog.get(self.bucket, self.path)
        self.db.sql(
            f"update zils_storage_objects set cleanup_after=now()-interval '1 second',lease_until=now()-interval '1 second',grant_expires_at=now()-interval '1 second' where generation={literal(row['generation'])}"
        )
        router.reap()
        self.assertEqual(legacy.removed.count(self.path), 2)
        self.assertEqual(self.catalog.get(self.bucket, self.path)["state"], "deleted")

    def test_deleted_source_never_reappears(self):
        legacy = Legacy()
        self.catalog.change(self.bucket, self.path, "delete", str(uuid.uuid4()))
        self.assertNotEqual(self.copy(legacy)["copy_state"], "verified")
        self.assertEqual(legacy.downloads, 0)
        self.assertFalse(
            StorageRouter(self.db, legacy, self.store, self.catalog, "supabase").exists(
                self.bucket, self.path
            )
        )

    def test_delete_during_copy_prevents_publish(self):
        legacy = Legacy()
        original = self.store._call

        def deleted(method, **values):
            result = original(method, **values)
            if method == "complete_multipart_upload":
                self.catalog.change(self.bucket, self.path, "delete", str(uuid.uuid4()))
            return result

        with patch.object(self.store, "_call", deleted):
            self.assertNotEqual(self.copy(legacy)["copy_state"], "verified")
        self.assertEqual(self.catalog.get(self.bucket, self.path)["state"], "deleting")

    def test_zero_truncated_changed_sources_and_dry_run(self):
        for raw, size in [(b"", 0), (b"ab", 3)]:
            self.path = f"{uuid.uuid4()}/inputs/train.jsonl"
            legacy = Legacy(raw)
            item = self.item(legacy) | {"source_size": size}
            self.assertNotEqual(self.copy(legacy, item=item)["copy_state"], "verified")
            self.assertFalse(self.catalog.get(self.bucket, self.path))
        with patch("scripts.migrate_storage.eligibility", return_value=("deferred", "running")):
            self.assertEqual(
                migrate_one(self.db, Legacy(), self.store, self.item(Legacy()), apply=True)[
                    "copy_state"
                ],
                "deferred",
            )
        self.assertEqual(
            migrate_one(self.db, Legacy(), None, self.item(Legacy()), apply=False)["copy_state"],
            "planned",
        )


class InventoryTests(unittest.TestCase):
    def test_fingerprint_and_private_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bytes"
            source.write_bytes(b"abc")
            self.assertEqual(fingerprint(source), (3, hashlib.sha256(b"abc").hexdigest()))
            report = Path(directory) / "report.json"
            write_report(report, {"objects": []})
            self.assertEqual(report.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(report.read_text()), {"objects": []})

    def test_inventory_pages_and_folders_and_active_jobs(self):
        db = InventoryDB()
        result = inventory(db, page_size=2)
        self.assertEqual(len(result["objects"]), 3)
        self.assertTrue(all(item["eligibility"] == "eligible" for item in result["objects"]))
        self.assertTrue(any(call["offset"] == 2 for call in db.calls))
        db.status = "running"
        self.assertTrue(all(item["eligibility"] == "deferred" for item in inventory(db)["objects"]))
