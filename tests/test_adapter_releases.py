"""Accepted training artifacts become private API releases without database writes."""

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

import fez
from fez import models
from tests.test_jobs import POLICY, examples

OWNER = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


def fixture(root, owner=OWNER, checkpoint=None):
    from fez import jobs

    root.mkdir()
    if checkpoint is None:
        models.write_metadata(root, temperature=0.75)
        (root / "adapter_config.json").write_text("{}")
        (root / "adapter_model.safetensors").write_bytes(
            b"fixture tensors; GPU tests use real tensors"
        )
    else:
        for name in models.JEVK5_FILES:
            shutil.copyfile(Path(checkpoint) / name, root / name)
    checkpoint = fez.checkpoint_hash(root)
    job_id, round_id = str(uuid.uuid4()), str(uuid.uuid4())
    manifest = jobs.build(
        root / "data",
        "support-routing",
        examples(),
        POLICY,
        allow_training_data_export=True,
        model=models.JEVK5,
    )
    manifest_bytes = (root / "data/manifest.json").read_bytes()
    job_sha = hashlib.sha256(manifest_bytes).hexdigest()
    baseline = {"status": "evaluated", "brier": 0.3, "accuracy": 0.6, "skill": 0.5, "uid": 0}
    winner = {
        "status": "evaluated",
        "brier": 0.1,
        "accuracy": 0.9,
        "skill": 0.85,
        "uid": 1,
        "sha256": checkpoint,
    }
    delivery = jobs.select(baseline, [winner], POLICY)
    release = {
        "job_id": job_id,
        "job_sha256": job_sha,
        "round_id": round_id,
        "model": models.spec(models.JEVK5),
        "base": models.spec(models.JEVK5)["base"],
        "base_revision": models.spec(models.JEVK5)["base_revision"],
        "initial_sha256": "a" * 64,
        "submitted_sha256": "b" * 64,
        "baseline_brier": baseline["brier"],
        **delivery,
    }
    (root / "release.json").write_text(json.dumps(release))
    job = {
        "id": job_id,
        "owner_id": owner,
        "name": "support-routing",
        "status": "completed",
        "updated_at": "2026-10-04T00:00:00+00:00",
        "manifest": manifest,
        "job_sha256": job_sha,
        "initial_sha256": "a" * 64,
        "acceptance": POLICY,
        "release_prefix": f"{job_id}/releases/{round_id}",
        "result": {
            "model": models.spec(models.JEVK5),
            "baseline": baseline,
            "miners": [{k: v for k, v in winner.items() if k != "sha256"}],
            "delivery": delivery,
        },
    }
    return job


class Source:
    """Only the read operations available to the release publisher."""

    def __init__(self, job, source):
        self.job, self.source, self.downloads = job, source, []

    def rows(self, table, query):
        assert table == "fez_training_jobs"
        assert query == f"id=eq.{self.job['id']}"
        return [copy.deepcopy(self.job)]

    def download(self, bucket, path, destination, limit):
        self.downloads.append((bucket, path))
        name = path.rsplit("/", 1)[1]
        origin = self.source / ("data/manifest.json" if name == "manifest.json" else name)
        data = origin.read_bytes()
        assert len(data) <= limit
        Path(destination).write_bytes(data)
        return len(data)


class ReleaseTest(unittest.TestCase):
    def test_acceptance_owner_and_hashes_are_bound_and_retry_is_idempotent(self):
        from fez.adapter_releases import publish, read_release

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            store = Source(job, root / "source")
            release = publish(store, job["id"], root / "published")
            self.assertEqual(release["owner_id"], OWNER)
            self.assertEqual(release["temperature"], 0.75)
            self.assertEqual(release["checkpoint_sha256"], job["result"]["delivery"]["sha256"])
            self.assertEqual(read_release(root / "published" / release["release_id"]), release)
            self.assertEqual(publish(store, job["id"], root / "published"), release)
            self.assertEqual(len(store.downloads), 5)
            self.assertTrue(
                all(p[0] in ("fez-training-data", "fez-training-models") for p in store.downloads)
            )

    def test_rejected_or_incomplete_jobs_publish_nothing(self):
        from fez.adapter_releases import publish

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            for status, delivery in (("running", "accepted"), ("completed", "no_qualifying_model")):
                changed = copy.deepcopy(job)
                changed["status"] = status
                changed["result"]["delivery"]["status"] = delivery
                source = Source(changed, root / "source")
                self.assertIsNone(publish(source, job["id"], root / "published"))
                self.assertEqual(source.downloads, [])
            self.assertFalse((root / "published").exists())

    def test_forged_acceptance_and_tampered_artifacts_are_rejected(self):
        from fez.adapter_releases import publish, read_release

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            bad = copy.deepcopy(job)
            bad["result"]["miners"][0]["accuracy"] = 0.1
            with self.assertRaises(ValueError):
                publish(Source(bad, root / "source"), job["id"], root / "bad")
            release = publish(Source(job, root / "source"), job["id"], root / "published")
            target = root / "published" / release["release_id"] / "adapter_model.safetensors"
            target.chmod(0o600)
            target.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                read_release(target.parent)
            with self.assertRaises(ValueError):
                publish(Source(job, root / "source"), job["id"], root / "published")

    def test_registry_preserves_shared_and_old_releases_and_prevents_cross_owner_aliases(self):
        from fez.adapter_releases import merge_registry, publish, registry_entry
        from fez.api import Registry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            release = publish(Source(job, root / "source"), job["id"], root / "published")
            entry = registry_entry(
                release, "http://127.0.0.1:8931", "ADAPTER_TOKEN", alias="support"
            )
            old = {**entry, "id": "older-release", "fingerprint": "a" * 64}
            shared = {**entry, "id": "shared-release", "aliases": ["zils-shared"], "owners": None}
            result = merge_registry({"models": [old, shared]}, entry)
            registry = Registry(result["models"])
            self.assertEqual(registry.resolve("support", OWNER)["id"], release["release_id"])
            self.assertEqual(registry.resolve("older-release", OWNER)["fingerprint"], "a" * 64)
            self.assertEqual(registry.resolve("zils-shared", OTHER)["id"], "shared-release")
            from fez.decisions import DecisionError

            with self.assertRaises(DecisionError):
                registry.resolve("support", OTHER)
            with self.assertRaises(ValueError):
                merge_registry({"models": [{**old, "owners": [OTHER]}]}, entry)
            self.assertEqual(merge_registry(result, entry), result)

    def test_failed_download_or_registry_collision_preserves_previous_release(self):
        from fez.adapter_releases import publish, read_release, register, registry_entry
        from fez.cloud import APIError

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            release = publish(Source(job, root / "source"), job["id"], root / "published")
            entry = registry_entry(release, "http://127.0.0.1:8931", "ADAPTER_TOKEN", "support")
            registry = root / "registry.json"
            register(registry, entry)
            before = registry.read_bytes()
            newer = fixture(root / "newer")

            class Interrupted(Source):
                def download(self, *args):
                    super().download(*args)
                    raise APIError(503, "Interrupted download")

            with self.assertRaises(APIError):
                publish(Interrupted(newer, root / "newer"), newer["id"], root / "published")
            self.assertEqual(registry.read_bytes(), before)
            self.assertEqual(read_release(root / "published" / release["release_id"]), release)
            self.assertEqual(list((root / "published").glob(".stage-*")), [])
            with self.assertRaises(ValueError):
                register(registry, {**entry, "id": "other-customer", "owners": [OTHER]})
            self.assertEqual(registry.read_bytes(), before)
            # Retrying without an alias must not remove a previously assigned customer name.
            register(registry, {**entry, "aliases": []})
            self.assertEqual(registry.read_bytes(), before)

    def test_symlinked_manifest_and_changed_calibration_fail_verification(self):
        from fez.adapter_releases import publish, read_release

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            release = publish(Source(job, root / "source"), job["id"], root / "published")
            target = root / "published" / release["release_id"]
            manifest = target / "manifest.json"
            manifest.unlink()
            manifest.symlink_to(root / "source/data/manifest.json")
            with self.assertRaises(ValueError):
                read_release(target)
            manifest.unlink()
            shutil.copyfile(root / "source/data/manifest.json", manifest)
            model = target / "model.json"
            model.chmod(0o600)
            models.write_metadata(target, temperature=1.5)
            with self.assertRaises(ValueError):
                read_release(target)
