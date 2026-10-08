"""Image dataset identity and private worker exports."""

import copy
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from zils import benchmark, models

OWNER = "10000000-0000-4000-8000-000000000002"
JOB = "10000000-0000-4000-8000-000000000003"
POLICY = {
    "min_accuracy": 0.8,
    "min_brier_improvement": 0.01,
    "positive_class": "damaged",
    "min_positive_recall": 0.9,
    "max_false_positive_rate": 0.1,
}


def fixture():
    job = {
        "id": JOB,
        "owner_id": OWNER,
        "status": "uploading",
        "model_profile": models.spec(models.IMAJEV),
        "acceptance": POLICY,
    }
    splits, assets = {}, {}
    for split in ("train", "calibration", "test"):
        splits[split] = []
        for i, label in enumerate(("normal", "damaged")):
            aid = str(uuid.uuid4())
            sha = hashlib.sha256((split + str(i)).encode()).hexdigest()
            assets[aid] = {
                "id": aid,
                "owner_id": OWNER,
                "job_id": JOB,
                "purpose": "training",
                "state": "ready",
                "canonical_sha256": sha,
                "pixel_sha256": sha,
                "canonical_bytes": 100,
                "width": 8,
                "height": 8,
                "preprocessor": models.spec(models.IMAJEV)["preprocessor"],
                "expires_at": "2099-01-01T00:00:00Z",
                "filename": "secret-" + label + ".png",
                "canonical_path": "private/path",
            }
            splits[split].append(
                {
                    "id": split + str(i),
                    "group_id": split + str(i),
                    "family": "inspection",
                    "state": {},
                    "question": {
                        "type": "choice",
                        "instructions": "Inspect",
                        "criteria": {"normal": None, "damaged": None},
                    },
                    "label": label,
                    "image": {"asset_id": aid},
                }
            )
    return job, splits, assets


class ImageJobsTest(unittest.TestCase):
    def test_content_and_outcome_order_participate_but_filenames_do_not(self):
        from zils.image_jobs import case_fingerprint

        _, splits, _ = fixture()
        row = splits["train"][0]
        self.assertNotEqual(case_fingerprint(row, "a" * 64), case_fingerprint(row, "b" * 64))
        self.assertEqual(
            case_fingerprint(row, "a" * 64),
            case_fingerprint({**row, "filename": "answer.png"}, "a" * 64),
        )
        changed = copy.deepcopy(row)
        changed["question"]["criteria"] = {"damaged": None, "normal": None}
        self.assertNotEqual(case_fingerprint(row, "a" * 64), case_fingerprint(changed, "a" * 64))

    def test_distinct_photos_with_identical_prompts_and_private_export(self):
        from zils import image_jobs

        job, splits, assets = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "job"
            manifest = image_jobs.build(root, job, splits, assets, POLICY)
            self.assertEqual(benchmark.audit(root), splits)
            exported = (root / "miner-training.jsonl").read_text()
            for row in splits["calibration"] + splits["test"]:
                self.assertNotIn(row["id"], exported)
                self.assertNotIn(row["image"]["asset_id"], exported)
            for value in ("secret-", "private/path", "https:", "owner_id", "group_id"):
                self.assertNotIn(value, exported)
            self.assertEqual(
                json.loads(exported.splitlines()[0])["images"][0]["sha256"],
                assets[splits["train"][0]["image"]["asset_id"]]["canonical_sha256"],
            )
            tampered = copy.deepcopy(manifest)
            tampered["assets"][splits["train"][0]["image"]["asset_id"]]["canonical_sha256"] = (
                "f" * 64
            )
            with self.assertRaises(ValueError):
                image_jobs.audit(root, tampered)
            self.assertEqual(models.job_model(job), models.IMAJEV)
            with self.assertRaises(ValueError):
                models.job_model({**job, "manifest": {"model": models.spec(models.JEVK5)}})

    def test_server_rejects_tenant_binding_duplicates_groups_and_incomplete_data(self):
        from zils import image_jobs

        for change in (
            "owner_id",
            "job_id",
            "purpose",
            "state",
            "expires_at",
            "canonical_sha256",
            "pixels",
            "groups",
            "coverage",
            "question",
            "client_hash",
        ):
            job, splits, assets = fixture()
            train = assets[splits["train"][0]["image"]["asset_id"]]
            test = assets[splits["test"][0]["image"]["asset_id"]]
            if change in ("owner_id", "job_id"):
                test[change] = str(uuid.uuid4())
            elif change == "purpose":
                test[change] = "prediction"
            elif change == "state":
                test[change] = "verifying"
            elif change == "expires_at":
                test[change] = "2000-01-01T00:00:00Z"
            elif change == "canonical_sha256":
                test[change] = train[change]
            elif change == "pixels":
                test["pixel_sha256"] = train["pixel_sha256"]
            elif change == "groups":
                splits["test"][0]["group_id"] = splits["train"][0]["group_id"]
            elif change == "coverage":
                splits["test"].pop()
            elif change == "question":
                splits["test"][0]["question"]["instructions"] = "Different task"
            elif change == "client_hash":
                splits["test"][0]["image"]["sha256"] = "f" * 64
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as tmp,
                self.assertRaises(ValueError),
            ):
                image_jobs.build(Path(tmp) / "job", job, splits, assets, POLICY)

    def test_worker_downloads_validate_all_ids_before_any_grant(self):
        from zils import image_jobs
        from zils.coordinator import Service

        job, splits, assets = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = image_jobs.build(Path(tmp) / "job", job, splits, assets, POLICY)
            job.update(
                manifest=manifest, job_sha256=benchmark.file_hash(Path(tmp) / "job/manifest.json")
            )

            class Store:
                def __init__(self):
                    self.grants = []

                def rows(self, table, query):
                    return (
                        [job]
                        if table == "fez_training_jobs"
                        else [{"job_id": JOB, "state": "leased"}]
                    )

                def rpc(self, name, args):
                    return assets.get(args.get("p_asset"))

                def signed(self, bucket, path):
                    self.grants.append(path)
                    return {"url": "https://storage.example/photo"}

            store = Store()
            service = Service(store, "https://queue.example")
            body = {
                "job_id": JOB,
                "lease_token": str(uuid.uuid4()),
                "asset_ids": [
                    splits["train"][0]["image"]["asset_id"],
                    splits["test"][0]["image"]["asset_id"],
                ],
            }
            with patch("zils.coordinator.queue_protocol.verify", return_value=("worker", body)):
                with self.assertRaises(Exception):
                    service.worker("/v1/workers/image-downloads", {})
                self.assertEqual(store.grants, [])
                body["asset_ids"].pop()
                result = service.worker("/v1/workers/image-downloads", {})
                self.assertEqual(len(result["images"]), 1)
                self.assertEqual(len(store.grants), 1)
                assets.clear()
                with self.assertRaises(Exception) as failure:
                    service.worker("/v1/workers/image-downloads", {})
                self.assertEqual(failure.exception.status, 404)
                import requests

                from tests.test_queue import server
                from zils.coordinator import handler

                with server(handler(service, "https://app.example")) as url:
                    response = requests.post(
                        url + "/v1/workers/image-downloads", json={}, timeout=5
                    )
                self.assertEqual(response.status_code, 404)

    def test_customer_creation_freezes_profile_before_upload_and_feature_off_creates_nothing(self):
        from tests.test_queue import Store
        from zils.coordinator import Service

        store = Store()
        store.url = "https://storage.example"
        service = Service(store, "https://queue.example", models.JEVK5)
        body = {
            "name": "image-inspection",
            "model": models.IMAJEV,
            "acceptance": POLICY,
            "allow_training_data_export": True,
            "image_intake": {
                "version": "zils-image-intake/v1",
                "seed": "fixture",
                "snapshot_sha256": "a" * 64,
            },
        }
        with patch.dict(
            "os.environ", {"ZILS_IMAGES_ENABLED": "false", "ZILS_IMAGE_TRAINING_ENABLED": "true"}
        ):
            with self.assertRaises(Exception):
                service.customer("POST", "/v1/jobs", "owner-token", body)
            self.assertEqual(store.tables["fez_training_jobs"], [])
        with patch.dict(
            "os.environ", {"ZILS_IMAGES_ENABLED": "true", "ZILS_IMAGE_TRAINING_ENABLED": "true"}
        ):
            result = service.customer("POST", "/v1/jobs", "owner-token", body)
        self.assertEqual(result["job"]["model"], models.spec(models.IMAJEV))
        frozen = store.tables["fez_training_jobs"][0]
        self.assertEqual(models.job_model(frozen), models.IMAJEV)
        self.assertIsNone(result["job"]["result"])
        self.assertEqual(result["job"].get("image_intake"), body["image_intake"])

    def test_upgrade_rejects_cross_family_and_predecessor_training_holdout_overlap(self):
        from tests.test_version_selection import versioned_fixture
        from zils import image_jobs, version_selection

        with tempfile.TemporaryDirectory() as tmp:
            parent = versioned_fixture(Path(tmp) / "parent")
            parent["owner_id"] = OWNER
            parent["result"]["workflow"] = {
                "state": "ready",
                "model_id": f"zils-adapter-{parent['id']}-{parent['result']['delivery']['sha256']}",
            }
            job, splits, assets = fixture()
            job["acceptance"] = {**POLICY, "previous_job_id": parent["id"]}

            class Store:
                def rows(self, *args):
                    return [parent]

            with self.assertRaisesRegex(ValueError, "profile|family"):
                version_selection.freeze(Store(), job)
            manifest = image_jobs.build(
                Path(tmp) / "image", {**job, "acceptance": POLICY}, splits, assets, POLICY
            )
            previous = copy.deepcopy(manifest)
            aid = splits["test"][0]["image"]["asset_id"]
            previous["assets"][aid]["split"] = "train"
            with self.assertRaisesRegex(ValueError, "overlap"):
                image_jobs.validate_predecessor(manifest, previous)
