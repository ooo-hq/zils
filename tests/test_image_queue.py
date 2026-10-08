"""Profile-aware admission and image capacity isolation."""

import unittest
import uuid
from unittest.mock import Mock, patch

from tests.test_queue import Store
from zils import coordinator, models
from zils.cloud import APIError


class ImageQueueTest(unittest.TestCase):
    def make_service(self):
        store = Store()
        store.url = "https://storage.example"
        for model in (models.JEVK5, models.IMAJEV):
            job = {
                "id": str(uuid.uuid4()),
                "status": "queued",
                "model_profile": models.spec(model),
                "initial_sha256": "a" * 64,
                "job_sha256": "b" * 64,
                "manifest": {
                    "model": models.spec(model),
                    "files": {"miner-training.jsonl": "c" * 64},
                },
            }
            store.tables[coordinator.JOBS].append(job)
            for worker in ("text", "image"):
                store.tables[coordinator.ASSIGNMENTS].append(
                    {"job_id": job["id"], "hotkey": worker, "state": "ready", "attempts": 0}
                )
        for worker, model in (("text", models.JEVK5), ("image", models.IMAJEV)):
            store.tables["fez_training_workers"].append({"hotkey": worker, "enabled": True})
            store.tables.setdefault("zils_worker_profiles", []).append(
                {
                    "hotkey": worker,
                    "profile_id": model,
                    "enabled": True,
                    "min_free_mib": 12288,
                    "evidence": {"max_seconds": 1200},
                    **models.profile_identity(model),
                }
            )
        return coordinator.Service(store, "https://queue.example"), store

    def call(self, service, worker, body, action="claim"):
        with patch("zils.coordinator.queue_protocol.verify", return_value=(worker, body)):
            return service.worker("/v1/workers/" + action, {})

    def test_installed_and_operator_verified_intersection_filters_before_urls(self):
        service, store = self.make_service()
        for worker, model in (("image", models.IMAJEV), ("text", models.JEVK5)):
            result = self.call(service, worker, {"supported_profiles": [model]})["assignment"]
            self.assertEqual(result["model"]["id"], model)
        before = len(store.tickets)
        with self.assertRaises(APIError):
            self.call(service, "text", {"supported_profiles": [models.IMAJEV]})
        self.assertEqual(len(store.tickets), before)
        self.assertNotEqual(
            self.call(service, "image", {})["assignment"]["model"]["id"], models.IMAJEV
        )

    def test_revoked_runtime_identity_cannot_receive_another_grant(self):
        service, store = self.make_service()
        row = self.call(service, "image", {"supported_profiles": [models.IMAJEV]})["assignment"]
        store.tables["zils_worker_profiles"][1]["runtime_sha256"] = "f" * 64
        count = len(store.tickets)
        with self.assertRaises(APIError):
            self.call(
                service,
                "image",
                {"job_id": row["job_id"], "lease_token": row["lease_token"]},
                "uploads",
            )
        self.assertEqual(len(store.tickets), count)

    def test_workflow_waits_for_operator_qualification_on_the_selected_image_worker(self):
        import tempfile
        from pathlib import Path

        from tests.test_adapter_releases import fixture
        from tests.test_workflow import Store as WorkflowStore
        from zils.workflow import Workflow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            job.update(
                status="awaiting_approval", result=None, model_profile=models.spec(models.IMAJEV)
            )
            job["manifest"]["model"] = models.spec(models.IMAJEV)

            class QualifiedStore(WorkflowStore):
                qualified = False

                def rpc(self, name, values):
                    if name == "zils_worker_qualified":
                        return self.qualified
                    return super().rpc(name, values)

            store = QualifiedStore(job, root / "source")
            flow = Workflow(
                store,
                "text",
                root / "releases",
                lambda: {"ready": False},
                lambda _: None,
                image_hotkey="approved",
                image_capacity=lambda: {"ready": True},
            )
            flow.tick()
            self.assertEqual(store.assigned, [])
            self.assertEqual(job["result"]["workflow"]["state"], "waiting_worker")
            store.qualified = True
            flow.tick()
            self.assertEqual(store.assigned, [{"p_job": job["id"], "p_hotkeys": ["approved"]}])

    def test_image_capacity_is_required_even_without_a_text_capacity_setting(self):
        from zils.runtime import gpu_ready

        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(gpu_ready("cpu", model=models.IMAJEV))
            with patch("zils.runtime.subprocess.run", return_value=Mock(stdout="4096")):
                self.assertFalse(gpu_ready("cuda", model=models.IMAJEV))

    def test_low_capacity_skips_claim_without_consuming_an_attempt(self):
        from miner.queue import run_once

        client = Mock()
        with patch("miner.queue.gpu_ready", return_value=False):
            self.assertFalse(
                run_once(client, None, None, None, "cuda", references={models.IMAJEV: "ref"})
            )
        client.call.assert_not_called()

    def test_lost_lease_terminates_image_child_before_timeout(self):
        import sys
        import tempfile
        import time
        from pathlib import Path

        from zils.runtime import run_child

        with tempfile.TemporaryDirectory() as tmp:
            started = time.monotonic()
            with (
                patch("zils.runtime.gpu_ready", return_value=True),
                patch.dict("os.environ", {"ZILS_COMPUTE_LOCK": str(Path(tmp) / "lock")}),
            ):
                with self.assertRaises(APIError):
                    run_child(
                        [sys.executable, "-c", "import time; time.sleep(20)"],
                        Path(tmp) / "log",
                        "cuda",
                        model=models.IMAJEV,
                        timeout=25,
                        check_lease=Mock(side_effect=APIError(409, "cancelled")),
                    )
            self.assertLess(time.monotonic() - started, 3)

    def test_cancelled_job_cannot_refresh_image_urls(self):
        service, store = self.make_service()
        assignment = self.call(service, "image", {"supported_profiles": [models.IMAJEV]})[
            "assignment"
        ]
        job = next(j for j in store.tables[coordinator.JOBS] if j["id"] == assignment["job_id"])
        job["status"] = "failed"
        count = len(store.tickets)
        with self.assertRaises(APIError):
            self.call(
                service,
                "image",
                {
                    "job_id": job["id"],
                    "lease_token": assignment["lease_token"],
                    "asset_ids": [str(uuid.uuid4())],
                },
                "image-downloads",
            )
        self.assertEqual(len(store.tickets), count)

    def test_fixture_http_image_job_accepts_a_qualifying_candidate(self):
        from tests.image_flow_fixture import run

        run(self)

    def test_fixture_http_image_job_retains_negative_quality_result(self):
        from tests.image_flow_fixture import run

        run(self, reject=True)
