"""Automatic assignment and publication preserve consent, capacity and ownership."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from tests.test_adapter_releases import OTHER, OWNER, Source, fixture
from zils.adapter_releases import publish, register, registry_entry
from zils.api import Registry


class Store(Source):
    def __init__(self, job, source):
        super().__init__(job, source)
        self.assigned = []
        self.enabled = True

    def rows(self, table, query):
        if table == "fez_training_workers":
            return [{"hotkey": "approved", "enabled": self.enabled, "uid": 1}]
        if table == "fez_training_assignments":
            return []
        if query == f"id=eq.{self.job['id']}":
            return [copy.deepcopy(self.job)]
        if "status=eq." + self.job["status"] in query:
            return [copy.deepcopy(self.job)]
        return []

    def patch(self, table, query, values):
        assert f"status=eq.{self.job['status']}" in query
        self.job.update(copy.deepcopy(values))
        return [copy.deepcopy(self.job)]

    def rpc(self, name, values):
        assert name == "fez_approve_training_job"
        self.assigned.append(values)
        self.job["status"] = "queued"


class WorkflowTest(unittest.TestCase):
    def test_restricted_registration_accepts_only_private_local_adapter_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            release = publish(Source(job, root / "source"), job["id"], root / "releases")
            entry = registry_entry(release, "http://127.0.0.1:8921", "TOKEN")
            path = root / "models.json"
            command = [
                sys.executable,
                "-m",
                "zils.workflow",
                "register",
                "--registry",
                str(path),
                "--customer-only",
                "--runtime-url",
                "http://127.0.0.1:8921",
                "--token-env",
                "TOKEN",
            ]
            for changed in (
                {"owners": None},
                {"url": "https://external.example"},
                {"aliases": ["zils-shared"]},
                {"id": "shared-release"},
            ):
                result = subprocess.run(
                    command, input=json.dumps({**entry, **changed}), text=True, capture_output=True
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(path.exists())
            result = subprocess.run(
                command, input=json.dumps(entry), text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"model_id": entry["id"]})
            self.assertEqual(json.loads(path.read_text())["models"], [entry])

    def test_capacity_deferral_preserves_retry_budget_and_cannot_reuse_lease(self):
        from bittensor_wallet import Keypair

        from tests.test_queue import Store as QueueStore
        from zils import coordinator, queue_protocol
        from zils.cloud import APIError

        store = QueueStore()
        key = Keypair.create_from_seed("0x" + "31" * 32)
        job_id, token = str(uuid.uuid4()), str(uuid.uuid4())
        row = {
            "job_id": job_id,
            "hotkey": key.ss58_address,
            "state": "leased",
            "lease_token": token,
            "attempts": 3,
        }
        store.tables[coordinator.ASSIGNMENTS].append(row)
        store.tables["fez_training_workers"].append({"hotkey": key.ss58_address, "enabled": True})
        service = coordinator.Service(store, "https://coordinator.example")
        path = "/v1/workers/defer"

        def message():
            return queue_protocol.sign(
                key, service.audience, path, {"job_id": job_id, "lease_token": token}
            )

        service.worker(path, message())
        self.assertEqual(row["state"], "ready")
        self.assertEqual(row["attempts"], 2)
        self.assertIsNone(row["lease_token"])
        with self.assertRaises(APIError):
            service.worker(path, message())
        self.assertEqual(row["attempts"], 2)

    def test_capacity_probe_failure_does_not_claim_a_job(self):
        from miner.queue import run_once
        from zils.runtime import gpu_ready

        class NoClaim:
            def call(self, *args):
                raise AssertionError("No capacity: no job should be claimed")

        with patch.dict(
            "os.environ", {"FEZ_GPU_MIN_FREE_MIB": "10000", "FEZ_NVIDIA_SMI": "/missing-probe"}
        ):
            self.assertFalse(gpu_ready("cuda"))
            self.assertFalse(run_once(NoClaim(), None, None, None, "cuda"))

    def test_pending_progress_is_not_misrepresented_as_an_evaluation_result(self):
        from zils.coordinator import public_job

        job = {
            "id": str(uuid.uuid4()),
            "status": "awaiting_approval",
            "result": {"workflow": {"state": "waiting_capacity"}},
        }
        public = public_job(job)
        self.assertIsNone(public["result"])
        self.assertEqual(public["workflow"]["state"], "waiting_capacity")

    def test_new_adapter_is_visible_without_restart_and_shared_requests_disable_adapter(self):
        from tests.test_adapter_server import Backend
        from zils.adapter_server import SharedEngine

        class SharedBackend(Backend):
            def shared_prepare(self, body):
                return {"reserved_tokens": 3}

            def shared_predict(self, prepared):
                return {
                    "route": {
                        "probabilities": {"billing": 0.25, "support": 0.75},
                        "input_tokens": 3,
                    }
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "releases").mkdir()
            backend = SharedBackend()
            backend.release.set()
            engine = SharedEngine(root / "releases", backend, "shared")
            request = {
                "model": "shared",
                "state": {},
                "questions": {
                    "route": {
                        "type": "choice",
                        "criteria": {"billing": "Billing", "support": "Support"},
                    }
                },
            }
            before = engine.predict(engine.prepare(request))
            job = fixture(root / "source")
            release = publish(Source(job, root / "source"), job["id"], root / "releases")
            customer = {**request, "model": release["release_id"]}
            prediction = engine.predict(engine.prepare(customer))
            self.assertEqual(prediction["route"]["probabilities"]["billing"], 1.0)
            self.assertEqual(engine.predict(engine.prepare(request)), before)
            self.assertEqual(engine.predict(engine.prepare(customer)), prediction)

    def test_capacity_and_consent_gate_assignment_without_losing_uploaded_data(self):
        from zils.workflow import Workflow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            job.update(status="awaiting_approval", result=None)
            store = Store(job, root / "source")
            capacity = {"ready": False, "reason": "Waiting for GPU memory."}
            flow = Workflow(
                store, "approved", root / "releases", lambda: capacity, lambda release: None
            )
            original = copy.deepcopy(job["manifest"])
            flow.tick()
            self.assertEqual(store.assigned, [])
            self.assertEqual(job["result"]["workflow"]["state"], "waiting_capacity")
            self.assertEqual(job["manifest"], original)
            capacity["ready"] = True
            store.enabled = False
            flow.tick()
            self.assertEqual(store.assigned, [])
            store.enabled = True
            job["manifest"]["data_access"] = "not-authorized"
            flow.tick()
            self.assertEqual(store.assigned, [])
            job["manifest"] = original
            flow.tick()
            self.assertEqual(store.assigned, [{"p_job": job["id"], "p_hotkeys": ["approved"]}])
            self.assertEqual(job["status"], "queued")

    def test_accepted_release_becomes_owner_model_only_after_activation_succeeds(self):
        from zils.workflow import Workflow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            store = Store(job, root / "source")
            registry = root / "models.json"
            calls = []

            def activate(release):
                calls.append(release["release_id"])
                if len(calls) == 1:
                    raise OSError("Temporary runtime outage")
                register(registry, registry_entry(release, "http://127.0.0.1:8921", "TOKEN"))
                return release["release_id"]

            flow = Workflow(store, "approved", root / "releases", lambda: {"ready": True}, activate)
            flow.tick()
            self.assertFalse(registry.exists())
            self.assertEqual(job["result"]["workflow"]["state"], "activation_failed")
            delivery = copy.deepcopy(job["result"]["delivery"])
            flow.tick()
            self.assertEqual(job["result"]["workflow"]["state"], "ready")
            catalog = Registry(json.loads(registry.read_text())["models"])
            self.assertEqual(catalog.resolve(calls[-1], OWNER)["id"], calls[-1])
            self.assertEqual(catalog.listing(OTHER), {"models": []})
            self.assertEqual(job["result"]["delivery"], delivery)
            flow.tick()
            self.assertEqual(len(calls), 2)

    def test_rejected_candidate_never_activates(self):
        from zils.workflow import Workflow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            job["result"]["delivery"]["status"] = "no_qualifying_model"
            store = Store(job, root / "source")
            calls = []
            Workflow(
                store, "approved", root / "releases", lambda: {"ready": True}, calls.append
            ).tick()
            self.assertEqual(calls, [])
            self.assertFalse((root / "releases").exists())

    def test_registry_reload_preserves_existing_requests_and_rejects_invalid_update(self):
        from zils.api import FileRegistry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = fixture(root / "source")
            release = publish(Source(job, root / "source"), job["id"], root / "releases")
            entry = registry_entry(release, "http://127.0.0.1:8921", "TOKEN")
            path = root / "models.json"
            shared = {**entry, "id": "shared", "owners": None, "aliases": ["default"]}
            register(path, {**shared, "owners": [OWNER]})
            current = FileRegistry(path)
            frozen = current.resolve("default", OWNER)
            register(path, entry)
            self.assertEqual(
                current.resolve(entry["id"], OWNER), Registry([entry]).resolve(entry["id"], OWNER)
            )
            self.assertEqual(current.resolve(frozen["id"], OWNER), frozen)
            path.write_text('{"models": [{"invalid": true}]}')
            self.assertEqual(current.resolve(frozen["id"], OWNER), frozen)
            path.write_text(json.dumps({"models": [frozen, {**entry, "url": 123}]}))
            self.assertEqual(current.resolve(frozen["id"], OWNER), frozen)
            path.write_text('{"models": []}')
            self.assertEqual(current.resolve(frozen["id"], OWNER), frozen)
            path.write_text(json.dumps({"models": [frozen, {**entry, "owners": ["invalid-uuid"]}]}))
            self.assertEqual(current.resolve(frozen["id"], OWNER), frozen)


if __name__ == "__main__":
    unittest.main()
