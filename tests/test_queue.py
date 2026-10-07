"""HTTP upload-to-result workflow with real hotkeys and fixture model execution.

Supabase REST/storage are a local double; SQL behavior is tested separately in PostgreSQL.
"""

import copy
import json
import tempfile
import threading
import time
import unittest
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs

import requests
from bittensor_wallet import Keypair

import zils
from miner.queue import Client, run_once
from tests.test_jobs import POLICY, examples
from zils import cloud, coordinator, queue_protocol
from zils.cloud import APIError

OWNER = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


@contextmanager
def server(handler):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as service:
        thread = threading.Thread(target=service.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{service.server_port}"
        finally:
            service.shutdown()
            thread.join(timeout=5)


class Store:
    """HTTP/storage fixture, deliberately not a replacement for SQL queue tests."""

    def __init__(self):
        self.tables = {
            name: [] for name in (coordinator.JOBS, coordinator.ASSIGNMENTS, "fez_training_workers")
        }
        self.objects, self.tickets, self.nonces = {}, {}, set()
        self.url = None

    def user(self, token):
        if token not in ("owner-token", "other-token"):
            raise APIError(401, "Sign in to continue.")
        return OWNER if token == "owner-token" else OTHER

    def matches(self, row, query):
        for key, values in parse_qs(query).items():
            value = values[0]
            if key in ("order", "limit", "select"):
                continue
            if value.startswith("eq.") and str(row.get(key)) != value[3:]:
                return False
            if value.startswith("not.in.") and row.get(key) in value[8:-1].split(","):
                return False
            if value.startswith("gt.") and str(row.get(key)) <= value[3:]:
                return False
        return True

    def rows(self, table, query=""):
        return copy.deepcopy([row for row in self.tables[table] if self.matches(row, query)])

    def patch(self, table, query, values):
        updated = []
        for row in self.tables[table]:
            if self.matches(row, query):
                row.update(values)
                updated.append(copy.deepcopy(row))
        return updated

    def rpc(self, name, p):
        jobs = self.tables[coordinator.JOBS]
        assignments = self.tables[coordinator.ASSIGNMENTS]
        if name == "fez_create_training_job":
            row = {
                "id": str(uuid.uuid4()),
                "owner_id": p["p_owner"],
                "name": p["p_name"],
                "acceptance": p["p_acceptance"],
                "status": "uploading",
                "created_at": "2026-09-30T00:00:00Z",
                "error": None,
                "result": None,
            }
            jobs.append(row)
            return copy.deepcopy(row)
        if name == "fez_worker_nonce":
            if not any(
                w["hotkey"] == p["p_hotkey"] and w["enabled"]
                for w in self.tables["fez_training_workers"]
            ):
                raise APIError(409, "Worker not approved.")
            nonce = (p["p_hotkey"], p["p_nonce"])
            if nonce in self.nonces:
                raise APIError(409, "Replay.")
            self.nonces.add(nonce)
            return None
        if name == "fez_claim_processing":
            row = next(
                (
                    j
                    for j in jobs
                    if (j["status"] == "validating" and p["p_stage"] == "validating")
                    or (
                        j["status"] == "running"
                        and p["p_stage"] == "evaluating"
                        and all(
                            a["state"] == "submitted" for a in assignments if a["job_id"] == j["id"]
                        )
                    )
                ),
                None,
            )
            if row:
                row.update(
                    status=p["p_stage"],
                    lease_token=str(uuid.uuid4()),
                    lease_until="2999-01-01T00:00:00Z",
                )
            return copy.deepcopy(row)
        if name == "fez_finish_processing":
            row = next(
                j for j in jobs if j["id"] == p["p_job"] and j["lease_token"] == p["p_token"]
            )
            row.update(p["p_values"], status=p["p_status"], lease_token=None)
            return None
        if name == "fez_approve_training_job":
            row = next(j for j in jobs if j["id"] == p["p_job"])
            row["status"] = "queued"
            assignments.extend(
                {"job_id": row["id"], "hotkey": key, "state": "ready"} for key in p["p_hotkeys"]
            )
            return None
        if name == "fez_claim_training":
            row = next(
                (
                    a
                    for a in assignments
                    if a["hotkey"] == p["p_hotkey"] and a["state"] in ("ready", "leased")
                ),
                None,
            )
            if row:
                row.update(
                    state="leased",
                    lease_token=row.get("lease_token") or str(uuid.uuid4()),
                    lease_until="2999-01-01T00:00:00Z",
                    uid=1,
                )
                next(j for j in jobs if j["id"] == row["job_id"])["status"] = "running"
            return copy.deepcopy(row)
        if name in ("fez_renew_training", "fez_submit_training", "fez_fail_training"):
            row = next(
                a for a in assignments if a["job_id"] == p["p_job"] and a["hotkey"] == p["p_hotkey"]
            )
            if row["lease_token"] != p["p_token"]:
                raise APIError(409, "Wrong lease.")
            if name == "fez_submit_training":
                row.update(state="submitted", sha256=p["p_sha256"])
            if name == "fez_fail_training" and row["state"] == "leased":
                row["state"] = "ready"
            return None
        raise AssertionError(name)

    def signed(self, bucket, path, *, upload=False):
        ticket = str(uuid.uuid4())
        self.tickets[ticket] = (bucket, path, upload)
        result = {"url": self.url + "/" + ticket}
        if upload:
            result.update(method="PUT", headers={"Content-Type": "application/octet-stream"})
        return result

    def exists(self, bucket, path):
        return (bucket, path) in self.objects

    def download(self, bucket, path, destination, limit):
        return cloud.download(self.signed(bucket, path)["url"], destination, limit)

    def upload(self, bucket, path, source):
        cloud.upload(self.signed(bucket, path, upload=True)["url"], source)

    def handler(self):
        store = self

        class Files(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def handle_file(self):
                ticket = store.tickets.get(self.path[1:])
                if not ticket or ticket[2] != (self.command == "PUT"):
                    self.send_error(403)
                    return
                bucket, path, put = ticket
                if put:
                    if (bucket, path) in store.objects:
                        self.send_error(409)
                        return
                    store.objects[bucket, path] = self.rfile.read(
                        int(self.headers["Content-Length"])
                    )
                body = b"{}" if put else store.objects.get((bucket, path))
                self.send_response(200 if body is not None else 404)
                self.send_header("Content-Length", str(len(body or b"")))
                self.end_headers()
                self.wfile.write(body or b"")

            do_GET = handle_file
            do_PUT = handle_file

        return Files


class QueueTest(unittest.TestCase):
    def test_supabase_storage_contract_and_temporary_failure(self):
        response = Mock(status_code=200, content=b"{}")
        response.json.return_value = {
            "url": "/object/upload/sign/fez-training-data/job/train.jsonl?token=fixture"
        }
        store = cloud.Supabase("https://project.supabase.co", "server-only-fixture")
        with patch("zils.cloud.requests.request", return_value=response) as request:
            result = store.signed(cloud.DATA_BUCKET, "job/train.jsonl", upload=True)
            self.assertEqual(result["method"], "PUT")
            self.assertEqual(result["headers"]["x-upsert"], "false")
            self.assertNotIn("Authorization", result["headers"])
            self.assertEqual(
                result["url"], "https://project.supabase.co/storage/v1" + response.json()["url"]
            )
            self.assertEqual(request.call_args.kwargs["headers"]["x-upsert"], "false")
        with patch(
            "zils.cloud.requests.request",
            side_effect=requests.ConnectionError("secret upstream URL"),
        ):
            with self.assertRaises(APIError) as failure:
                store.rows(coordinator.JOBS)
            self.assertNotIn("secret", str(failure.exception))

    def test_auth_replay_and_tenant_boundaries(self):
        store = Store()
        key = Keypair.create_from_seed("0x" + "12" * 32)
        store.tables["fez_training_workers"].append(
            {"hotkey": key.ss58_address, "uid": 1, "enabled": True}
        )
        service = coordinator.Service(store, "https://training.example.com")
        path = "/v1/workers/claim"
        message = queue_protocol.sign(key, service.audience, path, {})
        self.assertEqual(service.worker(path, message), {"assignment": None})
        with self.assertRaises(APIError):
            service.worker(path, message)
        for change in (
            {"audience": "https://elsewhere.example"},
            {"path": "/v1/workers/submit"},
            {"timestamp": int(time.time()) - 300},
            {"body": {"uid": 99}},
        ):
            invalid = copy.deepcopy(message)
            invalid["payload"].update(change)
            with self.subTest(change=change), self.assertRaises(APIError):
                service.worker(path, invalid)
        with server(coordinator.handler(service, "https://zils.example")) as url:
            self.assertEqual(requests.get(url + "/v1/jobs", timeout=5).status_code, 401)
            self.assertEqual(
                requests.get(
                    url + "/v1/jobs",
                    headers={
                        "Authorization": "Bearer owner-token",
                        "Origin": "https://attacker.example",
                    },
                    timeout=5,
                ).status_code,
                403,
            )
            job = store.rpc(
                "fez_create_training_job",
                {"p_owner": OWNER, "p_name": "owned", "p_acceptance": POLICY},
            )
            headers = {"Authorization": "Bearer other-token"}
            self.assertEqual(
                requests.get(url + "/v1/jobs/" + job["id"], headers=headers, timeout=5).status_code,
                404,
            )
            self.assertEqual(
                requests.get(url + "/v1/jobs", headers=headers, timeout=5).json(),
                {"jobs": [], "models": []},
            )
            body = {"name": "test", "acceptance": POLICY, "allow_training_data_export": False}
            self.assertEqual(
                requests.post(url + "/v1/jobs", json=body, headers=headers, timeout=5).status_code,
                400,
            )

    def test_upload_claim_train_submit_evaluate_download(self):
        from zils import models

        self.queued_model_flow(models.KEV)

    def test_jevk5_upload_train_calibrate_and_download(self):
        from zils import models

        with patch("zils.jevk5.validate_inputs"):
            self.queued_model_flow(models.JEVK5)

    def queued_model_flow(self, model):
        import torch
        from kev.checkpoint import Meta, write_meta

        from zils import models

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            reference.mkdir()
            (reference / "adapter_config.json").write_text("{}")
            (reference / "adapter_model.safetensors").write_text("baseline")
            write_meta(
                reference, Meta(base=zils.BASE, head={"fixture": torch.zeros(1)}, temperature=1.0)
            )
            if model == models.JEVK5:
                (reference / "head.pt").unlink()
                models.write_metadata(reference, kind="base", temperature=1.22)
            worker = root / "fixture-python"
            import sys

            worker.write_text(
                f"#!{sys.executable}\n"
                + """import json, sys, os
assert 'SUPABASE_SERVICE_ROLE_KEY' not in os.environ
assert 'SUPABASE_DB_URL' not in os.environ
from pathlib import Path
import torch
from kev.checkpoint import Meta, read_meta, write_meta
from zils import models
if '-m' in sys.argv:
    out = Path(sys.argv[sys.argv.index('--out')+1]); out.mkdir()
    data = Path(sys.argv[sys.argv.index('--data')+1])
    assert all(set(json.loads(line))=={'state','questions'} for line in data.read_text().splitlines())
    (out/'adapter_config.json').write_text('{}')
    (out/'adapter_model.safetensors').write_text('trained')
    if 'zils.jevk5' in sys.argv:
        models.write_metadata(out)
    else:
        write_meta(out, Meta(base='Qwen/Qwen3.5-0.8B-Base',head={'fixture':torch.ones(1)},temperature=1.0))
else:
    path = Path(sys.argv[sys.argv.index('--checkpoint')+1]); temperature=models.temperature(path)
    p=0.9 if (path/'adapter_model.safetensors').exists() and (path/'adapter_model.safetensors').read_text()=='trained' else 0.5
    a,b=p**(1/temperature),(1-p)**(1/temperature)
    rows=json.load(sys.stdin)
    assert all(set(r)=={'id','state','question'} for r in rows)
    print(json.dumps({'runtime':{'temperature':temperature},'predictions':[
        {'id':r['id'],'elapsed_ms':1,'probabilities':{'true':a/(a+b),'false':b/(a+b)}} for r in rows]}))
"""
            )
            worker.chmod(0o700)
            store = Store()
            key = Keypair.create_from_seed("0x" + "13" * 32)
            store.tables["fez_training_workers"].append(
                {"hotkey": key.ss58_address, "uid": 1, "enabled": True}
            )
            service = coordinator.Service(store, "http://127.0.0.1:8910", model)
            with (
                server(store.handler()) as storage_url,
                server(coordinator.handler(service, "https://zils.example")) as url,
            ):
                store.url = storage_url
                service.audience = url
                headers = {"Authorization": "Bearer owner-token"}
                response = requests.post(
                    url + "/v1/jobs",
                    headers=headers,
                    json={
                        "name": "support-v1",
                        "acceptance": POLICY,
                        "allow_training_data_export": True,
                    },
                    timeout=5,
                )
                self.assertEqual(response.status_code, 200, response.text)
                created = response.json()
                job_id = created["job"]["id"]
                job_url = url + "/v1/jobs/" + job_id
                self.assertEqual(
                    requests.post(
                        job_url + "/submit", headers=headers, json={}, timeout=5
                    ).status_code,
                    409,
                )
                for split, cases in examples().items():
                    path = root / (split + ".jsonl")
                    path.write_text("".join(json.dumps(c) + "\n" for c in cases))
                    cloud.upload(created["uploads"][split]["url"], path)
                resumed = requests.post(
                    job_url + "/uploads", headers=headers, json={}, timeout=5
                ).json()
                self.assertTrue(all(u["uploaded"] for u in resumed["uploads"].values()))
                self.assertEqual(
                    requests.post(job_url + "/submit", headers=headers, json={}, timeout=5).json()[
                        "job"
                    ]["status"],
                    "validating",
                )
                engine = coordinator.Processor(
                    service,
                    root / "processor",
                    reference,
                    SimpleNamespace(runtime_python=str(worker), device="cpu"),
                )
                self.assertTrue(engine.tick())
                self.assertEqual(service.job(job_id)["status"], "awaiting_approval")
                self.assertEqual(service.job(job_id)["manifest"]["model"], models.spec(model))
                self.assertEqual(
                    requests.get(url + "/v1/config", timeout=5).json()["model"], models.spec(model)
                )
                client = Client(url, key)
                self.assertIsNone(
                    client.call("claim")["assignment"], "no training data before operator approval"
                )
                store.rpc(
                    "fez_approve_training_job", {"p_job": job_id, "p_hotkeys": [key.ss58_address]}
                )
                assignment = client.call("claim")["assignment"]
                self.assertEqual(assignment["model"], models.spec(model))
                self.assertEqual(assignment["base_revision"], models.spec(model)["base_revision"])
                self.assertEqual(set(assignment["training"]), {"url"})
                self.assertNotIn("manifest", assignment)
                self.assertNotIn("acceptance", assignment)
                with patch.dict(
                    "os.environ",
                    {
                        "SUPABASE_SERVICE_ROLE_KEY": "must-stay-on-server",
                        "SUPABASE_DB_URL": "must-stay-on-server",
                    },
                ):
                    self.assertTrue(run_once(client, root / "miner", reference, str(worker), "cpu"))
                self.assertFalse(run_once(client, root / "miner", reference, str(worker), "cpu"))
                self.assertTrue(engine.tick())
                result = requests.get(job_url, headers=headers, timeout=5).json()["job"]
                self.assertEqual(result["status"], "completed", result)
                self.assertEqual(result["result"]["delivery"]["status"], "accepted")
                serialized = json.dumps(result)
                self.assertNotIn('"predictions"', serialized)
                self.assertNotIn('"checkpoint"', serialized)
                downloads = requests.get(job_url + "/downloads", headers=headers, timeout=5).json()[
                    "downloads"
                ]
                accepted = root / "accepted"
                accepted.mkdir()
                for name in models.candidate_files(model):
                    cloud.download(downloads[name]["url"], accepted / name, zils.MAX_ARTIFACT_BYTES)
                self.assertEqual(
                    zils.checkpoint_hash(accepted), result["result"]["delivery"]["sha256"]
                )
                self.assertEqual(
                    requests.get(
                        job_url + "/downloads",
                        headers={"Authorization": "Bearer other-token"},
                        timeout=5,
                    ).status_code,
                    404,
                )
                if model == models.JEVK5:
                    from zils.adapter_releases import publish, register, registry_entry

                    release = publish(store, job_id, root / "serving")
                    entry = registry_entry(release, "http://127.0.0.1:8921", "TOKEN")
                    catalog = root / "models.json"
                    register(catalog, entry, selection=release["selection"])
                    store.patch(
                        coordinator.JOBS,
                        f"id=eq.{job_id}",
                        {
                            "result": {
                                **service.job(job_id)["result"],
                                "workflow": {"state": "ready", "model_id": release["release_id"]},
                            }
                        },
                    )
                    upgrade_body = {
                        "name": "support-v2",
                        "acceptance": {**POLICY, "previous_job_id": job_id},
                        "allow_training_data_export": True,
                    }
                    foreign = requests.post(
                        url + "/v1/jobs",
                        headers={"Authorization": "Bearer other-token"},
                        json=upgrade_body,
                        timeout=5,
                    )
                    self.assertEqual(foreign.status_code, 400)
                    upgraded = requests.post(
                        url + "/v1/jobs", headers=headers, json=upgrade_body, timeout=5
                    )
                    self.assertEqual(upgraded.status_code, 200, upgraded.text)
                    created = upgraded.json()
                    upgrade_id = created["job"]["id"]
                    for split, cases in examples().items():
                        for case in cases:
                            case["id"] = "upgrade-" + case["id"]
                            case["group_id"] = "upgrade-" + case["group_id"]
                            case["state"]["ticket"] = "upgrade-" + case["state"]["ticket"]
                        path = root / ("upgrade-" + split + ".jsonl")
                        path.write_text("".join(json.dumps(c) + "\n" for c in cases))
                        cloud.upload(created["uploads"][split]["url"], path)
                    requests.post(
                        url + f"/v1/jobs/{upgrade_id}/submit", headers=headers, json={}, timeout=5
                    ).raise_for_status()
                    self.assertTrue(engine.tick())
                    self.assertEqual(
                        service.job(upgrade_id)["manifest"]["selection"]["previous"]["model_id"],
                        release["release_id"],
                    )
                    store.rpc(
                        "fez_approve_training_job",
                        {"p_job": upgrade_id, "p_hotkeys": [key.ss58_address]},
                    )
                    self.assertTrue(run_once(client, root / "miner", reference, str(worker), "cpu"))
                    before = catalog.read_bytes()
                    self.assertTrue(engine.tick())
                    upgraded = service.job(upgrade_id)
                    self.assertEqual(upgraded["status"], "completed", upgraded)
                    self.assertEqual(
                        upgraded["result"]["baseline_reference_sha256"],
                        release["checkpoint_sha256"],
                    )
                    # The fixture trains the same quality again: beating the base is insufficient for v2.
                    self.assertEqual(
                        upgraded["result"]["delivery"]["status"], "no_qualifying_model"
                    )
                    self.assertIsNone(publish(store, upgrade_id, root / "serving"))
                    self.assertEqual(catalog.read_bytes(), before)
