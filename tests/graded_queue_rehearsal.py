"""Signed HTTP, real SQL leases, fixture training and real validator/calibration flow.

This proves orchestration only. Fixture probabilities do not measure hardware or model quality.
"""

import secrets
import sys
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from bittensor_wallet import Keypair

import zils
from miner.queue import Client, run_once
from tests.api_database import Database, literal
from tests.test_jobs import POLICY, examples
from tests.test_queue import Store as Storage, server
from zils import benchmark, coordinator, jobs, miner_grading as grading, models
from zils.graded_scheduler import GradedScheduler, benchmark_binding

FIXTURE = """import json, os, sys
from pathlib import Path
from zils import models
assert 'SUPABASE_SERVICE_ROLE_KEY' not in os.environ
if '-m' in sys.argv:
    rows=Path(sys.argv[sys.argv.index('--data')+1]).read_text().splitlines()
    assert all(set(json.loads(r)) == {'state','questions'} for r in rows)
    out=Path(sys.argv[sys.argv.index('--out')+1]); out.mkdir()
    (out/'adapter_config.json').write_text('{}')
    (out/'adapter_model.safetensors').write_text('fixture-trained')
    models.write_metadata(out)
else:
    checkpoint=Path(sys.argv[sys.argv.index('--checkpoint')+1])
    temperature=models.temperature(checkpoint)
    p=0.9 if (checkpoint/'adapter_model.safetensors').exists() else 0.5
    a,b=p**(1/temperature),(1-p)**(1/temperature)
    rows=json.load(sys.stdin)
    assert all(set(r)=={'id','state','question'} for r in rows)
    print(json.dumps({'runtime':{'temperature':temperature}, 'predictions':[
        {'id':r['id'],'elapsed_ms':1,'probabilities':{'true':a/(a+b),'false':b/(a+b)}} for r in rows]}))
"""


class QueueStore(Storage):
    def __init__(self, db):
        super().__init__()
        self.db = db

    def rows(self, table, query=""):
        return self.db.rows(table, query)

    def patch(self, table, query, values):
        return self.db.patch(table, query, values)

    def rpc(self, name, values):
        values = dict(values)
        for key in ("p_supported_profiles", "p_hotkeys"):
            if isinstance(values.get(key), list):
                values[key] = "{" + ",".join(values[key]) + "}"
        if name == "zils_claim_profile_processing":
            values["p_profiles"] = "{" + ",".join(values["p_profiles"]) + "}"
        return self.db.rpc(name, values)


def run(command):
    db = Database(command)
    owner = str(uuid.uuid4())
    db.sql(f"insert into auth.users(id) values({literal(owner)})")
    keys = [Keypair.create_from_seed("0x" + secrets.token_hex(32)) for _ in range(2)]
    for index, key in enumerate(keys):
        db.sql(
            f"insert into fez_training_workers(hotkey,uid,resource_id) values({literal(key.ss58_address)},{63100 + index},{literal(str(uuid.uuid4()))})"
        )
    with tempfile.TemporaryDirectory(prefix="zils-graded-http-") as tmp:
        root = Path(tmp)
        reference = root / "reference"
        reference.mkdir()
        models.write_metadata(reference, kind="base")
        runner = root / "fixture-python"
        runner.write_text(f"#!{sys.executable}\n" + FIXTURE)
        runner.chmod(0o700)
        store = QueueStore(db)
        service = coordinator.Service(store, "http://127.0.0.1:8910", models.JEVK5)
        with (
            server(store.handler()) as storage_url,
            server(coordinator.handler(service, "https://fixture.example")) as url,
        ):
            service.audience, store.url = url, storage_url
            clients = [Client(url, key) for key in keys]
            engine = coordinator.Processor(
                service,
                root / "processor",
                reference,
                SimpleNamespace(runtime_python=str(runner), device="cpu"),
            )

            def create_job():
                jid = str(uuid.uuid4())
                folder = root / jid
                manifest = jobs.build(
                    folder,
                    jid,
                    examples(),
                    POLICY,
                    allow_training_data_export=True,
                    model=models.JEVK5,
                    workload=grading.workload_profile([100, 100], model=models.JEVK5),
                )
                sha = benchmark.file_hash(folder / "manifest.json")
                initial = zils.checkpoint_hash(reference)
                db.sql(
                    f"insert into fez_training_jobs(id,owner_id,name,status,acceptance,manifest,job_sha256,initial_sha256) values({literal(jid)},{literal(owner)},'signed-rehearsal','awaiting_approval',{literal(POLICY)},{literal(manifest)},{literal(sha)},{literal(initial)})"
                )
                job = service.job(jid)
                for name in (*benchmark.FILES, "manifest.json"):
                    store.upload(
                        coordinator.DATA_BUCKET, coordinator.prepared_path(job, name), folder / name
                    )
                return job

            first = create_job()
            context = {
                "model": models.JEVK5,
                **models.profile_identity(models.JEVK5),
                "trainer_sha256": grading.trainer_identity(),
                "rubric": "local-fixture/v1",
                "benchmark_sha256": benchmark_binding(first),
                "band": "tokens-512",
            }
            config = {
                "mode": "graded",
                "policy_version": grading.POLICY,
                "pool": [k.ss58_address for k in keys],
                "contexts": {"tokens-512": context},
                "qualification_slots": 1,
            }
            db.rpc("zils_configure_routing", {"p_config": config})
            scheduler = GradedScheduler(store, config)
            descriptor = {
                "context": context,
                "benchmark_sha256": context["benchmark_sha256"],
                "min_accuracy": 0.8,
                "quality_floor": 0.0,
            }

            def heartbeat():
                payload = {
                    "profiles": [
                        {
                            "model": models.JEVK5,
                            **models.profile_identity(models.JEVK5),
                            "trainer_sha256": context["trainer_sha256"],
                            "ready": True,
                        }
                    ]
                }
                for client in clients:
                    client.call("heartbeat", payload)

            for index, key in enumerate(keys):
                for number in range(1 if index == 0 else 3):
                    now = datetime.now(UTC)
                    certificate = {
                        "context": context,
                        "id": str(uuid.uuid4()),
                        "kind": "capacity" if index == 0 else "quality",
                        "artifact_valid": True,
                        "verified_at": now.isoformat(),
                        "expires_at": (now + timedelta(days=7)).isoformat(),
                        "capacity": {"examples": 100, "total_tokens": 20000, "max_tokens": 512},
                        "seconds_per_token": 0.5,
                        "baseline_brier": 0.5,
                        "candidate_brier": 0.45,
                        "uniform_brier": 0.5,
                        "accuracy": 0.9,
                        "min_accuracy": 0.8,
                        "quality_floor": 0.0,
                    }
                    db.rpc(
                        "zils_import_qualification",
                        {
                            "p_hotkey": key.ss58_address,
                            "p_report": certificate,
                            "p_verified_by": "local-fixture",
                            "p_evidence_sha256": grading.digest(certificate),
                        },
                    )

            def evaluate(job):
                token = str(uuid.uuid4())
                db.sql(
                    f"update fez_training_jobs set status='evaluating',lease_token={literal(token)},lease_until=now()+interval '20 minutes' where id={literal(job['id'])}"
                )
                current = service.job(job["id"])
                work = root / ("evaluate-" + token)
                work.mkdir()
                values = engine.evaluate(current, work)
                db.rpc(
                    "fez_finish_processing",
                    {
                        "p_job": job["id"],
                        "p_token": token,
                        "p_status": "completed",
                        "p_values": values,
                    },
                )
                return service.job(job["id"])

            for number in range(3):
                heartbeat()
                job = first if number == 0 else create_job()
                db.rpc(
                    "zils_authorize_qualification", {"p_job": job["id"], "p_benchmark": descriptor}
                )
                assigned = scheduler.tick_qualification(datetime.now(UTC))
                assert (
                    assigned["status"] == "reserved" and assigned["hotkey"] == keys[0].ss58_address
                ), assigned
                assert run_once(clients[0], root / "miner-one", reference, str(runner), "cpu")
                result = evaluate(job)
                assert (
                    result["release_prefix"] is None
                    and result["result"]["delivery"]["status"] == "qualification_complete"
                )
            heartbeat()
            customer = create_job()
            assert scheduler.assign(customer, datetime.now(UTC))["hotkey"] == keys[0].ss58_address
            assignment = clients[0].call("claim")["assignment"]
            clients[0].call(
                "defer", {"job_id": customer["id"], "lease_token": assignment["lease_token"]}
            )
            assert scheduler.assign(customer, datetime.now(UTC))["hotkey"] == keys[1].ss58_address
            assert run_once(clients[1], root / "miner-two", reference, str(runner), "cpu")
            result = evaluate(customer)
            assert result["status"] == "completed"
            attempts = db.rows("zils_training_attempts", f"job_id=eq.{customer['id']}")
            assert sorted(a["outcome"] for a in attempts) == ["capacity_deferred", "valid"], (
                attempts
            )
            audits = db.rows("zils_assignment_decisions", f"job_id=eq.{customer['id']}")
            assert len(audits) == 2
            for audit in audits:
                source = audit["input_snapshot"]
                replay = grading.rank_candidates(
                    source["job"],
                    source["workers"],
                    grading.timestamp(audit["decision"]["evaluated_at"]),
                )
                assert replay["snapshot_sha256"] == audit["decision"]["snapshot_sha256"]
            print(
                "Signed HTTP + PostgreSQL rehearsal: three qualification rounds, customer choice, second-host deferral fallback, validated completion; fixture artifacts only."
            )
