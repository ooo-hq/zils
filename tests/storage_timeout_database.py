"""A candidate transport deadline cannot settle a job as a quality failure."""

import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.api_database import Database, literal
from zils import benchmark, cloud, coordinator, models, validator
from zils.runtime import digest


def run(command):
    db = Database(command)
    owner, job_id, lease = (str(uuid.uuid4()) for _ in range(3))
    db.sql(f"insert into auth.users(id) values({literal(owner)})")
    db.sql(
        f"insert into fez_training_jobs(id,owner_id,name,acceptance,status,lease_token,lease_until,model_profile) values({literal(job_id)},{literal(owner)},'timeout-test','{{}}','evaluating',{literal(lease)},now()+interval '20 minutes',{literal(models.spec(models.KEV))})"
    )
    before = db.rows("zils_billing_accounts", "order=owner_id.asc")
    rpc, subprocess_run = db.rpc, subprocess.run

    def claim(name, values):
        if name == "zils_claim_profile_processing":
            return (
                db.rows("fez_training_jobs", f"id=eq.{job_id}")[0]
                if values["p_stage"] == "evaluating"
                else None
            )
        return rpc(name, values)

    def timeout(args, *positional, **kwargs):
        if "--download" in args:
            raise subprocess.TimeoutExpired(args, 600)
        return subprocess_run(args, *positional, **kwargs)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        processor = coordinator.Processor.__new__(coordinator.Processor)
        processor.root, processor.store = root, db
        processor.references = {models.KEV: root / "unused-reference"}
        processor.args = SimpleNamespace(device="cpu", runtime_python="unused")

        def evaluate(job, work):
            data = work / "benchmark"
            data.mkdir()
            (data / "manifest.json").write_text(json.dumps({}))
            config = {
                "benchmark_sha256": digest(data / "manifest.json"),
                "base_revision": models.spec(models.KEV)["base_revision"],
            }
            registry = {1: {"claim": {"endpoint": "storage"}}}

            def fetch(claim, destination, **kwargs):
                cloud.download("https://storage.example/candidate", destination, 512 * 1024**2)

            return validator.evaluate_round(
                config, work, work, registry, processor.args, fetch_checkpoint=fetch
            )

        processor.evaluate = evaluate
        with (
            patch.object(db, "rpc", side_effect=claim),
            patch.object(cloud.subprocess, "run", side_effect=timeout),
            patch.object(benchmark, "audit"),
            patch.object(
                validator,
                "run_child",
                side_effect=AssertionError("Storage outage reached quality evaluation"),
            ),
        ):
            assert processor.tick()
        job = db.rows("fez_training_jobs", f"id=eq.{job_id}")[0]
        assert job["status"] == "evaluating" and job["result"] is None, job["status"]
        assert db.rows("zils_billing_accounts", "order=owner_id.asc") == before
        # Dispose only this fixture's job after demonstrating retriable state.
        db.sql(f"delete from fez_training_jobs where id={literal(job_id)}")
    print("Storage download timeout: candidate evaluation deferred; job and billing preserved.")
