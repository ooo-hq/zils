"""Real PostgreSQL grading reservations, fencing, evidence and client isolation."""

import copy
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from tests.api_database import Database, literal
from tests.test_miner_grading import CONTEXT, report
from zils import miner_grading as grading, models
from zils.cloud import APIError


def denied(call):
    try:
        call()
    except APIError:
        return
    raise AssertionError("operation should have been rejected")


def run(command):
    db = Database(command)
    owner, resource = str(uuid.uuid4()), str(uuid.uuid4())
    db.sql(f"insert into auth.users(id) values({literal(owner)})")
    for index, key in enumerate(("graded-a", "graded-b")):
        db.sql(
            f"insert into fez_training_workers(hotkey,uid,resource_id) values({literal(key)},{63000 + index},{literal(resource)})"
        )
    config = {
        "mode": "graded",
        "policy_version": grading.POLICY,
        "pool": ["graded-a", "graded-b"],
        "contexts": {"tokens-512": CONTEXT},
        "qualification_slots": 1,
    }
    db.sql(f"insert into zils_routing_policy(id,config) values(true,{literal(config)})")
    now = datetime.now(UTC)
    for key in config["pool"]:
        for i in range(3):
            evidence = report(str(uuid.uuid4()))
            evidence["verified_at"] = (now - timedelta(minutes=1)).isoformat()
            evidence["expires_at"] = (now + timedelta(days=7)).isoformat()
            db.rpc(
                "zils_import_qualification",
                {
                    "p_hotkey": key,
                    "p_report": evidence,
                    "p_verified_by": "local-fixture",
                    "p_evidence_sha256": grading.digest([key, i]),
                },
            )
        db.rpc("zils_worker_heartbeat", {"p_hotkey": key, "p_profiles": presence()})

    def new_job():
        jid = str(uuid.uuid4())
        manifest = {
            "model": models.spec(models.JEVK5),
            "data_access": "approved-workers-training-export",
            "workload": grading.workload_profile([100] * 10, model=models.JEVK5),
        }
        db.sql(
            f"insert into fez_training_jobs(id,owner_id,name,status,acceptance,manifest,job_sha256,initial_sha256) values({literal(jid)},{literal(owner)},'grading','awaiting_approval','{{}}',{literal(manifest)},{literal(grading.digest(jid))},{literal('e' * 64)})"
        )
        return jid

    def decision(jid, preferred=None):
        state = db.rpc("zils_grading_snapshot", {"p_job": jid})
        job = state["job"]
        result = grading.rank_candidates(job, state["workers"], datetime.now(UTC))
        if preferred:
            result["selected_hotkey"] = preferred
        result["state_sha256"] = state["state_sha256"]
        return result

    def reserve(jid, chosen=None):
        d = decision(jid, chosen)
        return db.rpc(
            "zils_reserve_graded_job",
            {"p_job": jid, "p_hotkey": d["selected_hotkey"], "p_decision": d},
        )

    jobs = [new_job(), new_job()]
    ds = [decision(jobs[0], "graded-a"), decision(jobs[1], "graded-b")]
    with ThreadPoolExecutor(2) as pool:
        results = list(
            pool.map(
                lambda pair: db.rpc(
                    "zils_reserve_graded_job",
                    {
                        "p_job": pair[0],
                        "p_hotkey": pair[1]["selected_hotkey"],
                        "p_decision": pair[1],
                    },
                ),
                zip(jobs, ds),
            )
        )
    assert sorted(x["status"] for x in results) == ["reserved", "retry"], results
    rows = db.rows("zils_worker_reservations")
    assert len(rows) == 1
    active, key = rows[0]["job_id"], rows[0]["hotkey"]
    original_deadline = db.rows("fez_training_jobs", f"id=eq.{active}")[0]["deadline"]
    claim = db.rpc(
        "zils_claim_profile_training",
        {"p_hotkey": key, "p_supported_profiles": "{" + models.JEVK5 + "}"},
    )
    assert claim["job_id"] == active
    assert len(db.rows("zils_training_attempts")) == 1
    db.rpc(
        "zils_release_graded_attempt",
        {
            "p_job": active,
            "p_hotkey": key,
            "p_token": claim["lease_token"],
            "p_outcome": "capacity_deferred",
        },
    )
    assert not db.rows("zils_worker_reservations")
    assert db.rows("fez_training_jobs", f"id=eq.{active}")[0]["deadline"] == original_deadline
    assert db.rows("fez_training_assignments", f"job_id=eq.{active}")[0]["attempts"] == 0
    assert reserve(active)["status"] == "retry", "resource cooldown bypassed"
    denied(
        lambda: db.rpc(
            "fez_submit_training",
            {
                "p_job": active,
                "p_hotkey": key,
                "p_token": claim["lease_token"],
                "p_sha256": "f" * 64,
            },
        )
    )
    db.sql("update zils_resource_cooldowns set until_at=now()-interval '1 second'")
    other = "graded-b" if key == "graded-a" else "graded-a"
    assert reserve(active, other)["status"] == "reserved"
    claim2 = db.rpc(
        "zils_claim_profile_training",
        {"p_hotkey": other, "p_supported_profiles": "{" + models.JEVK5 + "}"},
    )
    assert claim2["lease_token"] != claim["lease_token"]
    # Fixed/manual approvals cannot evade the same physical slot.
    denied(
        lambda: db.rpc(
            "fez_approve_training_job", {"p_job": new_job(), "p_hotkeys": "{" + key + "}"}
        )
    )
    # Cancellation and a competing reservation serialize before any row lock.
    waiting = new_job()
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(
            db.patch,
            "fez_training_jobs",
            f"id=eq.{active}",
            {"status": "failed", "error": "Cancelled by customer."},
        )
        b = pool.submit(reserve, waiting, key)
        a.result()
        b.result()
    assert len(db.rows("zils_worker_reservations")) <= 1
    denied(
        lambda: db.rpc(
            "fez_renew_training",
            {"p_job": active, "p_hotkey": other, "p_token": claim2["lease_token"]},
        )
    )
    # Clear fixture work to avoid influencing the existing billing regressions.
    db.sql(
        f"update fez_training_jobs set status='failed',error='Cancelled by customer.' where owner_id={literal(owner)}"
    )
    # Two schedulers competing for the same job commit one assignment.
    same = new_job()
    dsame = decision(same)
    with ThreadPoolExecutor(2) as pool:
        race = list(
            pool.map(
                lambda _: db.rpc(
                    "zils_reserve_graded_job",
                    {"p_job": same, "p_hotkey": dsame["selected_hotkey"], "p_decision": dsame},
                ),
                range(2),
            )
        )
    assert sorted(x["status"] for x in race) == ["reserved", "retry"]
    winner = dsame["selected_hotkey"]
    live = db.rpc(
        "zils_claim_profile_training",
        {"p_hotkey": winner, "p_supported_profiles": "{" + models.JEVK5 + "}"},
    )
    db.rpc(
        "fez_submit_training",
        {"p_job": same, "p_hotkey": winner, "p_token": live["lease_token"], "p_sha256": "d" * 64},
    )
    assert not db.rows("zils_worker_reservations")
    processor_token = str(uuid.uuid4())
    db.sql(
        f"update fez_training_jobs set status='evaluating',lease_token={literal(processor_token)},lease_until=now()+interval '20 minutes' where id={literal(same)}"
    )
    obs = {
        "hotkey": winner,
        "lease_token": live["lease_token"],
        "sha256": "d" * 64,
        "job_sha256": db.rows("fez_training_jobs", f"id=eq.{same}")[0]["job_sha256"],
        "outcome": "valid",
    }
    args = {"p_job": same, "p_processor_token": processor_token, "p_observations": [obs]}
    db.rpc("zils_record_graded_evaluation", args)
    db.rpc("zils_record_graded_evaluation", args)
    denied(
        lambda: db.rpc(
            "zils_record_graded_evaluation",
            {**args, "p_observations": [{**obs, "sha256": "0" * 64}]},
        )
    )
    denied(
        lambda: db.rpc(
            "zils_record_graded_evaluation", {**args, "p_processor_token": str(uuid.uuid4())}
        )
    )
    db.rpc(
        "fez_finish_processing",
        {
            "p_job": same,
            "p_token": processor_token,
            "p_status": "completed",
            "p_values": {"result": {"delivery": {"status": "no_qualifying_model"}}},
        },
    )
    attempts = db.rows("zils_training_attempts", f"job_id=eq.{same}")
    assert len(attempts) == 1 and attempts[0]["outcome"] == "valid"
    # Terminal evidence cannot be edited, even accidentally by trusted callers.
    denied(
        lambda: db.patch(
            "zils_training_attempts", f"job_id=eq.{same}", {"outcome": "worker_failure"}
        )
    )
    expired = new_job()
    assert reserve(expired)["status"] == "reserved"
    c = db.rpc(
        "zils_claim_profile_training",
        {"p_hotkey": winner, "p_supported_profiles": "{" + models.JEVK5 + "}"},
    )
    # Ranking can select the other identity after last-assignment tie-breaking.
    if c is None:
        winner = "graded-b" if winner == "graded-a" else "graded-a"
        c = db.rpc(
            "zils_claim_profile_training",
            {"p_hotkey": winner, "p_supported_profiles": "{" + models.JEVK5 + "}"},
        )
    db.sql(
        f"update zils_worker_reservations set expires_at=now()-interval '1 second' where job_id={literal(expired)}"
    )
    with ThreadPoolExecutor(2) as pool:
        reap = pool.submit(db.rpc, "zils_reap_graded", {})
        renew = pool.submit(
            denied,
            lambda: db.rpc(
                "fez_renew_training",
                {"p_job": expired, "p_hotkey": winner, "p_token": c["lease_token"]},
            ),
        )
        reap.result()
        renew.result()
    assert db.rows("fez_training_jobs", f"id=eq.{expired}")[0]["status"] == "awaiting_approval"
    assert db.rows("zils_training_attempts", f"job_id=eq.{expired}")[0]["outcome"] == "abandoned"
    denied(
        lambda: db.rpc(
            "fez_submit_training",
            {
                "p_job": expired,
                "p_hotkey": winner,
                "p_token": c["lease_token"],
                "p_sha256": "d" * 64,
            },
        )
    )
    db.sql(
        f"update fez_training_jobs set status='failed',error='Cancelled by customer.' where owner_id={literal(owner)} and status<>'completed'"
    )
    for table in (
        "zils_worker_qualifications",
        "zils_worker_presence",
        "zils_training_attempts",
        "zils_worker_reservations",
        "zils_assignment_decisions",
        "zils_job_scheduling",
        "zils_routing_policy",
    ):
        for role in ("anon", "authenticated"):
            denied(
                lambda table=table, role=role: db.sql(
                    f"set role {role}; select to_jsonb(t) from {table} t"
                )
            )
    denied(
        lambda: db.sql(
            "set role authenticated; select to_jsonb(zils_grading_snapshot(gen_random_uuid()))"
        )
    )
    forged = copy.deepcopy(presence())
    forged[0]["quality_band"] = 999
    denied(lambda: db.rpc("zils_worker_heartbeat", {"p_hotkey": key, "p_profiles": forged}))
    qualification_rounds(db)
    fixed_lease_retries(db)
    qualification_skips_blocked_job(db)
    print(
        "Graded routing: shared-resource races, lease fencing, cooldown, cancellation and RLS passed."
    )


def presence():
    return [
        {
            "model": models.JEVK5,
            **models.profile_identity(models.JEVK5),
            "trainer_sha256": CONTEXT["trainer_sha256"],
            "ready": True,
        }
    ]


def fixed_lease_retries(db):
    """Mapped fixed workers retain exclusive capacity through allowed lease retries."""
    owner, resource = str(uuid.uuid4()), str(uuid.uuid4())
    key, alias = "fixed-retry", "fixed-retry-alias"
    db.sql(f"insert into auth.users(id) values({literal(owner)})")
    for index, hotkey in enumerate((key, alias)):
        db.sql(
            f"insert into fez_training_workers(hotkey,uid,resource_id) values({literal(hotkey)},{63020 + index},{literal(resource)})"
        )
    identity = models.profile_identity(models.IMAJEV)
    image_evidence = {
        "examples": 4,
        "max_pixels": 400000,
        "max_input_tokens": 4096,
        "decoder_verified": True,
        "reload_verified": True,
        "finite_gradients": True,
        "peak_gpu_reserved_bytes": 11 * 1024**3,
        "probe_sha256": "a" * 64,
        "trainer_sha256": "b" * 64,
        "max_seconds": 1200,
    }
    db.sql(
        f"insert into zils_worker_profiles(hotkey,profile_id,profile_sha256,runtime_sha256,verified_by,min_free_mib,evidence) values({literal(key)},{literal(models.IMAJEV)},{literal(identity['profile_sha256'])},{literal(identity['runtime_sha256'])},'local-fixture',12288,{literal(image_evidence)})"
    )
    for model in (models.KEV, models.JEVK5, models.IMAJEV):
        jid, competing = str(uuid.uuid4()), str(uuid.uuid4())
        for job_id in (jid, competing):
            db.sql(
                f"insert into fez_training_jobs(id,owner_id,name,status,acceptance,manifest) values({literal(job_id)},{literal(owner)},'fixed-retry','awaiting_approval','{{}}',{literal({'model': models.spec(model)})})"
            )
        db.rpc("fez_approve_training_job", {"p_job": jid, "p_hotkeys": "{" + key + "}"})
        deadline = db.rows("fez_training_jobs", f"id=eq.{jid}")[0]["deadline"]

        def claim():
            if model == models.KEV:
                return db.rpc("fez_claim_training", {"p_hotkey": key})
            return db.rpc(
                "zils_claim_profile_training",
                {"p_hotkey": key, "p_supported_profiles": "{" + model + "}"},
            )

        first = claim()
        assert first["attempts"] == 1
        db.sql(
            f"update fez_training_assignments set lease_until=now()-interval '1 second' where job_id={literal(jid)}"
        )
        denied(
            lambda: db.rpc(
                "fez_renew_training",
                {"p_job": jid, "p_hotkey": key, "p_token": first["lease_token"]},
            )
        )
        denied(
            lambda: db.rpc(
                "fez_approve_training_job", {"p_job": competing, "p_hotkeys": "{" + alias + "}"}
            )
        )
        second = claim()
        assert second is not None, "fixed worker must reclaim an expired lease with retries left"
        assert second["attempts"] == 2 and second["lease_token"] != first["lease_token"]
        reservation = db.rows("zils_worker_reservations", f"job_id=eq.{jid}")[0]
        assert reservation["expires_at"] == deadline
        denied(
            lambda: db.rpc(
                "fez_submit_training",
                {
                    "p_job": jid,
                    "p_hotkey": key,
                    "p_token": first["lease_token"],
                    "p_sha256": "f" * 64,
                },
            )
        )
        db.rpc(
            "fez_submit_training",
            {"p_job": jid, "p_hotkey": key, "p_token": second["lease_token"], "p_sha256": "f" * 64},
        )
        assert not db.rows("zils_worker_reservations", f"job_id=eq.{jid}")
        db.sql(
            f"update fez_training_jobs set status='failed',error='Cancelled by customer.' where owner_id={literal(owner)}"
        )
    print("Fixed mapped workers: expired leases retry without losing resource exclusivity.")


def qualification_skips_blocked_job(db):
    """A blocked older workload cannot starve a ready provisional miner."""
    from zils.graded_scheduler import GradedScheduler, benchmark_binding

    owner, resource, key = str(uuid.uuid4()), str(uuid.uuid4()), "qualification-later"
    db.sql(f"insert into auth.users(id) values({literal(owner)})")
    db.sql(
        f"insert into fez_training_workers(hotkey,uid,resource_id) values({literal(key)},63022,{literal(resource)})"
    )
    contexts, jobs = {}, []
    for index, length in enumerate((100, 600, 600)):
        manifest = {
            "model": models.spec(models.JEVK5),
            "data_access": "approved-workers-training-export",
            "files": {"calibration.jsonl": "a" * 64},
            "counts": {"train": 1, "test": 1, "calibration": 1},
            "workload": grading.workload_profile([length], model=models.JEVK5),
        }
        binding = benchmark_binding({"manifest": manifest, "initial_sha256": "e" * 64})
        context = {**CONTEXT, "band": manifest["workload"]["band"], "benchmark_sha256": binding}
        contexts[context["band"]] = context
        jid = str(uuid.uuid4())
        db.sql(
            f"insert into fez_training_jobs(id,owner_id,name,status,acceptance,manifest,job_sha256,initial_sha256,created_at) values({literal(jid)},{literal(owner)},'qualification-order','awaiting_approval','{{}}',{literal(manifest)},{literal(grading.digest(jid))},{literal('e' * 64)},now()-interval '{3 - index} minutes')"
        )
        jobs.append((jid, context))
    config = {
        "mode": "graded",
        "policy_version": grading.POLICY,
        "pool": [key],
        "contexts": contexts,
        "qualification_slots": 1,
    }
    db.rpc("zils_configure_routing", {"p_config": config})
    for jid, context in jobs:
        db.rpc(
            "zils_authorize_qualification",
            {
                "p_job": jid,
                "p_benchmark": {
                    "context": context,
                    "benchmark_sha256": context["benchmark_sha256"],
                    "min_accuracy": 0.8,
                    "quality_floor": 0.0,
                },
            },
        )
    now = datetime.now(UTC)
    capacity = {
        **report("later-capacity"),
        "kind": "capacity",
        "context": jobs[1][1],
        "capacity": {"examples": 100, "total_tokens": 20000, "max_tokens": 1024},
        "verified_at": now.isoformat(),
        "expires_at": (now + timedelta(days=7)).isoformat(),
    }
    db.rpc(
        "zils_import_qualification",
        {
            "p_hotkey": key,
            "p_report": capacity,
            "p_verified_by": "local-fixture",
            "p_evidence_sha256": grading.digest(capacity),
        },
    )
    db.rpc("zils_worker_heartbeat", {"p_hotkey": key, "p_profiles": presence()})
    scheduler = GradedScheduler(db, config)
    now = datetime.now(UTC)
    assert scheduler.preview({"id": jobs[0][0]}, now)["selected_hotkey"] is None
    assert scheduler.preview({"id": jobs[1][0]}, now)["selected_hotkey"] == key
    assert scheduler.tick_qualification(now)["status"] == "reserved", (
        "blocked oldest qualification must not starve eligible work"
    )
    rows = db.rows("zils_worker_reservations")
    assert len(rows) == 1 and rows[0]["job_id"] == jobs[1][0]
    assert scheduler.tick_qualification(now)["status"] == "waiting"
    assert len(db.rows("zils_worker_reservations")) == 1
    db.sql(
        f"update fez_training_jobs set status='failed',error='Cancelled by customer.' where owner_id={literal(owner)}"
    )
    print("Qualification scheduling: skips blocked oldest job and retains one active slot.")


def qualification_rounds(db):
    """A provisional resource earns three validator reports before customer work."""
    from zils.graded_scheduler import GradedScheduler, benchmark_binding

    owner = str(uuid.uuid4())
    db.sql(f"insert into auth.users(id) values({literal(owner)})")
    key = "qualification-worker"
    resource = str(uuid.uuid4())
    db.sql(
        f"insert into fez_training_workers(hotkey,uid,resource_id) values({literal(key)},63010,{literal(resource)})"
    )
    manifest = {
        "model": models.spec(models.JEVK5),
        "data_access": "approved-workers-training-export",
        "files": {
            "miner-training.jsonl": "f" * 64,
            "train.jsonl": "e" * 64,
            "calibration.jsonl": "d" * 64,
            "test.jsonl": "c" * 64,
        },
        "counts": {"train": 10, "calibration": 10, "test": 10},
        "workload": grading.workload_profile([100] * 10, model=models.JEVK5),
    }
    binding = benchmark_binding({"manifest": manifest, "initial_sha256": "e" * 64})
    context = {**CONTEXT, "benchmark_sha256": binding}
    config = {
        "mode": "graded",
        "policy_version": grading.POLICY,
        "pool": [key],
        "contexts": {"tokens-512": context},
        "qualification_slots": 1,
    }
    db.rpc("zils_configure_routing", {"p_config": config})
    descriptor = {
        "context": context,
        "benchmark_sha256": binding,
        "min_accuracy": 0.8,
        "quality_floor": 0.0,
    }
    now = datetime.now(UTC)
    capacity = {
        **report("capacity"),
        "context": context,
        "kind": "capacity",
        "verified_at": now.isoformat(),
        "expires_at": (now + timedelta(days=7)).isoformat(),
    }
    db.rpc(
        "zils_import_qualification",
        {
            "p_hotkey": key,
            "p_report": capacity,
            "p_verified_by": "local-probe",
            "p_evidence_sha256": grading.digest(capacity),
        },
    )
    db.rpc("zils_worker_heartbeat", {"p_hotkey": key, "p_profiles": presence()})
    scheduler = GradedScheduler(db, config)

    def fresh_job():
        jid = str(uuid.uuid4())
        db.sql(
            f"insert into fez_training_jobs(id,owner_id,name,status,acceptance,manifest,job_sha256,initial_sha256) values({literal(jid)},{literal(owner)},'qualification','awaiting_approval','{{}}',{literal(manifest)},{literal(grading.digest([manifest, jid]))},{literal('e' * 64)})"
        )
        return db.rows("fez_training_jobs", f"id=eq.{jid}")[0]

    customer = fresh_job()
    assert scheduler.preview(customer, datetime.now(UTC))["selected_hotkey"] is None
    for index in range(3):
        job = fresh_job()
        db.rpc("zils_authorize_qualification", {"p_job": job["id"], "p_benchmark": descriptor})
        assert scheduler.tick_qualification(datetime.now(UTC))["status"] == "reserved"
        claim = db.rpc(
            "zils_claim_profile_training",
            {"p_hotkey": key, "p_supported_profiles": "{" + models.JEVK5 + "}"},
        )
        sha = grading.digest(index)
        db.rpc(
            "fez_submit_training",
            {"p_job": job["id"], "p_hotkey": key, "p_token": claim["lease_token"], "p_sha256": sha},
        )
        token = str(uuid.uuid4())
        db.sql(
            f"update fez_training_jobs set status='evaluating',lease_token={literal(token)},lease_until=now()+interval '20 minutes' where id={literal(job['id'])}"
        )
        observation = {
            "hotkey": key,
            "lease_token": claim["lease_token"],
            "job_sha256": job["job_sha256"],
            "sha256": sha,
            "outcome": "valid",
            "qualification": {
                "baseline_brier": 0.5,
                "candidate_brier": 0.38,
                "uniform_brier": 1.0,
                "accuracy": 0.9,
                "cases": 10,
                "calibration_sha256": "d" * 64,
                "evaluated_sha256": "a" * 64,
            },
        }
        db.rpc(
            "fez_finish_processing",
            {
                "p_job": job["id"],
                "p_token": token,
                "p_status": "completed",
                "p_values": {
                    "grading_observations": [observation],
                    "result": {"delivery": {"status": "accepted"}},
                    "release_prefix": "must-not-publish",
                },
            },
        )
        finished = db.rows("fez_training_jobs", f"id=eq.{job['id']}")[0]
        assert finished["result"]["delivery"]["status"] == "qualification_complete"
        assert finished["release_prefix"] is None
        result = scheduler.preview(customer, datetime.now(UTC))
        assert (result["selected_hotkey"] == key) == (index == 2), result
    assert scheduler.assign(customer, datetime.now(UTC))["status"] == "reserved"
    db.sql(
        f"update fez_training_jobs set status='failed',error='Cancelled by customer.' where owner_id={literal(owner)} and status<>'completed'"
    )
    expired = fresh_job()
    db.sql(
        f"update zils_job_scheduling set deadline=now()-interval '1 second' where job_id={literal(expired['id'])}"
    )
    db.rpc("zils_reap_graded", {})
    assert db.rows("fez_training_jobs", f"id=eq.{expired['id']}")[0]["status"] == "failed"
    print(
        "Qualification lane: capacity-only worker earns three reports, then receives customer fixture; no qualification release activates."
    )
