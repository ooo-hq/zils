"""Actual PostgreSQL profile intersections, concurrent claims and revoked leases."""

import uuid
from concurrent.futures import ThreadPoolExecutor

from tests.api_database import Database, literal
from zils import models
from zils.cloud import APIError


def run(command):
    db = Database(command)
    owner = str(uuid.uuid4())
    db.sql(f"insert into auth.users(id) values({literal(owner)})")
    jobs = {}
    for index, (worker, model) in enumerate(
        (("image-probe", models.IMAJEV), ("text-probe", models.JEVK5))
    ):
        db.sql(f"insert into fez_training_workers(hotkey,uid) values('{worker}',{61000 + index})")
        identity = models.profile_identity(model)
        evidence = {
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
            f"insert into zils_worker_profiles(hotkey,profile_id,profile_sha256,runtime_sha256,verified_by,min_free_mib,evidence) values('{worker}',{literal(model)},{literal(identity['profile_sha256'])},{literal(identity['runtime_sha256'])},'operator',12288,{literal(evidence)})"
        )
        jid = str(uuid.uuid4())
        jobs[model] = jid
        db.sql(
            f"insert into fez_training_jobs(id,owner_id,name,acceptance,status,model_profile) values({literal(jid)},{literal(owner)},'images','{{}}','awaiting_approval',{literal(models.spec(model))})"
        )
        db.rpc("fez_approve_training_job", {"p_job": jid, "p_hotkeys": "{" + worker + "}"})
        wrong = "text-probe" if index == 0 else "image-probe"
        # Even a mistakenly installed cross-profile assignment cannot be claimed.
        db.sql(
            f"insert into fez_training_assignments(job_id,hotkey,uid) select {literal(jid)},hotkey,uid from fez_training_workers where hotkey={literal(wrong)} on conflict do nothing"
        )

    for model, wrong in ((models.IMAJEV, "text-probe"), (models.JEVK5, "image-probe")):
        db.sql(
            f"insert into fez_training_assignments(job_id,hotkey,uid) select {literal(jobs[model])},hotkey,uid from fez_training_workers where hotkey={literal(wrong)} on conflict do nothing"
        )

    def claim(worker, model):
        return db.rpc(
            "zils_claim_profile_training",
            {"p_hotkey": worker, "p_supported_profiles": "{" + model + "}"},
        )

    with ThreadPoolExecutor(2) as pool:
        results = list(
            pool.map(
                lambda pair: claim(*pair),
                [("image-probe", models.IMAJEV), ("text-probe", models.JEVK5)],
            )
        )
    assert [x["job_id"] for x in results] == [jobs[models.IMAJEV], jobs[models.JEVK5]]
    # Same worker concurrent claims serialize to one lease and one attempt.
    with ThreadPoolExecutor(2) as pool:
        again = list(pool.map(lambda _: claim("image-probe", models.IMAJEV), range(2)))
    assert all(x["lease_token"] == results[0]["lease_token"] and x["attempts"] == 1 for x in again)
    assert db.rpc("fez_claim_training", {"p_hotkey": "text-probe"})["job_id"] == jobs[models.JEVK5]
    db.sql(
        f"update fez_training_assignments set state='failed' where job_id={literal(jobs[models.JEVK5])}"
    )
    assert db.rpc("fez_claim_training", {"p_hotkey": "image-probe"}) is None
    try:
        claim("text-probe", models.IMAJEV)
    except APIError:
        pass
    else:
        raise AssertionError("self-reported image capability authorized a worker")
    old = results[0]
    db.sql(
        f"update fez_training_assignments set lease_until=now()-interval '1 second' where job_id={literal(old['job_id'])} and hotkey='image-probe'"
    )
    refreshed = claim("image-probe", models.IMAJEV)
    assert refreshed["lease_token"] != old["lease_token"] and refreshed["attempts"] == 2
    auth = {"p_job": old["job_id"], "p_hotkey": "image-probe", "p_token": old["lease_token"]}
    for action, values in (
        ("fez_renew_training", auth),
        ("fez_submit_training", {**auth, "p_sha256": "d" * 64}),
    ):
        try:
            db.rpc(action, values)
        except APIError:
            pass
        else:
            raise AssertionError("expired lease reused")
    auth["p_token"] = refreshed["lease_token"]
    db.sql("update zils_worker_profiles set enabled=false where hotkey='image-probe'")
    try:
        db.rpc("fez_renew_training", auth)
    except APIError:
        pass
    else:
        raise AssertionError("revoked image worker renewed")
    db.sql("update zils_worker_profiles set enabled=true where hotkey='image-probe'")
    db.sql(f"update fez_training_jobs set status='failed' where id={literal(old['job_id'])}")
    try:
        db.rpc("fez_renew_training", auth)
    except APIError:
        pass
    else:
        raise AssertionError("cancelled image job renewed")
    for sql in (
        "set role authenticated; select * from zils_worker_profiles",
        "set role authenticated; select zils_claim_profile_training('image-probe',array['imajev-4b-v1'])",
    ):
        try:
            db.sql(sql)
        except APIError:
            pass
        else:
            raise AssertionError("customer accessed operator qualification")
