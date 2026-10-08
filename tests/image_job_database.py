"""Frozen job profiles and service-only creation in real PostgreSQL."""

import uuid

from tests.api_database import Database, literal
from zils import models
from zils.cloud import APIError


def run(command):
    db = Database(command)
    owner = str(uuid.uuid4())
    db.sql(
        f"insert into auth.users(id,email,email_confirmed_at) values({literal(owner)},{literal(owner + '@example.com')},now())"
    )
    db.rpc(
        "zils_access_invite",
        {"p_email": owner + "@example.com", "p_action": "approve", "p_actor": owner},
    )
    db.rpc("zils_access_claim", {"p_owner": owner})
    args = {
        "p_owner": owner,
        "p_name": "images",
        "p_acceptance": {"min_accuracy": 0.8, "min_brier_improvement": 0.01},
        "p_model": models.spec(models.IMAJEV),
    }
    job = db.rpc("zils_create_profile_job", args)
    assert job["model_profile"] == models.spec(models.IMAJEV)
    assert job["manifest"] is None
    for operation in (
        f"update fez_training_jobs set model_profile={literal(models.spec(models.JEVK5))} where id={literal(job['id'])}",
        f"update fez_training_jobs set model_profile=null where id={literal(job['id'])}",
        f"set role authenticated; select zils_create_profile_job({literal(owner)},'forbidden','{{}}', {literal(models.spec(models.IMAJEV))})",
    ):
        try:
            db.sql(operation)
        except APIError:
            pass
        else:
            raise AssertionError("job profile changed or customer bypassed coordinator")
    try:
        db.rpc("zils_create_profile_job", {**args, "p_model": {**args["p_model"], "max_pixels": 1}})
    except APIError:
        pass
    else:
        raise AssertionError("unregistered model profile accepted")
    # A stale draft must not revive expired image assets by entering validating.
    db.sql(
        f"update fez_training_jobs set created_at=now()-interval '25 hours' where id={literal(job['id'])}"
    )
    try:
        db.rpc(
            "zils_submit_image_job",
            {"p_owner": owner, "p_job": job["id"], "p_assets": [str(uuid.uuid4())]},
        )
    except APIError:
        pass
    else:
        raise AssertionError("expired image draft submitted")
    assert (
        db.sql(f"select to_jsonb(status) from fez_training_jobs where id={literal(job['id'])}")
        == "uploading"
    )
    db.sql(f"update fez_training_jobs set created_at=now() where id={literal(job['id'])}")
    asset = db.rpc(
        "zils_image_create",
        {
            "p_owner": owner,
            "p_purpose": "training",
            "p_job": job["id"],
            "p_filename": "photo.png",
            "p_source_bytes": 100,
            "p_source_sha256": "a" * 64,
        },
    )
    aid = asset["id"]
    try:
        db.rpc("zils_submit_image_job", {"p_owner": owner, "p_job": job["id"], "p_assets": [aid]})
    except APIError:
        pass
    else:
        raise AssertionError("incomplete image submitted")
    lease = db.rpc("zils_image_claim_finalize", {"p_owner": owner, "p_asset": aid})
    db.rpc(
        "zils_image_finish",
        {
            "p_owner": owner,
            "p_asset": aid,
            "p_token": lease["finalize_token"],
            "p_sha256": "a" * 64,
            "p_pixels": "b" * 64,
            "p_bytes": 100,
            "p_width": 8,
            "p_height": 8,
        },
    )
    submitted = db.rpc(
        "zils_submit_image_job", {"p_owner": owner, "p_job": job["id"], "p_assets": [aid]}
    )
    assert submitted["status"] == "validating"
    assert (
        db.sql(f"select to_jsonb(referenced) from zils_image_assets where id={literal(aid)}")
        is True
    )
    legacy = db.rpc("fez_create_training_job", {k: v for k, v in args.items() if k != "p_model"})
    assert legacy["model_profile"] is None
