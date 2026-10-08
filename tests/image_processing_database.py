"""Processing capacity intersections in actual PostgreSQL."""

import uuid

from tests.api_database import Database, literal
from zils import models
from zils.cloud import APIError


def run(command):
    db = Database(command)
    owner = str(uuid.uuid4())
    db.sql(f"insert into auth.users(id) values({literal(owner)})")
    # Isolate from earlier fixtures without altering their evidence.
    db.sql(
        "update fez_training_jobs set status='failed' where status in ('validating','evaluating','queued','running')"
    )
    jobs = {}
    for index, model in enumerate((models.IMAJEV, models.JEVK5)):
        jid = str(uuid.uuid4())
        jobs[model] = jid
        db.sql(
            f"insert into fez_training_jobs(id,owner_id,name,status,acceptance,model_profile,created_at) values({literal(jid)},{literal(owner)},'capacity','evaluating','{{}}',{literal(models.spec(model))},now()-interval '{2 - index} day')"
        )
    args = {"p_stage": "evaluating", "p_profiles": "{" + models.JEVK5 + "}"}
    picked = db.rpc("zils_claim_profile_processing", args)
    assert picked["id"] == jobs[models.JEVK5], picked
    assert db.rpc("zils_claim_profile_processing", args) is None
    assert db.rpc("fez_claim_processing", {"p_stage": "evaluating"}) is None
    image = db.rpc(
        "zils_claim_profile_processing",
        {"p_stage": "evaluating", "p_profiles": "{" + models.IMAJEV + "}"},
    )
    assert image["id"] == jobs[models.IMAJEV], image
    assert (
        db.rpc("zils_claim_profile_processing", {"p_stage": "evaluating", "p_profiles": "{}"})
        is None
    )
    try:
        db.sql(
            "set role authenticated; select zils_claim_profile_processing('evaluating',array['imajev-4b-v1'])"
        )
    except APIError:
        pass
    else:
        raise AssertionError("Customer could claim processing")
