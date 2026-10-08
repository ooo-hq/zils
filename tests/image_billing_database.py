"""Production-schema upgrade: image work uses the existing auth and prepaid ledger."""

import copy
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from tests.api_database import Database, literal
from tests.test_image_api import OWNER, Assets
from tests.test_image_contract import BODY
from zils import models
from zils.api import Gateway, Registry
from zils.api_store import Store
from zils.cloud import APIError
from zils.decisions import DecisionError
from zils.image_contract import IMAGE_CAPABILITIES


def run(command):
    db = Database(command)
    owner = OWNER
    db.sql(f"insert into auth.users(id) values({literal(owner)}) on conflict do nothing")
    db.sql("update zils_billing_settings set mode='test'")
    # No invitation table or RPC is required by the current production sign-in flow.
    assert db.sql("select to_jsonb(to_regclass('public.zils_access_applications'))") is None
    assert db.sql("select to_jsonb(count(*)) from pg_proc where proname='zils_api_admit'") == 1
    db.user = lambda token: owner
    store = Store(db)
    assert store.session_owner("supabase-session") == owner
    store.ensure_account(owner)
    db.sql(
        f"update zils_api_accounts set requests_per_second=null,tokens_per_second=null where owner_id={literal(owner)}"
    )
    key = store.create_key(owner, "image billing")
    assert store.authenticate(key["key"])["owner_id"] == owner
    purchase = str(uuid.uuid4())
    db.rpc(
        "zils_billing_checkout",
        {"p_owner": owner, "p_mode": "test", "p_purchase": purchase, "p_amount": 500},
    )
    db.rpc(
        "zils_billing_attach_checkout",
        {
            "p_purchase": purchase,
            "p_mode": "test",
            "p_session": "cs_" + purchase,
            "p_url": "https://checkout.stripe.com/c/pay/test",
        },
    )
    db.rpc(
        "zils_billing_fulfill",
        {
            "p_purchase": purchase,
            "p_mode": "test",
            "p_session": "cs_" + purchase,
            "p_payment": "pi_" + purchase,
            "p_amount": 500,
            "p_event": "evt_" + purchase,
        },
    )

    def summary():
        return db.rpc("zils_billing_summary", {"p_owner": owner, "p_mode": "test"})

    initial_usage = summary()["usage"]
    entry = {
        "id": "image-release",
        "fingerprint": "a" * 64,
        "aliases": ["images"],
        "owners": None,
        "url": "http://127.0.0.1:8923",
        "token_env": "IMAGE_BILLING_FIXTURE",
        "release_date": "2026-10-08",
        "description": "Billing fixture",
        "capabilities": IMAGE_CAPABILITIES,
    }
    gateway = Gateway(store, Registry([entry]), image_store=Assets())
    predictions = {
        "inspection": {
            "probabilities": {"normal": 0.8, "damaged": 0.1, "__unknown__": 0.1},
            "input_tokens": 442,
        }
    }
    with (
        patch.dict(os.environ, {"IMAGE_BILLING_FIXTURE": "fixture", "ZILS_IMAGES_ENABLED": "1"}),
        patch("zils.api.RuntimeClient.prepare", return_value=(442, 173)),
        patch("zils.api.RuntimeClient.predict", return_value=predictions) as predict,
    ):
        for token, path in (("session", "/v1/image-decisions"), (key["key"], "/v1/systemone")):
            rid = str(uuid.uuid4())
            status, response = gateway.dispatch("POST", path, token, copy.deepcopy(BODY), rid)
            assert status == 200 and response["usage"]["billable_input_tokens"] == 173
            store.finish_usage(rid, 442, "completed")  # replay cannot debit twice
        assert summary()["balance_nanos"] == str(5_000_000_000 - 2 * 173 * 42)
        balance = summary()["balance_nanos"]
        predict.side_effect = DecisionError(503, "runtime_unavailable", "Fixture")
        try:
            gateway.dispatch(
                "POST", "/v1/image-decisions", "session", copy.deepcopy(BODY), str(uuid.uuid4())
            )
        except DecisionError:
            pass
        else:
            raise AssertionError("Failed image execution accepted")
        assert summary()["balance_nanos"] == balance
        assert summary()["reserved_nanos"] == "0"
        image_usage = next(
            row for row in summary()["usage"]["models"] if row["model_id"] == entry["id"]
        )
        assert image_usage["model_name"] == "images"
        assert image_usage["calls"] == "2" and image_usage["failed_calls"] == "1"
        assert image_usage["active_calls"] == "0"
        assert image_usage["input_tokens"] == str(2 * 173)
        assert image_usage["spend_nanos"] == str(2 * 173 * 42)

    def draft():
        job = db.rpc(
            "zils_create_profile_job",
            {
                "p_owner": owner,
                "p_name": "image-credit-test",
                "p_acceptance": {"min_accuracy": 0.8, "min_brier_improvement": 0.01},
                "p_model": models.spec(models.IMAJEV),
            },
        )
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
        lease = db.rpc("zils_image_claim_finalize", {"p_owner": owner, "p_asset": asset["id"]})
        db.rpc(
            "zils_image_finish",
            {
                "p_owner": owner,
                "p_asset": asset["id"],
                "p_token": lease["finalize_token"],
                "p_sha256": "a" * 64,
                "p_pixels": "b" * 64,
                "p_bytes": 100,
                "p_width": 8,
                "p_height": 8,
            },
        )
        return {"p_owner": owner, "p_job": job["id"], "p_assets": [asset["id"]]}

    first = draft()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: db.rpc("zils_submit_image_job", first), range(4)))
    assert all(job["status"] == "validating" for job in results)
    assert summary()["free_training_runs"] == 0
    assert (
        db.sql(
            f"select to_jsonb(count(*)) from zils_billing_reservations where id={literal(first['p_job'])}"
        )
        == 1
    )
    db.sql(f"update fez_training_jobs set status='completed' where id={literal(first['p_job'])}")
    assert summary()["balance_nanos"] == balance  # included run

    second = draft()
    db.rpc("zils_submit_image_job", second)
    assert summary()["reserved_nanos"] == "2000000000"
    db.sql(
        f"update fez_training_jobs set status='failed',error='Invalid dataset' where id={literal(second['p_job'])}"
    )
    assert summary()["reserved_nanos"] == "0" and summary()["balance_nanos"] == balance
    third = draft()
    db.rpc("zils_submit_image_job", third)
    db.sql(f"update fez_training_jobs set status='completed' where id={literal(third['p_job'])}")
    assert summary()["balance_nanos"] == str(int(balance) - 2_000_000_000)
    usage = summary()["usage"]
    assert int(usage["training_runs"]) == int(initial_usage["training_runs"]) + 2
    assert int(usage["failed_training_runs"]) == int(initial_usage["failed_training_runs"]) + 1
    assert (
        int(usage["training_spend_nanos"])
        == int(initial_usage["training_spend_nanos"]) + 2_000_000_000
    )

    blocked = draft()
    db.sql(
        f"update zils_billing_accounts set balance_nanos=0 where owner_id={literal(owner)} and mode='test'"
    )
    try:
        db.rpc("zils_submit_image_job", blocked)
    except APIError:
        pass
    else:
        raise AssertionError("Unfunded image run submitted")
    assert (
        db.sql(
            f"select to_jsonb(status) from fez_training_jobs where id={literal(blocked['p_job'])}"
        )
        == "uploading"
    )
    assert (
        db.sql(
            f"select to_jsonb(referenced) from zils_image_assets where id={literal(blocked['p_assets'][0])}"
        )
        is False
    )
    assert summary()["reserved_nanos"] == "0"
    db.sql("update zils_billing_settings set mode='off'")
    print(
        "Image billing: existing sessions/keys, single charge, failure release, included/paid runs, concurrent submit and insufficient-credit rollback passed."
    )
