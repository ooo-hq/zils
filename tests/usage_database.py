"""Account usage reporting against real PostgreSQL, without provider services."""

import uuid

from tests.api_database import Database, literal
from zils.api_store import Store


def run(command):
    db = Database(command)
    owner, other = str(uuid.uuid4()), str(uuid.uuid4())
    db.sql(f"insert into auth.users values ({literal(owner)}),({literal(other)})")
    db.sql("update zils_billing_settings set mode='test'")
    store = Store(db)
    key = store.create_key(owner, "reporting")["id"]
    store.create_key(other, "isolated")
    db.sql(f"""insert into zils_billing_accounts(owner_id,mode,balance_nanos,free_training_runs,bonus_eligible)
      values({literal(owner)},'test',10000000000,1,true)""")

    def call(model, status="completed", tokens=10):
        rid = str(uuid.uuid4())
        store.admit(owner, key, rid, 100, tokens, model, "Support" if model else None)
        if status != "started":
            store.finish_usage(rid, 80 if status == "completed" else None, status)
        return rid

    # More records than the 100-row credit activity UI; totals must not be truncated.
    for _ in range(105):
        call("support-v1")
    call(None, tokens=20)  # Existing workers remain compatible and history is explicit.
    call("support-v1", "failed")
    call("support-v1", "started")
    old = call("old-model", tokens=30)
    db.sql(
        f"update zils_api_usage set created_at=now()-interval '31 days' where request_id={literal(old)}"
    )
    db.sql(
        f"update zils_billing_ledger set created_at=now()-interval '31 days' where reference={literal(old)}"
    )

    for status in ("completed", "completed", "failed", "validating"):
        job = db.rpc(
            "fez_create_training_job",
            {"p_owner": owner, "p_name": "report-test", "p_acceptance": {}},
        )
        db.patch("fez_training_jobs", "id=eq." + job["id"], {"status": "validating"})
        if status != "validating":
            db.patch("fez_training_jobs", "id=eq." + job["id"], {"status": status})
    report = db.rpc("zils_billing_summary", {"p_owner": owner, "p_mode": "test"})
    usage = report["usage"]
    assert len(report["transactions"]) == 100
    assert usage["calls"] == "106", usage
    assert usage["input_tokens"] == "1070", usage  # Logical tokens, not runtime token counts.
    assert usage["failed_calls"] == usage["active_calls"] == "1"
    assert usage["training_runs"] == "2"  # Includes the free completed run.
    assert usage["failed_training_runs"] == usage["active_training_runs"] == "1"
    assert usage["inference_spend_nanos"] == str(1070 * 42)
    assert usage["training_spend_nanos"] == "2000000000"
    assert {row["model_id"] for row in usage["models"]} == {"support-v1", None}
    for field in ("calls", "failed_calls", "active_calls", "input_tokens"):
        assert sum(int(row[field]) for row in usage["models"]) == int(usage[field])
    assert sum(int(row["spend_nanos"]) for row in usage["models"]) == int(
        usage["inference_spend_nanos"]
    )
    empty = db.rpc("zils_billing_summary", {"p_owner": other, "p_mode": "test"})["usage"]
    assert empty["calls"] == empty["input_tokens"] == empty["training_runs"] == "0"
    assert empty["models"] == []
    # Switching modes must not expose test usage as live usage.
    db.sql("update zils_billing_settings set mode='live'")
    assert (
        db.rpc("zils_billing_usage_summary", {"p_owner": owner, "p_mode": "live"})["calls"] == "0"
    )
    db.sql("update zils_billing_settings set mode='test'")
    for role in ("anon", "authenticated"):
        assert not db.sql(
            f"select to_jsonb(has_function_privilege('{role}','zils_billing_usage_summary(uuid,text)','EXECUTE'))"
        )
    # A purged raw record must not erase a recently posted charge from account spend.
    db.sql(
        f"delete from zils_api_usage where owner_id={literal(owner)} and model_id='support-v1' and status='completed'"
    )
    after = db.rpc("zils_billing_usage_summary", {"p_owner": owner, "p_mode": "test"})
    assert after["inference_spend_nanos"] == usage["inference_spend_nanos"]
    assert sum(int(row["spend_nanos"]) for row in after["models"]) == int(
        after["inference_spend_nanos"]
    )
    print(
        "Usage reporting: exact totals, legacy models, date bounds, free training, mode/owner isolation and retention passed."
    )
