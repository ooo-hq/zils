"""Exercise prepaid money against real PostgreSQL, including competing requests."""

import json
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from tests.api_database import Database, literal
from tests.test_api import BODY
from zils.api_store import Store
from zils.batches import Worker


def run(command):
    db = Database(command)
    assert db.sql("select to_jsonb(to_regclass('public.zils_billing_accounts') is not null)"), (
        "Prepaid billing ledger is missing"
    )
    owner, other = str(uuid.uuid4()), str(uuid.uuid4())
    db.sql(f"insert into auth.users values ({literal(owner)}),({literal(other)})")
    db.sql("update zils_billing_settings set mode='test'")

    def rpc(name, **args):
        return db.rpc(name, args)

    def summary(who=owner, mode="test"):
        return rpc("zils_billing_summary", p_owner=who, p_mode=mode)

    def purchase(who=owner):
        pid = str(uuid.uuid4())
        rpc("zils_billing_checkout", p_owner=who, p_mode="test", p_purchase=pid, p_amount=500)
        rpc(
            "zils_billing_attach_checkout",
            p_purchase=pid,
            p_mode="test",
            p_session="cs_" + pid,
            p_url="https://checkout.stripe.com/c/pay/test",
        )
        return pid

    def pay(pid, **kwargs):
        return rpc(
            "zils_billing_fulfill",
            p_purchase=pid,
            p_mode="test",
            p_session="cs_" + pid,
            p_payment="pi_" + pid,
            p_amount=500,
            p_event="evt_" + str(uuid.uuid4()),
            **kwargs,
        )

    assert summary()["available_nanos"] == "0"
    pid = purchase()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: pay(pid), range(8)))
    s = summary()
    assert s["balance_nanos"] == "5000000000", s
    assert s["free_training_runs"] == 1
    assert len(s["transactions"]) == 1
    assert summary(other)["balance_nanos"] == "0"
    # Atomic admission across separate API keys: only one $4.20 request fits.
    key_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    for kid in key_ids:
        rpc(
            "zils_api_create_key",
            p_key={
                "id": kid,
                "owner_id": owner,
                "name": "billing-test",
                "prefix": "zils_sk_",
                "digest": "a" * 64,
            },
        )
    requests = [str(uuid.uuid4()) for _ in range(8)]

    def admit(pair):
        i, rid = pair
        return rpc(
            "zils_api_admit",
            p_owner=owner,
            p_key=key_ids[i % 2],
            p_request=rid,
            p_tokens=100000000,
            p_billable_tokens=100000000,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        admissions = list(pool.map(admit, enumerate(requests)))
    assert admissions.count("allowed") == 1, admissions
    assert admissions.count("insufficient_credit") == 7, admissions
    rid = requests[admissions.index("allowed")]
    assert summary()["available_nanos"] == "800000000"
    rpc("zils_api_finish_usage", p_request=rid, p_tokens=100000000, p_status="completed")
    rpc("zils_api_finish_usage", p_request=rid, p_tokens=0, p_status="failed")
    assert summary()["balance_nanos"] == "800000000"
    assert summary()["reserved_nanos"] == "0"
    assert (
        rpc(
            "zils_api_admit",
            p_owner=owner,
            p_key=key_ids[0],
            p_request=str(uuid.uuid4()),
            p_tokens=10,
        )
        == "billing_meter_unavailable"
    )
    # A failed inference releases, rather than spends, its reservation.
    failed = str(uuid.uuid4())
    assert (
        rpc(
            "zils_api_admit",
            p_owner=owner,
            p_key=key_ids[0],
            p_request=failed,
            p_tokens=100,
            p_billable_tokens=100,
        )
        == "allowed"
    )
    rpc("zils_api_finish_usage", p_request=failed, p_tokens=None, p_status="failed")
    assert summary()["available_nanos"] == "800000000"
    # Original processing tokens do not inflate the price of logical input.
    logical = str(uuid.uuid4())
    assert (
        rpc(
            "zils_api_admit",
            p_owner=owner,
            p_key=key_ids[0],
            p_request=logical,
            p_tokens=10000,
            p_billable_tokens=1000,
        )
        == "allowed"
    )
    rpc("zils_api_finish_usage", p_request=logical, p_tokens=9999, p_status="completed")
    assert summary()["balance_nanos"] == "799958000"
    # Cancellation/expiry while the model runs rejects its result and releases credit.
    for terminal in ("cancelled", "expired"):
        batch = str(uuid.uuid4())
        db.sql(
            "insert into zils_api_batches(id,owner_id,idempotency_key,input_path,status,total) "
            f"values({literal(batch)},{literal(owner)},{literal(batch)},'fixture','running',1)"
        )
        db.sql(
            "insert into zils_api_batch_items(batch_id,line,custom_id,body,frozen) "
            f"values({literal(batch)},1,'fixture',{literal(json.dumps(BODY))},'{{}}')"
        )
        store = Store(db)
        request_ids = []

        def evaluate(who, key, body, request_id, **kwargs):
            assert who == owner and kwargs["lane"] == "bulk"
            store.admit(who, key, request_id, 100, 100)
            request_ids.append(request_id)
            assert summary()["reserved_nanos"] == "4200"
            if terminal == "expired":
                db.sql(
                    "update zils_api_batches set deadline=now()-interval '1 second' where id="
                    + literal(batch)
                )
            stopped = rpc("zils_api_batch_action", p_owner=owner, p_id=batch, p_action="cancel")
            assert stopped["status"] == terminal
            return {"usage": {"input_tokens": 10}}

        assert Worker(db, SimpleNamespace(store=store, evaluate=evaluate)).once()
        assert len(request_ids) == 1
        assert summary()["reserved_nanos"] == "0", terminal
        assert summary()["balance_nanos"] == "799958000", terminal
        assert db.rows("zils_api_usage", "request_id=eq." + request_ids[0])[0]["status"] == "failed"
    # Full refunds are cumulative, idempotent and can leave debt after consumption.
    for amount in (250, 500, 250, 500):
        rpc(
            "zils_billing_refund",
            p_mode="test",
            p_payment="pi_" + pid,
            p_refunded=amount,
            p_event="evt_" + str(uuid.uuid4()),
        )
    assert summary()["balance_nanos"] == "-4200042000"
    assert summary()["free_training_runs"] == 0
    # Refund snapshot received before fulfillment never exposes spendable credit.
    reversed_pid = purchase(other)
    pay(reversed_pid, p_refunded=500)
    assert summary(other)["available_nanos"] == "0"
    assert summary(other)["free_training_runs"] == 0
    # Training grants one bonus, then reserves $2; failed infrastructure restores it.
    trainee = str(uuid.uuid4())
    db.sql(f"insert into auth.users values ({literal(trainee)})")
    train_payment = purchase(trainee)
    pay(train_payment)

    def job():
        item = rpc(
            "fez_create_training_job", p_owner=trainee, p_name="billing-run", p_acceptance={}
        )
        return item["id"]

    def state(jid, status, error=None):
        return db.patch("fez_training_jobs", "id=eq." + jid, {"status": status, "error": error})

    first = job()
    state(first, "validating")
    assert summary(trainee)["free_training_runs"] == 0
    assert summary(trainee)["available_nanos"] == "5000000000"
    state(first, "failed", "Infrastructure failure.")
    assert summary(trainee)["free_training_runs"] == 1
    second = job()
    state(second, "validating")
    state(second, "completed")
    assert summary(trainee)["free_training_runs"] == 0
    third = job()
    state(third, "validating")
    assert summary(trainee)["available_nanos"] == "3000000000"
    state(third, "failed", "Infrastructure failure.")
    assert summary(trainee)["available_nanos"] == "5000000000"
    fourth = job()
    state(fourth, "validating")
    state(fourth, "evaluating")
    state(fourth, "failed", "Cancelled by customer.")
    assert summary(trainee)["balance_nanos"] == "3000000000"
    # Completed evaluation is charged even if no qualifying model was delivered.
    fifth = job()
    state(fifth, "validating")
    state(fifth, "completed")
    assert summary(trainee)["balance_nanos"] == "1000000000"
    sixth = job()
    try:
        state(sixth, "validating")
        raise AssertionError("training accepted without sufficient credit")
    except Exception as error:
        assert "Insufficient prepaid credit" in str(error), error
    assert db.rows("fez_training_jobs", "id=eq." + sixth)[0]["status"] == "uploading"
    # Leasing/defer is not execution: cancellation must restore the reservation.
    pay(purchase(trainee))
    deferred = job()
    state(deferred, "validating")
    state(deferred, "running")
    db.sql(
        "insert into fez_training_workers(hotkey,uid) values ('billing-deferred-worker',250) on conflict do nothing"
    )
    db.sql(
        f"insert into fez_training_assignments(job_id,hotkey,uid,state,attempts) values ({literal(deferred)},'billing-deferred-worker',250,'ready',0)"
    )
    before = summary(trainee)["balance_nanos"]
    state(deferred, "failed", "Cancelled by customer.")
    assert summary(trainee)["balance_nanos"] == before
    # A claim that selected a queued job before cancellation must not revive it.
    racing = job()
    state(racing, "validating")
    db.patch(
        "fez_training_jobs",
        "id=eq." + racing,
        {"status": "queued", "deadline": "2099-01-01T00:00:00Z"},
    )
    db.sql(
        f"insert into fez_training_assignments(job_id,hotkey,uid,state,attempts) values ({literal(racing)},'billing-deferred-worker',250,'ready',0)"
    )
    lock = subprocess.Popen(
        [*command, "-qAt"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        lock.stdin.write(
            f"begin; select id from fez_training_jobs where id={literal(racing)} for update;\n"
        )
        lock.stdin.flush()
        assert lock.stdout.readline().strip() == racing
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(rpc, "fez_claim_training", p_hotkey="billing-deferred-worker")
            deadline = time.monotonic() + 5
            while not db.sql(
                "select to_jsonb(exists(select 1 from pg_stat_activity where wait_event_type='Lock' and query like 'select to_jsonb(fez_claim_training%'))"
            ):
                if pending.done():
                    # Profile-aware queues skip jobs already locked by cancellation.
                    # Earlier queues wait for the lock and reject the later update.
                    assert pending.result() is None, "worker claimed a cancellation-locked job"
                    break
                assert time.monotonic() < deadline, "claim did not reach the locked job"
                time.sleep(0.01)
            lock.stdin.write(
                f"update fez_training_jobs set status='failed',error='Cancelled by customer.' where id={literal(racing)}; commit; select 'cancelled';\n"
            )
            lock.stdin.flush()
            assert lock.stdout.readline().strip() == "cancelled"
            try:
                assert pending.result(timeout=5) is None, "worker revived a cancelled paid job"
            except Exception as error:
                assert "job no longer accepts training" in str(error), error
        assert db.rows("fez_training_assignments", "job_id=eq." + racing)[0]["attempts"] == 0
        assert db.rows("fez_training_jobs", "id=eq." + racing)[0]["status"] == "failed"
        assert db.rows("zils_billing_reservations", "id=eq." + racing)[0]["status"] == "released"
    finally:
        lock.stdin.close()
        lock.terminate()
        lock.wait(timeout=5)
    # An unresolved dispute blocks all keys, and an older event cannot clear it.
    extra = purchase()
    pay(extra)
    rpc(
        "zils_billing_dispute",
        p_mode="test",
        p_payment="pi_" + extra,
        p_disputed=True,
        p_event="evt_dispute_open",
        p_event_created=200,
    )
    rpc(
        "zils_billing_dispute",
        p_mode="test",
        p_payment="pi_" + extra,
        p_disputed=False,
        p_event="evt_dispute_old",
        p_event_created=100,
    )
    rpc(
        "zils_billing_dispute",
        p_mode="test",
        p_payment="pi_" + extra,
        p_disputed=False,
        p_event="evt_dispute_equal",
        p_event_created=200,
    )
    assert (
        rpc(
            "zils_api_admit",
            p_owner=owner,
            p_key=key_ids[0],
            p_request=str(uuid.uuid4()),
            p_tokens=100,
            p_billable_tokens=100,
        )
        == "insufficient_credit"
    )
    rpc(
        "zils_billing_dispute",
        p_mode="test",
        p_payment="pi_" + extra,
        p_disputed=False,
        p_event="evt_dispute_won",
        p_event_created=300,
    )
    # Recover a successful Stripe payment if saving its Checkout response failed.
    unattached = str(uuid.uuid4())
    rpc("zils_billing_checkout", p_owner=other, p_mode="test", p_purchase=unattached, p_amount=500)
    pay(unattached)
    assert summary(other)["balance_nanos"] == "5000000000"
    # Service-only tables and RPCs cannot be accessed by a browser session.
    for role in ("anon", "authenticated"):
        for statement in (
            "select * from zils_billing_accounts",
            "update zils_billing_settings set mode='live'",
            f"select zils_billing_summary({literal(owner)},'test')",
            f"select zils_billing_settle({literal(rid)},false)",
        ):
            try:
                db.sql(f"set role {role}; {statement}")
                raise AssertionError("unprivileged billing access succeeded")
            except Exception as error:
                assert "permission denied" in str(error), error
    # Test funds cannot become live funds.
    db.sql("update zils_billing_settings set mode='live'")
    assert summary(owner, "live")["balance_nanos"] == "0"
    db.sql("update zils_billing_settings set mode='test'")
    print(
        "Prepaid ledger: concurrent admission, exact charges, training, disputes, refunds and isolation passed."
    )
