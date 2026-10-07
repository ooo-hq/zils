"""Exercise real admission SQL, including concurrent reservations and old keys."""

import uuid
from concurrent.futures import ThreadPoolExecutor

from tests.api_database import Database, literal
from zils.cloud import APIError


def run(command):
    db = Database(command)
    actor, owner, stranger = (str(uuid.uuid4()) for _ in range(3))
    db.sql(
        "alter table auth.users add column if not exists email text, add column if not exists email_confirmed_at timestamptz"
    )
    for uid, email in [
        (actor, "admin@example.com"),
        (owner, "member@example.com"),
        (stranger, "stranger@example.com"),
    ]:
        db.sql(
            f"insert into auth.users(id,email,email_confirmed_at) values({literal(uid)},{literal(email)},now())"
        )
    assert db.rpc("zils_access_allowed", {"p_owner": owner}) is False
    assert (
        db.rpc(
            "zils_access_apply",
            {
                "p_email": " MEMBER@EXAMPLE.COM ",
                "p_use_case": "Route support decisions",
                "p_source": "a" * 64,
            },
        )
        == "accepted"
    )
    db.rpc(
        "zils_access_apply",
        {"p_email": "member@example.com", "p_use_case": "Do not overwrite", "p_source": "a" * 64},
    )
    view = db.rpc("zils_access_overview", {"p_status": "all", "p_offset": 0})
    assert view["capacity"] == 25 and view["allocated"] == 0 and view["waiting"] == 1
    assert view["applications"][0]["use_case"] == "Route support decisions"
    result = db.rpc(
        "zils_access_invite",
        {"p_email": "member@example.com", "p_action": "approve", "p_actor": actor},
    )
    assert result["send"] is True and result["application"]["status"] == "invited"
    retry = db.rpc(
        "zils_access_invite",
        {"p_email": "member@example.com", "p_action": "approve", "p_actor": actor},
    )
    assert retry["send"] is False
    assert db.rpc("zils_access_claim", {"p_owner": stranger})["status"] == "waiting"
    assert db.rpc("zils_access_claim", {"p_owner": owner})["status"] == "active"
    assert db.rpc("zils_access_allowed", {"p_owner": owner}) is True
    assert db.rpc("zils_access_overview", {"p_status": "all", "p_offset": 0})["allocated"] == 1
    # A new session with the same email cannot take an already-bound account.
    impostor = str(uuid.uuid4())
    db.sql(
        f"insert into auth.users(id,email,email_confirmed_at) values({literal(impostor)},'member@example.com',now())"
    )
    assert db.rpc("zils_access_claim", {"p_owner": impostor})["status"] != "active"
    # Membership belongs to the authenticated account after first redemption.
    db.sql(f"update auth.users set email='changed@example.com' where id={literal(owner)}")
    assert db.rpc("zils_access_claim", {"p_owner": owner})["status"] == "active"
    same = db.rpc(
        "zils_access_invite",
        {"p_email": "changed@example.com", "p_action": "approve", "p_actor": actor},
    )
    assert same["send"] is False
    assert db.rpc("zils_access_overview", {"p_status": "all", "p_offset": 0})["allocated"] == 1
    key = {
        "id": str(uuid.uuid4()),
        "owner_id": owner,
        "name": "app",
        "prefix": "zils_sk_test",
        "digest": "a" * 64,
    }
    db.rpc("zils_api_create_key", {"p_key": key})
    assert len(db.rpc("zils_api_auth", {"p_id": key["id"]})) == 1
    db.rpc(
        "zils_access_invite",
        {"p_email": "member@example.com", "p_action": "pause", "p_actor": actor},
    )
    assert not db.rpc("zils_access_allowed", {"p_owner": owner})
    assert db.rpc("zils_api_auth", {"p_id": key["id"]}) == []
    assert (
        db.rpc(
            "zils_api_admit",
            {"p_owner": owner, "p_key": key["id"], "p_request": str(uuid.uuid4()), "p_tokens": 1},
        )
        == "disabled"
    )
    for function, args in [
        ("zils_api_create_key", {"p_key": {**key, "id": str(uuid.uuid4())}}),
        ("fez_create_training_job", {"p_owner": owner, "p_name": "denied", "p_acceptance": {}}),
    ]:
        try:
            db.rpc(function, args)
        except APIError:
            pass
        else:
            raise AssertionError(function + " bypassed paused membership")
    # One remaining seat, two simultaneous approvals: exactly one succeeds.
    db.sql("update zils_access_settings set capacity=1")

    def approve(email):
        return db.rpc(
            "zils_access_invite", {"p_email": email, "p_action": "approve", "p_actor": actor}
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(approve, ["one@example.com", "two@example.com"]))
    assert sorted(r["status"] for r in results) == ["full", "ok"]
    invited = next(r["application"] for r in results if r["status"] == "ok")
    db.sql(
        f"update zils_access_applications set expires_at=now()-interval '1 second' where id={literal(invited['id'])}"
    )
    db.sql(f"update auth.users set email={literal(invited['email'])} where id={literal(stranger)}")
    assert db.rpc("zils_access_claim", {"p_owner": stranger})["status"] == "expired"
    assert approve("replacement@example.com")["status"] == "ok"
    assert db.rpc("zils_access_overview", {"p_status": "all", "p_offset": 0})["allocated"] == 1
    # Waiting-list growth never consumes seats; bursts are capped per source.
    for i in range(12):
        result = db.rpc(
            "zils_access_apply",
            {
                "p_email": f"waiting{i}@example.com",
                "p_use_case": "A useful decision",
                "p_source": "b" * 64,
            },
        )
        assert result == ("accepted" if i < 10 else "limited")
    assert db.rpc("zils_access_overview", {"p_status": "all", "p_offset": 0})["allocated"] == 1
    for role in ["anon", "authenticated"]:
        for operation in [
            "select * from zils_access_applications",
            f"select zils_access_allowed('{owner}')",
            f"select zils_access_invite('attacker@example.com','approve','{actor}')",
        ]:
            try:
                db.sql(f"set role {role}; {operation}")
            except APIError:
                pass
            else:
                raise AssertionError(role + " accessed private admission records")
    print(
        "Early access: capacity race, expiry, identity binding, pause, old keys, rate limits, and private RPCs passed."
    )
