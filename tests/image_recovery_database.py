"""Real SQL fencing for interrupted image finalization and stale cleanup."""

import uuid
from concurrent.futures import ThreadPoolExecutor

from tests.api_database import Database, literal
from zils.cloud import APIError


def run(command):
    db = Database(command)
    owner, other = str(uuid.uuid4()), str(uuid.uuid4())
    for user in (owner, other):
        db.sql(f"insert into auth.users(id) values({literal(user)})")
        db.rpc("zils_image_ensure_account", {"p_owner": user})
    asset = db.rpc(
        "zils_image_create",
        {
            "p_owner": owner,
            "p_purpose": "prediction",
            "p_job": None,
            "p_filename": "recovery.png",
            "p_source_bytes": 100,
            "p_source_sha256": "a" * 64,
        },
    )
    aid = asset["id"]

    def claim(token):
        return db.rpc(
            "zils_image_claim_finalize_request",
            {"p_owner": owner, "p_asset": aid, "p_token": token},
        )

    def release(token, user=owner):
        db.rpc("zils_image_release_finalize", {"p_owner": user, "p_asset": aid, "p_token": token})

    tokens = [str(uuid.uuid4()), str(uuid.uuid4())]
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, tokens))
    assert sum(row is not None for row in claims) == 1
    first = next(row for row in claims if row is not None)
    token = first["finalize_token"]
    assert claim(token) == first  # Lost replies neither steal nor extend the lease.
    release(str(uuid.uuid4()))
    release(token, other)
    assert claim(token) == first
    assert claim(str(uuid.uuid4())) is None
    release(token)
    assert db.rpc("zils_image_get", {"p_owner": owner, "p_asset": aid})["state"] == "uploading"

    next_token = str(uuid.uuid4())
    second = claim(next_token)
    release(token)  # A delayed cleanup from the previous request cannot unlock its successor.
    assert claim(next_token) == second
    finish = {
        "p_owner": owner,
        "p_asset": aid,
        "p_token": token,
        "p_sha256": "b" * 64,
        "p_pixels": "c" * 64,
        "p_bytes": 200,
        "p_width": 8,
        "p_height": 8,
    }
    assert db.rpc("zils_image_finish", finish) is None
    assert db.rpc("zils_image_finish", {**finish, "p_token": next_token})["state"] == "ready"
    release(next_token)
    assert db.rpc("zils_image_get", {"p_owner": owner, "p_asset": aid})["state"] == "ready"
    for role in ("anon", "authenticated"):
        for name in ("zils_image_claim_finalize_request", "zils_image_release_finalize"):
            try:
                db.sql(f"set role {role}; select {name}('{owner}','{aid}','{next_token}')")
            except APIError:
                pass
            else:
                raise AssertionError("Untrusted role accessed finalization leases")
    db.rpc("zils_image_delete", {"p_owner": owner, "p_asset": aid})
    release(next_token)
    assert db.rpc("zils_image_get", {"p_owner": owner, "p_asset": aid}) is None
    print(
        "Image recovery: concurrent claims, lost replies, stale cleanup and role isolation passed."
    )
