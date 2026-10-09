"""Exercise storage fencing and private access against actual PostgreSQL."""

import uuid
from concurrent.futures import ThreadPoolExecutor

from tests.api_database import Database, literal
from zils.cloud import APIError


def run(command):
    db = Database(command)
    bucket, path = "zils-images", f"{uuid.uuid4()}/asset/source.png"
    base = {"p_bucket": bucket, "p_path": path}
    token = str(uuid.uuid4())
    values = {"provider": "spaces", "physical_bucket": "test-storage", "max_bytes": 100}

    def call(action, owner=token, **data):
        return db.rpc(
            "zils_storage_object",
            {**base, "p_action": action, "p_token": owner, "p_values": data},
        )

    assert call("get") is None
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda t: call("allocate", t, **values), [token, str(uuid.uuid4())]))
    assert sum("error" not in r for r in rows) == 1, rows
    row = next(r for r in rows if "error" not in r)
    token = row["token"]
    assert call("allocate", token, **values) == row
    generation = row["generation"]
    assert row["physical_key"] == f"objects/{generation}/{bucket}/{path}"
    assert call("bind", str(uuid.uuid4()), generation=generation, upload_id="one") == {
        "error": "conflict"
    }
    bound = call("bind", token, generation=generation, upload_id="one")
    assert bound["upload_id"] == "one"
    assert call("bind", token, generation=generation, upload_id="one") == bound
    assert call("bind", token, generation=generation, upload_id="two")["error"] == "conflict"
    assert call("grant", token, generation=generation, seconds=601)["error"] == "invalid"
    assert (
        call("grant", str(uuid.uuid4()), generation=generation, seconds=600)["error"] == "conflict"
    )
    granted = call("grant", token, generation=generation, seconds=600)
    assert granted["grant_expires_at"]
    assert call("allocate", str(uuid.uuid4()), **values)["error"] == "conflict"
    seal_values = {"generation": generation, "part_etag": '"' + "a" * 32 + '"', "part_size": 3}
    seal_tokens = [str(uuid.uuid4()), str(uuid.uuid4())]
    with ThreadPoolExecutor(max_workers=2) as pool:
        seals = list(pool.map(lambda t: call("seal", t, **seal_values), seal_tokens))
    assert sum("error" not in r for r in seals) == 1, seals
    sealed = next(r for r in seals if "error" not in r)
    seal_token = sealed["token"]
    assert call("seal", seal_token, **seal_values) == sealed
    assert call("grant", token, generation=generation, seconds=600)["error"] == "conflict"
    assert call("commit", token, generation=generation, size_bytes=3)["error"] == "conflict"
    assert call("commit", seal_token, generation=generation, size_bytes=101)["error"] == "invalid"
    ready = call("commit", seal_token, generation=generation, size_bytes=3)
    assert ready["state"] == "ready" and ready["size_bytes"] == 3
    assert call("commit", seal_token, generation=generation, size_bytes=3) == ready
    assert call("commit", seal_token, generation=generation, size_bytes=4)["error"] == "conflict"
    assert call("allocate", token, **values)["error"] == "conflict"

    # Namespace isolation and a lost creation response require a fresh physical key.
    base["p_bucket"] = "fez-training-data"
    first = call("allocate", token, **values)
    db.sql(
        "update zils_storage_objects set lease_until=now()-interval '1 second' "
        f"where generation={literal(first['generation'])}"
    )
    successor = call("allocate", str(uuid.uuid4()), **values)
    assert successor["physical_key"] != first["physical_key"]
    assert (
        call("bind", token, generation=first["generation"], upload_id="late")["error"] == "conflict"
    )
    assert (
        db.sql(
            "select count(*) from zils_storage_retired "
            f"where generation={literal(first['generation'])}"
        )
        == 1
    )
    base["p_bucket"] = bucket
    assert call("get") == ready

    # Deletion wins over late completion and leaves a permanent tombstone.
    deletion_token = str(uuid.uuid4())
    deleting = call("delete", deletion_token)
    assert deleting["state"] == "deleting"
    assert call("commit", seal_token, generation=generation, size_bytes=3)["error"] == "conflict"
    assert call("deleted", deletion_token, generation=generation)["error"] == "conflict"
    db.sql(
        "update zils_storage_objects set cleanup_after=now()-interval '1 second' "
        f"where generation={literal(generation)}"
    )
    assert call("deleted", deletion_token, generation=generation)["state"] == "deleted"
    assert call("allocate", token, **values)["error"] == "conflict"
    base["p_path"] = f"{uuid.uuid4()}/never-uploaded"
    assert call("delete", deletion_token)["state"] == "deleted"
    assert call("register_legacy", token, size_bytes=3)["error"] == "conflict"

    # Copies keep their legacy source until verified; a lost commit is idempotent.
    base["p_path"] = f"{uuid.uuid4()}/legacy.png"
    legacy = call("register_legacy", token, size_bytes=3)
    assert legacy["provider"] == "supabase" and legacy["state"] == "ready"
    copying = call("begin_copy", token, physical_bucket="test-storage", sha256="b" * 64)
    assert copying["legacy_readable"] and copying["provider"] == "spaces"
    g = copying["generation"]
    call("bind", token, generation=g, upload_id="copy")
    assert (
        call("seal", str(uuid.uuid4()), generation=g, part_etag='"' + "a" * 32 + '"', part_size=3)[
            "error"
        ]
        == "conflict"
    )
    call("seal", token, generation=g, part_etag='"' + "a" * 32 + '"', part_size=3)
    assert call("commit", token, generation=g, size_bytes=3)["error"] == "invalid"
    assert call("commit", token, generation=g, size_bytes=3, sha256="c" * 64)["error"] == "invalid"
    copied = call("commit", token, generation=g, size_bytes=3, sha256="b" * 64)
    assert copied["state"] == "ready" and not copied["legacy_readable"]

    # Expired sealing can be reclaimed, but the previous token cannot commit/renew.
    base["p_path"] = f"{uuid.uuid4()}/recovery"
    a = call("allocate", token, **values)
    g = a["generation"]
    call("bind", token, generation=g, upload_id="recovery")
    call("seal", token, generation=g, part_etag='"' + "a" * 32 + '"', part_size=3)
    db.sql(
        f"update zils_storage_objects set lease_until=now()-interval '1 second' where generation={literal(g)}"
    )
    t = str(uuid.uuid4())
    assert call("seal", t, generation=g, part_etag='"' + "a" * 32 + '"', part_size=3)["token"] == t
    assert call("renew", token, generation=g)["error"] == "conflict"
    assert call("commit", token, generation=g, size_bytes=3)["error"] == "conflict"
    assert call("renew", t, generation=g)["state"] == "sealing"

    # Cleanup enumeration is bounded and cannot select a still-valid grant.
    db.sql(
        f"update zils_storage_objects set lease_until=now()-interval '1 second' where generation={literal(g)}"
    )
    pending = db.sql("select coalesce(jsonb_agg(t),'[]') from zils_storage_pending(now(),100) t")
    assert any(item["generation"] == g for item in pending)
    db.sql(
        f"update zils_storage_objects set grant_expires_at=now()+interval '1 hour' where generation={literal(g)}"
    )
    pending = db.sql("select coalesce(jsonb_agg(t),'[]') from zils_storage_pending(now(),100) t")
    assert not any(item["generation"] == g for item in pending)
    assert db.sql("select count(*) from zils_storage_pending(now(),0)") == 0

    base["p_path"] = f"{uuid.uuid4()}/delete-race"
    a = call("allocate", token, **values)
    g = a["generation"]
    call("bind", token, generation=g, upload_id="race")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(call, "delete", str(uuid.uuid4())),
            pool.submit(
                call,
                "seal",
                str(uuid.uuid4()),
                generation=g,
                part_etag='"' + "a" * 32 + '"',
                part_size=3,
            ),
        ]
        for future in futures:
            future.result()
    assert call("get")["state"] == "deleting"

    for invalid in ("../secret", "a//b", "a/%2f/b", "/absolute", "a/./b", "a\\b"):
        base["p_path"] = invalid
        assert call("allocate", token, **values)["error"] == "invalid"
    for role in ("anon", "authenticated"):
        for sql in (
            "select count(*) from zils_storage_objects",
            "select count(*) from zils_storage_retired",
            "select zils_storage_object('zils-images','a/b','get')",
            "select count(*) from zils_storage_pending(now(),100)",
        ):
            try:
                db.sql(f"set role {role}; {sql}")
            except APIError:
                pass
            else:
                raise AssertionError(f"Untrusted role {role} accessed private object state")
    print(
        "Storage catalog: fencing, recovery, copy verification, tombstones and role isolation passed."
    )
