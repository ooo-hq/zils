"""Real image metadata transactions, tenant policies and finalization fencing."""

import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor

from tests.api_database import Database, literal
from zils.cloud import APIError


def run(command):
    db = Database(command)
    owner, other = str(uuid.uuid4()), str(uuid.uuid4())
    for user in (owner, other):
        db.sql(
            f"insert into auth.users(id,email,email_confirmed_at) values({literal(user)},{literal(user + '@example.com')},now())"
        )
    db.rpc("zils_image_ensure_account", {"p_owner": owner})
    db.sql(f"update zils_api_accounts set enabled=false where owner_id={literal(owner)}")
    try:
        db.rpc("zils_image_ensure_account", {"p_owner": owner})
    except APIError:
        pass
    else:
        raise AssertionError("disabled account reenabled")
    db.sql(f"update zils_api_accounts set enabled=true where owner_id={literal(owner)}")

    def create():
        return db.rpc(
            "zils_image_create",
            {
                "p_owner": owner,
                "p_purpose": "prediction",
                "p_job": None,
                "p_filename": "photo.png",
                "p_source_bytes": 100,
                "p_source_sha256": hashlib.sha256(b"fixture").hexdigest(),
            },
        )

    asset = create()
    aid = asset["id"]
    assert asset["source_path"] != asset["canonical_path"]
    assert db.rpc("zils_image_get", {"p_owner": other, "p_asset": aid}) is None
    assert db.rpc("zils_image_get", {"p_owner": owner, "p_asset": aid})["id"] == aid

    def claim(_):
        return db.rpc("zils_image_claim_finalize", {"p_owner": owner, "p_asset": aid})

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, range(2)))
    assert sum(c is not None for c in claims) == 1
    lease = next(c for c in claims if c is not None)
    finish = {
        "p_owner": owner,
        "p_asset": aid,
        "p_token": lease["finalize_token"],
        "p_sha256": "a" * 64,
        "p_pixels": "b" * 64,
        "p_bytes": 200,
        "p_width": 8,
        "p_height": 8,
    }
    assert db.rpc("zils_image_finish", finish)["state"] == "ready"
    assert db.rpc("zils_image_finish", {**finish, "p_sha256": "c" * 64}) is None
    assert (
        db.rpc("zils_image_get", {"p_owner": owner, "p_asset": aid})["canonical_sha256"] == "a" * 64
    )
    # Owner metadata access, no mutation, no access via existing permissive Storage policy.
    assert (
        db.sql(
            f"set role authenticated; set request.jwt.claim.sub={literal(other)}; select count(*) from zils_image_assets"
        )
        == 0
    )
    assert (
        db.sql(
            f"set role authenticated; set request.jwt.claim.sub={literal(owner)}; select count(*) from zils_image_assets"
        )
        == 1
    )
    for role in ("anon", "authenticated"):
        for operation in (
            f"select zils_image_get('{owner}','{aid}')",
            "delete from zils_image_assets",
            "insert into storage.objects(bucket_id) values('zils-images')",
        ):
            try:
                db.sql(f"set role {role}; {operation}")
            except APIError:
                pass
            else:
                raise AssertionError(role + " accessed private image resources")
    db.sql("insert into storage.objects(bucket_id) values('zils-images')")
    assert (
        db.sql(
            "set role authenticated; select count(*) from storage.objects where bucket_id='zils-images'"
        )
        == 0
    )
    # Delete races cannot be undone by an old finalizer; retain keys until grants expire.
    second = create()
    lease = db.rpc("zils_image_claim_finalize", {"p_owner": owner, "p_asset": second["id"]})
    db.rpc("zils_image_delete", {"p_owner": owner, "p_asset": second["id"]})
    assert (
        db.rpc(
            "zils_image_finish",
            {**finish, "p_asset": second["id"], "p_token": lease["finalize_token"]},
        )
        is None
    )
    assert db.rpc("zils_image_get", {"p_owner": owner, "p_asset": second["id"]}) is None
    assert db.rpc("zils_image_cleanup_claim", {"p_limit": 100}) == []
    db.sql(
        f"update zils_image_assets set grant_expires_at=now()-interval '1 second',finalize_until=now()-interval '1 second' where id={literal(second['id'])}"
    )
    expired = db.rpc("zils_image_cleanup_claim", {"p_limit": 100})
    assert len(expired) == 1 and expired[0]["id"] == second["id"]

    # Atomic owner quotas under concurrent creates.
    def quota(_):
        return create()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(quota, range(24)))
    assert sum(r.get("error") == "limited" for r in results) == 5
    print(
        "Images: owner isolation, private storage, leases, cancellation, disabled accounts and concurrent quotas passed."
    )
    # Real Python asset flow on real transactions; substitute only blob transport.
    import importlib.util

    assert importlib.util.find_spec("zils.image_store") is not None, "image store missing"
    import base64
    import json
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from tests.test_image_assets import png
    from zils.decisions import DecisionError
    from zils.image_store import ImageStore

    class Blobs(Database):
        def download(self, bucket, path, destination, limit, *, max_seconds=600):
            assert max_seconds == 45
            return super().download(bucket, path, destination, limit)

        def signed(self, bucket, path, *, upload=False):
            assert bucket == "zils-images"
            payload = (
                base64.urlsafe_b64encode(json.dumps({"exp": int(time.time()) + 7200}).encode())
                .decode()
                .rstrip("=")
            )
            return {
                "url": "https://storage.example/storage/v1/object/upload/sign/zils-images/"
                + path
                + "?token=x."
                + payload
                + ".x",
                "method": "PUT",
                "headers": {"x-upsert": "false"},
            }

        def upload(self, bucket, path, source):
            assert path not in self.objects, "immutable write overwritten"
            self.objects[path] = Path(source).read_bytes()

        def remove(self, bucket, paths):
            for path in paths:
                self.objects.pop(path, None)

    blobs = Blobs(command)
    service = ImageStore(blobs)
    # Free active quota without discarding the rolling grant history.
    db.sql(f"update zils_image_assets set state='deleted' where owner_id={literal(owner)}")
    raw = png()
    draft = service.create(
        owner, "prediction", None, "private/photo.png", len(raw), hashlib.sha256(raw).hexdigest()
    )
    assert "source_path" not in draft["asset"] and "canonical_path" not in draft["asset"]
    row = db.rpc("zils_image_get", {"p_owner": owner, "p_asset": draft["asset"]["id"]})
    blobs.objects[row["source_path"]] = raw
    ready = service.complete(owner, row["id"])
    assert ready["state"] == "ready" and ready["width"] == 8 and ready["sha256"]
    assert service.complete(owner, row["id"]) == ready
    assert service.resolve(owner, row["id"]) == ready
    try:
        service.resolve(other, row["id"])
    except DecisionError as error:
        assert error.status == 404
    else:
        raise AssertionError("cross-owner asset exposed")
    bad = service.create(owner, "prediction", None, "wrong.png", len(raw), "0" * 64)
    row = db.rpc("zils_image_get", {"p_owner": owner, "p_asset": bad["asset"]["id"]})
    blobs.objects[row["source_path"]] = raw
    try:
        service.complete(owner, row["id"])
    except DecisionError as error:
        assert error.status == 422
    else:
        raise AssertionError("changed source bytes published")
    assert db.rpc("zils_image_get", {"p_owner": owner, "p_asset": row["id"]}) is None
    # A crash after writing canonical bytes is resumed without overwriting the object.
    draft = service.create(
        owner, "prediction", None, "resume.png", len(raw), hashlib.sha256(raw).hexdigest()
    )
    aid = draft["asset"]["id"]
    row = db.rpc("zils_image_get", {"p_owner": owner, "p_asset": aid})
    blobs.objects[row["source_path"]] = raw
    original_rpc = blobs.rpc

    def interrupted(function, args):
        if function == "zils_image_finish":
            raise APIError(503, "simulated interruption after immutable upload")
        return original_rpc(function, args)

    blobs.rpc = interrupted
    try:
        service.complete(owner, aid)
    except APIError:
        pass
    else:
        raise AssertionError("interruption not exercised")
    assert row["canonical_path"] in blobs.objects
    blobs.rpc = original_rpc
    db.sql(
        f"update zils_image_assets set finalize_until=now()-interval '1 second' where id={literal(aid)}"
    )
    assert service.complete(owner, aid)["state"] == "ready"
    service.delete_unused(owner, aid)
    assert service.cleanup()["removed"] == 0  # upload grant can still produce a late source
    db.sql(
        f"update zils_image_assets set grant_expires_at=now()-interval '1 second',finalize_until=now()-interval '1 second' where id={literal(aid)}"
    )
    assert service.cleanup()["removed"] == 1
    assert row["source_path"] not in blobs.objects and row["canonical_path"] not in blobs.objects

    # Training assets respect frozen job identity, reference protection and terminal retention.
    job = str(uuid.uuid4())
    db.sql(
        f"insert into fez_training_jobs(id,owner_id,name,acceptance,model_profile) values({literal(job)},{literal(owner)},'image-fixture','{{}}','{{\"id\":\"imajev-4b-v1\"}}')"
    )
    draft = service.create(
        owner, "training", job, "train.png", len(raw), hashlib.sha256(raw).hexdigest()
    )
    aid = draft["asset"]["id"]
    row = db.rpc("zils_image_get", {"p_owner": owner, "p_asset": aid})
    blobs.objects[row["source_path"]] = raw
    recovered = service.resume(owner, aid)
    assert recovered["uploaded"] is True and "upload" not in recovered
    service.complete(owner, aid)
    assert service.resume(owner, aid)["asset"]["state"] == "ready"
    db.sql(
        f"update zils_image_assets set referenced=true,expires_at=now()-interval '1 day',grant_expires_at=now()-interval '1 day',finalize_until=now()-interval '1 second' where id={literal(aid)}"
    )
    unfinished = service.create(
        owner, "training", job, "unfinished.png", len(raw), hashlib.sha256(raw).hexdigest()
    )
    db.sql(f"update fez_training_jobs set status='running' where id={literal(job)}")
    try:
        service.resume(owner, unfinished["asset"]["id"])
    except DecisionError as error:
        assert error.status == 404
    else:
        raise AssertionError("issued an upload grant after the job stopped accepting uploads")
    db.sql(
        f"update zils_image_assets set grant_expires_at=now()-interval '1 day' where id={literal(unfinished['asset']['id'])}"
    )
    assert service.resolve(owner, aid)["state"] == "ready"  # live jobs survive draft TTL
    grant = service.read_reference(owner, aid, "training")
    remaining = (
        datetime.fromisoformat(grant["expires_at"]) - datetime.now(timezone.utc)
    ).total_seconds()
    assert 0 < remaining <= 600, (
        "runtime read deadline must follow the live grant, not expired draft TTL"
    )
    try:
        service.delete_unused(owner, aid)
    except APIError:
        pass
    else:
        raise AssertionError("referenced training image deleted")
    assert service.cleanup()["removed"] == 0
    db.sql(
        f"update fez_training_jobs set status='completed',updated_at=now()-interval '29 days' where id={literal(job)}"
    )
    assert service.resolve(owner, aid)["state"] == "ready"
    db.sql(
        f"update fez_training_jobs set updated_at=now()-interval '30 days'+interval '5 minutes' where id={literal(job)}"
    )
    grant = service.read_reference(owner, aid, "training")
    assert (
        0
        < (datetime.fromisoformat(grant["expires_at"]) - datetime.now(timezone.utc)).total_seconds()
        <= 300
    )
    db.sql(
        f"update fez_training_jobs set updated_at=now()-interval '31 days' where id={literal(job)}"
    )
    try:
        service.resolve(owner, aid)
    except DecisionError as error:
        assert error.status == 404
    else:
        raise AssertionError("expired training asset still accessible")
    assert service.cleanup()["removed"] == 2
    print(
        "Images: canonical bytes, source integrity, crash recovery and live/terminal retention passed."
    )
