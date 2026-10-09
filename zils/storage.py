"""Private logical object identities, independent of authentication and file hosts."""

import base64
import hashlib
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, quote, urlsplit

from . import settings
from .cloud import APIError, download, upload

LIMITS = {
    "fez-training-data": 128 * 1024**2,
    "fez-training-models": 512 * 1024**2,
    "zils-images": 10 * 1024**2,
    "zils-api-batches": 25 * 1024**2,
}


def fingerprint(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return path.stat().st_size, digest


def object_path(bucket, path):
    if (
        bucket not in LIMITS
        or not isinstance(path, str)
        or not 1 <= len(path) <= 1024
        or any(segment in ("", ".", "..") for segment in path.split("/"))
        or re.search(r"[\\%\x00-\x1f\x7f]", path)
    ):
        raise ValueError("Invalid storage object path")
    return path


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Storage expiry requires a timezone")
    return parsed


def expired(value):
    return timestamp(value) <= datetime.now(timezone.utc)


class ObjectCatalog:
    def __init__(self, db):
        self.db = db

    def get(self, bucket, path):
        object_path(bucket, path)
        return self.db.rpc(
            "zils_storage_object", {"p_bucket": bucket, "p_path": path, "p_action": "get"}
        )

    def change(self, bucket, path, action, token, values=None):
        object_path(bucket, path)
        row = self.db.rpc(
            "zils_storage_object",
            {
                "p_bucket": bucket,
                "p_path": path,
                "p_action": action,
                "p_token": token,
                "p_values": values or {},
            },
        )
        if not row or row.get("error"):
            raise APIError(409, "Storage operation changed or expired; retry.")
        return row

    def pending(self, limit=100):
        return self.db.rpc(
            "zils_storage_pending",
            {
                "p_before": datetime.now(timezone.utc).isoformat(),
                "p_limit": limit,
            },
        )

    def scan(self, limit=100, *, kind="all"):
        before = datetime.now(timezone.utc).isoformat()
        cursor = None
        while True:
            rows = self.db.rpc(
                "zils_storage_pending",
                {
                    "p_before": before,
                    "p_limit": limit,
                    "p_kind": kind,
                    "p_after": cursor,
                },
            )
            yield from rows
            if len(rows) < min(limit, 100) or not rows:
                return
            cursor = {key: rows[-1][key] for key in ("updated_at", "bucket", "path")}


def legacy_expiry(url):
    """Read only a grant returned by trusted Supabase Storage, never a client token."""
    try:
        payload = parse_qs(urlsplit(url).query)["token"][0].split(".")[1]
        expiry = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["exp"]
        if type(expiry) is not int:
            raise ValueError()
        deadline = datetime.fromtimestamp(expiry, timezone.utc)
        if (
            not datetime.now(timezone.utc)
            < deadline
            <= datetime.now(timezone.utc) + timedelta(days=1)
        ):
            raise ValueError()
        return deadline.isoformat()
    except (KeyError, IndexError, ValueError, TypeError, OverflowError):
        raise APIError(503, "Upload expiry could not be verified.") from None


class SupabaseStorage:
    def __init__(self, db):
        self.db = db

    def signed(self, bucket, path, *, upload=False):
        route = "upload/sign" if upload else "sign"
        data = self.db.request(
            "POST",
            f"/storage/v1/object/{route}/{bucket}/{quote(path, safe='/')}",
            {} if upload else {"expiresIn": 600},
            {"x-upsert": "false"},
        )
        relative = data["url"] if upload else data["signedURL"]
        if not relative.startswith("/object/"):
            raise APIError(503, "Unexpected storage response.")
        result = {"url": self.db.url + "/storage/v1" + relative, "provider": "supabase"}
        result["expires_at"] = (
            legacy_expiry(result["url"])
            if upload
            else (datetime.now(timezone.utc) + timedelta(seconds=600)).isoformat()
        )
        if upload:
            result.update(
                method="PUT",
                headers={"Content-Type": "application/octet-stream", "x-upsert": "false"},
            )
        return result

    def download(self, bucket, path, destination, limit, *, max_seconds=600):
        return download(
            self.signed(bucket, path)["url"], destination, limit, max_seconds=max_seconds
        )

    def upload(self, bucket, path, source):
        # Trusted processors use immutable, attempt-specific paths; never overwrite.
        url = self.db.url + "/storage/v1/object/" + bucket + "/" + quote(path, safe="/")
        upload(
            url,
            source,
            {**self.db.headers, "Content-Type": "application/octet-stream", "x-upsert": "false"},
            method="POST",
        )

    def exists(self, bucket, path):
        parent, name = path.rsplit("/", 1)
        rows = self.db.request(
            "POST",
            f"/storage/v1/object/list/{bucket}",
            {"prefix": parent, "search": name, "limit": 100},
        )
        return any(row["name"] == name for row in rows)

    def remove(self, bucket, paths):
        if not isinstance(paths, list) or not 1 <= len(paths) <= 100:
            raise ValueError("remove requires 1..100 exact object paths")
        return self.db.request(
            "DELETE", "/storage/v1/object/" + quote(bucket, safe=""), {"prefixes": paths}
        )

    def stat(self, bucket, path):
        parent, name = path.rsplit("/", 1)
        rows = self.db.request(
            "POST",
            f"/storage/v1/object/list/{bucket}",
            {"prefix": parent, "search": name, "limit": 100},
        )
        for row in rows:
            if row["name"] == name:
                size = (row.get("metadata") or {}).get("size")
                if type(size) is not int or size < 1:
                    raise APIError(503, "Legacy storage size could not be verified.")
                return {"size": size, "id": row.get("id"), "updated_at": row.get("updated_at")}
        return None


class StorageRouter:
    def __init__(self, db, legacy, spaces, catalog, write_provider):
        if write_provider not in ("spaces", "supabase"):
            raise ValueError("Unknown storage provider")
        self.db, self.legacy, self._spaces = db, legacy, spaces
        self.catalog, self.write_provider = catalog, write_provider

    @property
    def spaces(self):
        if self._spaces is None:
            from .spaces import SpacesStorage, configured_client

            client, bucket = configured_client()
            self._spaces = SpacesStorage(self.catalog, client, bucket)
        return self._spaces

    def _reader(self, bucket, path):
        row = self.catalog.get(bucket, path)
        if row and row["state"] in ("deleting", "deleted"):
            raise APIError(404, "Storage object is unavailable.")
        if not row or row["legacy_readable"]:
            return self.legacy
        return self.spaces if row["provider"] == "spaces" else self.legacy

    def _writer(self, bucket, path):
        row = self.catalog.get(bucket, path)
        if not row:
            existing = self.legacy.stat(bucket, path)
            if existing:
                row = self.catalog.change(
                    bucket,
                    path,
                    "register_legacy",
                    str(uuid.uuid4()),
                    {"size_bytes": existing["size"]},
                )
            elif self.write_provider == "supabase":
                row = self.catalog.change(
                    bucket,
                    path,
                    "allocate",
                    str(uuid.uuid4()),
                    {"provider": "supabase", "max_bytes": LIMITS[bucket]},
                )
        if row:
            if row["state"] in ("ready", "deleting", "deleted") or row["legacy_readable"]:
                raise APIError(409, "Uploaded files cannot be replaced.")
            return (self.spaces if row["provider"] == "spaces" else self.legacy), row
        return self.spaces, None

    def _legacy_ready(self, bucket, path):
        row = self.catalog.get(bucket, path)
        if not row or row["provider"] != "supabase" or row["state"] == "ready":
            return
        metadata = self.legacy.stat(bucket, path)
        if not metadata:
            return
        token = str(uuid.uuid4())
        row = self.catalog.change(
            bucket,
            path,
            "seal",
            token,
            {"generation": row["generation"], "part_size": metadata["size"]},
        )
        self.catalog.change(
            bucket,
            path,
            "commit",
            row["token"],
            {"generation": row["generation"], "size_bytes": metadata["size"]},
        )

    def signed(self, bucket, path, *, upload=False):
        if not upload:
            provider = self._reader(bucket, path)
            if provider is self.legacy:
                self._legacy_ready(bucket, path)
            return provider.signed(bucket, path)
        provider, row = self._writer(bucket, path)
        result = provider.signed(bucket, path, upload=True)
        if provider is self.legacy:
            self.catalog.change(
                bucket,
                path,
                "grant",
                row["token"],
                {"generation": row["generation"], "expires_at": result["expires_at"]},
            )
        return result

    def exists(self, bucket, path):
        try:
            provider = self._reader(bucket, path)
        except APIError as error:
            if error.status == 404:
                return False
            raise
        exists = provider.exists(bucket, path)
        if exists and provider is self.legacy:
            self._legacy_ready(bucket, path)
        return exists

    def download(self, bucket, path, destination, limit, *, max_seconds=600):
        provider = self._reader(bucket, path)
        if provider is self.legacy:
            self._legacy_ready(bucket, path)
        return provider.download(bucket, path, destination, limit, max_seconds=max_seconds)

    def upload(self, bucket, path, source):
        provider, row = self._writer(bucket, path)
        provider.upload(bucket, path, source)
        if provider is self.legacy:
            self._legacy_ready(bucket, path)

    def remove(self, bucket, paths):
        if not isinstance(paths, list) or not 1 <= len(paths) <= 100:
            raise ValueError("remove requires 1..100 exact object paths")
        for path in paths:
            row = self.catalog.get(bucket, path)
            if row and row["provider"] == "spaces":
                self.spaces.remove(bucket, [path])
                if row.get("legacy_copy"):
                    self.legacy.remove(bucket, [path])
            else:
                row = self.catalog.change(bucket, path, "delete", str(uuid.uuid4()))
                self.legacy.remove(bucket, [path])
                # Keep deleting until legacy grants expire; readers already see absence.
                if row["state"] == "deleting" and expired(row["cleanup_after"]):
                    self.catalog.change(
                        bucket, path, "deleted", row["token"], {"generation": row["generation"]}
                    )

    def reap(self, limit=100, *, dry_run=False, eligible=None):
        legacy_removed = 0
        for row in self.catalog.scan(limit, kind="deleting"):
            if legacy_removed >= limit:
                break
            if row["state"] != "deleting" or not expired(row["cleanup_after"]):
                continue
            if row["provider"] == "supabase" or row.get("legacy_copy"):
                if not dry_run:
                    row = self.catalog.change(
                        row["bucket"], row["path"], "delete", str(uuid.uuid4())
                    )
                    self.legacy.remove(row["bucket"], [row["path"]])
                    if row["provider"] == "supabase":
                        self.catalog.change(
                            row["bucket"],
                            row["path"],
                            "deleted",
                            row["token"],
                            {"generation": row["generation"]},
                        )
                    else:
                        self.spaces._cleanup(row)
                legacy_removed += 1
        return self.spaces.reap(limit, dry_run=dry_run, eligible=eligible) | {
            "legacy_removed": legacy_removed
        }


def storage_for(db):
    mode = settings.get("ZILS_STORAGE_CATALOG", "off")
    provider = settings.get("ZILS_STORAGE_WRITE_PROVIDER", "supabase")
    if mode not in ("on", "off") or provider not in ("supabase", "spaces"):
        raise ValueError("Invalid storage configuration")
    legacy = SupabaseStorage(db)
    if mode == "off":
        if provider != "supabase":
            raise ValueError("Spaces writes require ZILS_STORAGE_CATALOG=on")
        return legacy
    return StorageRouter(db, legacy, None, ObjectCatalog(db), provider)
