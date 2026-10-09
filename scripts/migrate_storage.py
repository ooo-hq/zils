"""Inventory and copy private files without deleting legacy source objects."""

import argparse
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from zils.cloud import APIError, Supabase
from zils.spaces import SpacesStorage, configured_client
from zils.storage import (
    LIMITS,
    ObjectCatalog,
    StorageRouter,
    SupabaseStorage,
    fingerprint,
    object_path,
    timestamp,
)

VERSION = "zils-storage-inventory/v1"


def write_report(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".storage-report-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def eligibility(db, bucket, path):
    """Unknown owners and active jobs fail closed; file paths never form raw filters."""
    pieces = object_path(bucket, path).split("/")
    try:
        if bucket in ("fez-training-data", "fez-training-models"):
            job_id = str(uuid.UUID(pieces[0]))
            rows = db.rows("fez_training_jobs", f"id=eq.{job_id}&limit=1")
            if not rows:
                return "deferred", "unknown_job"
            state = rows[0]["status"]
            return ("eligible" if state in ("completed", "failed") else "deferred"), state
        if bucket == "zils-images":
            owner, asset_id = str(uuid.UUID(pieces[0])), str(uuid.UUID(pieces[1]))
            rows = db.rows("zils_image_assets", f"id=eq.{asset_id}&owner_id=eq.{owner}&limit=1")
            if not rows or path not in (rows[0]["source_path"], rows[0]["canonical_path"]):
                return "deferred", "unknown_asset"
            asset = rows[0]
            if asset["state"] != "ready":
                return "deferred", asset["state"]
            if asset["purpose"] == "training":
                return eligibility(db, "fez-training-data", asset["job_id"] + "/inputs/train.jsonl")
            return (
                "eligible"
                if timestamp(asset["expires_at"]) > datetime.now(timezone.utc)
                else "deferred"
            ), "ready"
        owner, batch_id = str(uuid.UUID(pieces[0])), str(uuid.UUID(pieces[1]))
        rows = db.rows("zils_api_batches", f"id=eq.{batch_id}&owner_id=eq.{owner}&limit=1")
        if not rows or rows[0]["input_path"] != path or rows[0]["purged_at"]:
            return "deferred", "unknown_batch"
        state = rows[0]["status"]
        return (
            "eligible" if state in ("completed", "failed", "cancelled", "expired") else "deferred"
        ), state
    except (ValueError, KeyError, IndexError):
        return "deferred", "unrecognized_path"


def inventory(db, page_size=100):
    objects, seen = [], set()
    for bucket in LIMITS:
        folders, visited = [""], set()
        while folders:
            prefix = folders.pop()
            if prefix in visited:
                continue
            visited.add(prefix)
            offset = 0
            while True:
                page = db.request(
                    "POST",
                    "/storage/v1/object/list/" + bucket,
                    {
                        "prefix": prefix,
                        "limit": page_size,
                        "offset": offset,
                        "sortBy": {"column": "name", "order": "asc"},
                    },
                )
                if not isinstance(page, list):
                    raise APIError(503, "Storage listing unavailable.")
                for entry in page:
                    name = entry.get("name")
                    if not isinstance(name, str) or "/" in name:
                        raise APIError(503, "Storage listing invalid.")
                    path = object_path(bucket, f"{prefix}/{name}" if prefix else name)
                    if entry.get("id") is None and entry.get("metadata") is None:
                        folders.append(path)
                        continue
                    key = (bucket, path)
                    if key in seen:
                        raise APIError(409, "Storage listing changed; repeat inventory.")
                    seen.add(key)
                    size = (entry.get("metadata") or {}).get("size")
                    eligible, state = eligibility(db, bucket, path)
                    if type(size) is not int or not 0 < size <= LIMITS[bucket]:
                        eligible = "deferred"
                    objects.append(
                        {
                            "bucket": bucket,
                            "path": path,
                            "source_size": size,
                            "source_sha256": None,
                            "source_id": entry.get("id"),
                            "source_updated_at": entry.get("updated_at"),
                            "owner_state": state,
                            "eligibility": eligible,
                            "copy_state": "inventoried",
                        }
                    )
                if len(page) < page_size:
                    break
                offset += len(page)
    return {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objects": objects,
    }


def migrate_one(db, legacy, spaces, item, *, apply=False):
    result = dict(item)
    if not apply:
        return result | {
            "copy_state": "planned" if item["eligibility"] == "eligible" else "deferred"
        }
    try:
        bucket, path = item["bucket"], object_path(item["bucket"], item["path"])
        eligible, state = eligibility(db, bucket, path)
        if item["eligibility"] != "eligible" or eligible != "eligible":
            return result | {"copy_state": "deferred", "owner_state": state}
        row = spaces.catalog.get(bucket, path)
        if row and row["state"] in ("deleting", "deleted"):
            return result | {"copy_state": "deleted"}
        if row and row["provider"] == "spaces" and row["state"] == "ready":
            if (
                not row["sha256"]
                or row["size_bytes"] != item["source_size"]
                or (item.get("source_sha256") and row["sha256"] != item["source_sha256"])
            ):
                raise APIError(409, "Catalog does not match inventory.")
            return result | {
                "copy_state": "verified",
                "source_sha256": row["sha256"],
                "target_size": row["size_bytes"],
                "target_sha256": row["sha256"],
            }
        before = legacy.stat(bucket, path)
        if not before or before["size"] != item["source_size"]:
            raise APIError(422, "Source changed since inventory.")
        for field, value in (
            ("id", item.get("source_id")),
            ("updated_at", item.get("source_updated_at")),
        ):
            if value and before.get(field) != value:
                raise APIError(422, "Source changed since inventory.")
        with tempfile.TemporaryDirectory(prefix="zils-storage-copy-") as directory:
            source = Path(directory) / "source"
            legacy.download(bucket, path, source, LIMITS[bucket])
            size, sha256 = fingerprint(source)
            if (
                size != item["source_size"]
                or not 0 < size <= LIMITS[bucket]
                or (item.get("source_sha256") and item["source_sha256"] != sha256)
            ):
                raise APIError(422, "Source checksum mismatch.")
            result.update(source_sha256=sha256)
            if (
                before != legacy.stat(bucket, path)
                or eligibility(db, bucket, path)[0] != "eligible"
            ):
                raise APIError(409, "Source changed while downloading.")
            if row is None:
                spaces.catalog.change(
                    bucket, path, "register_legacy", str(uuid.uuid4()), {"size_bytes": size}
                )
            token = str(uuid.uuid4())
            spaces.catalog.change(
                bucket,
                path,
                "begin_copy",
                token,
                {"physical_bucket": spaces.bucket, "sha256": sha256},
            )
            copied = spaces.copy_verified(bucket, path, source, sha256, token)
        return result | {
            "copy_state": "verified",
            "target_size": copied["size_bytes"],
            "target_sha256": copied["sha256"],
        }
    except ValueError:
        return result | {"copy_state": "failed", "error_code": "invalid_file"}
    except APIError as error:
        return result | {
            "copy_state": "retry" if error.status in (409, 503) else "failed",
            "error_code": f"storage_{error.status}",
        }


def verify_one(spaces, item):
    result = dict(item)
    try:
        row = spaces.catalog.get(item["bucket"], item["path"])
        if (
            not row
            or row["state"] != "ready"
            or row["provider"] != "spaces"
            or not item.get("source_sha256")
        ):
            raise APIError(409, "Copy is not ready for verification.")
        with tempfile.TemporaryDirectory(prefix="zils-storage-audit-") as directory:
            target = Path(directory) / "object"
            spaces.download(item["bucket"], item["path"], target, LIMITS[item["bucket"]])
            size, sha256 = fingerprint(target)
        if (size, sha256) != (item["source_size"], item["source_sha256"]):
            raise APIError(422, "Copy checksum mismatch.")
        return result | {"copy_state": "verified", "target_size": size, "target_sha256": sha256}
    except ValueError:
        return result | {"copy_state": "failed", "error_code": "invalid_file"}
    except APIError as error:
        return result | {"copy_state": "failed", "error_code": f"storage_{error.status}"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "copy", "verify", "cleanup"])
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".private/storage-report.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    db = Supabase()
    spaces = None
    if args.action in ("verify", "cleanup") or (args.action == "copy" and args.apply):
        client, bucket = configured_client()
        spaces = SpacesStorage(ObjectCatalog(db), client, bucket)
    if args.action == "inventory":
        report = inventory(db)
    elif args.action == "cleanup":
        report = StorageRouter(db, SupabaseStorage(db), spaces, spaces.catalog, "spaces").reap(
            dry_run=not args.apply,
            eligible=lambda row: eligibility(db, row["bucket"], row["path"])[0] == "eligible",
        )
    else:
        if not args.inventory:
            parser.error("--inventory is required for copy/verify")
        source = json.loads(args.inventory.read_text())
        if source.get("version") != VERSION or not isinstance(source.get("objects"), list):
            parser.error("Unsupported inventory")
        seen = set()
        for item in source["objects"]:
            key = (item["bucket"], object_path(item["bucket"], item["path"]))
            if key in seen:
                parser.error("Duplicate object in inventory")
            seen.add(key)
        report = {"version": VERSION, "objects": []}
        for item in source["objects"]:
            result = (
                verify_one(spaces, item)
                if args.action == "verify"
                else migrate_one(db, SupabaseStorage(db), spaces, item, apply=args.apply)
            )
            report["objects"].append(result)
            write_report(args.output, report)
    write_report(args.output, report)
    print("Storage report written; legacy source copies retained.")
    return int(
        any(item.get("copy_state") in ("failed", "retry") for item in report.get("objects", []))
    )


if __name__ == "__main__":
    raise SystemExit(main())
