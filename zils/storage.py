"""Private logical object identities, independent of authentication and file hosts."""

import re
from datetime import datetime, timezone

from .cloud import APIError

LIMITS = {
    "fez-training-data": 128 * 1024**2,
    "fez-training-models": 512 * 1024**2,
    "zils-images": 10 * 1024**2,
    "zils-api-batches": 25 * 1024**2,
}


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
