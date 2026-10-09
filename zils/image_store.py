"""Owner-scoped immutable image uploads backed by fenced database transitions."""

import hashlib
import re
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .api_store import identifier
from .cloud import APIError, Supabase
from .decisions import DecisionError, invalid
from .image_assets import MAX_CANONICAL_BYTES, MAX_ENCODED_BYTES, canonicalize

BUCKET = "zils-images"


def not_found():
    return DecisionError(404, "not_found", "Image is unavailable.")


def public_asset(row):
    value = {key: row[key] for key in ("id", "state", "expires_at")}
    if row.get("canonical_sha256"):
        value.update(sha256=row["canonical_sha256"], width=row["width"], height=row["height"])
    return value


class ImageStore:
    def __init__(self, db=None):
        self.db = db or Supabase()

    def create(self, owner, purpose, job_id, filename, source_bytes, source_sha256):
        owner = identifier(owner)
        if (
            purpose not in ("prediction", "training")
            or (purpose == "training") != (job_id is not None)
            or not isinstance(filename, str)
            or not 1 <= len(filename) <= 255
            or any(ord(c) < 32 for c in filename)
            or type(source_bytes) is not int
            or not 1 <= source_bytes <= MAX_ENCODED_BYTES
            or not isinstance(source_sha256, str)
            or not re.fullmatch("[a-f0-9]{64}", source_sha256)
        ):
            raise invalid(["body"])
        row = self.db.rpc(
            "zils_image_create",
            {
                "p_owner": owner,
                "p_purpose": purpose,
                "p_job": identifier(job_id) if job_id else None,
                "p_filename": filename,
                "p_source_bytes": source_bytes,
                "p_source_sha256": source_sha256,
            },
        )
        if row is None:
            raise not_found()
        if row.get("error") == "limited":
            raise DecisionError(
                429,
                "image_limit",
                "Image upload allowance reached; remove unused images or retry later.",
            )
        signed = self.db.signed(BUCKET, row["source_path"], upload=True)
        self.db.rpc(
            "zils_image_grant",
            {"p_owner": owner, "p_asset": row["id"], "p_expires": signed["expires_at"]},
        )
        # A cancellation while Storage issued the grant must not restore access.
        self._get(owner, row["id"])
        return {"asset": public_asset(row), "upload": signed}

    def resume(self, owner, asset_id):
        row = self._get(owner, asset_id)
        if row["state"] == "ready" or self.db.exists(BUCKET, row["source_path"]):
            return {"asset": public_asset(row), "uploaded": True}
        if row["state"] != "uploading":
            raise DecisionError(
                409, "image_verifying", "Image verification is in progress; retry shortly."
            )
        if not row.get("uploadable"):
            raise not_found()
        signed = self.db.signed(BUCKET, row["source_path"], upload=True)
        self.db.rpc(
            "zils_image_grant",
            {"p_owner": owner, "p_asset": asset_id, "p_expires": signed["expires_at"]},
        )
        self._get(owner, asset_id)
        return {"asset": public_asset(row), "uploaded": False, "upload": signed}

    def _get(self, owner, asset_id, purpose=None):
        row = self.db.rpc(
            "zils_image_get", {"p_owner": identifier(owner), "p_asset": identifier(asset_id)}
        )
        if not row or (purpose is not None and row["purpose"] != purpose):
            raise not_found()
        return row

    def complete(self, owner, asset_id):
        owner, asset_id = identifier(owner), identifier(asset_id)
        request_token = str(uuid.uuid4())
        try:
            return self._complete(owner, asset_id, request_token)
        except APIError:
            # The claim may have committed even if its response was lost. Release
            # only this request's token; completed images and other owners survive.
            try:
                self.db.rpc(
                    "zils_image_release_finalize",
                    {"p_owner": owner, "p_asset": asset_id, "p_token": request_token},
                )
            except APIError:
                # A sustained outage still falls back to the existing lease expiry.
                pass
            raise

    def _complete(self, owner, asset_id, request_token):
        row = self._get(owner, asset_id)
        if row["state"] == "ready":
            return public_asset(row)
        lease = self.db.rpc(
            "zils_image_claim_finalize_request",
            {"p_owner": owner, "p_asset": asset_id, "p_token": request_token},
        )
        if not lease:
            self._get(owner, asset_id)
            raise DecisionError(
                409, "image_verifying", "Image verification is already running; retry shortly."
            )
        if lease["state"] == "ready":
            return public_asset(lease)
        try:
            with tempfile.TemporaryDirectory(prefix="zils-image-") as temp:
                source, canonical = Path(temp) / "source", Path(temp) / "canonical.png"
                self.db.download(
                    BUCKET, lease["source_path"], source, MAX_ENCODED_BYTES, max_seconds=45
                )
                raw = source.read_bytes()
                if (
                    len(raw) != lease["source_bytes"]
                    or hashlib.sha256(raw).hexdigest() != lease["source_sha256"]
                ):
                    raise ValueError("source bytes changed")
                image = canonicalize(raw)
                canonical.write_bytes(image.data)
                if self.db.exists(BUCKET, lease["canonical_path"]):
                    # Resume a crash after the immutable write, never overwrite it.
                    previous = Path(temp) / "previous.png"
                    self.db.download(
                        BUCKET,
                        lease["canonical_path"],
                        previous,
                        MAX_CANONICAL_BYTES,
                        max_seconds=45,
                    )
                    if hashlib.sha256(previous.read_bytes()).hexdigest() != image.sha256:
                        raise ValueError("canonical bytes changed")
                else:
                    self.db.upload(BUCKET, lease["canonical_path"], canonical)
                result = self.db.rpc(
                    "zils_image_finish",
                    {
                        "p_owner": owner,
                        "p_asset": asset_id,
                        "p_token": lease["finalize_token"],
                        "p_sha256": image.sha256,
                        "p_pixels": image.pixel_sha256,
                        "p_bytes": len(image.data),
                        "p_width": image.width,
                        "p_height": image.height,
                    },
                )
        except ValueError:
            self.db.rpc(
                "zils_image_fail",
                {"p_owner": owner, "p_asset": asset_id, "p_token": lease["finalize_token"]},
            )
            raise DecisionError(
                422, "invalid_image", "Use a complete still JPEG or PNG within the image limits."
            ) from None
        if not result:
            raise not_found()
        return public_asset(result)

    def resolve(self, owner, asset_id, purpose=None):
        row = self._get(owner, asset_id, purpose)
        if row["state"] != "ready":
            raise not_found()
        return public_asset(row)

    def read_reference(self, owner, asset_id, purpose=None):
        """Trusted runtime envelope. Never return this through a public asset endpoint."""
        row = self._get(owner, asset_id, purpose)
        if row["state"] != "ready":
            raise not_found()
        deadline = datetime.now(timezone.utc) + timedelta(minutes=10)
        if row["purpose"] == "prediction":
            deadline = min(
                deadline, datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
            )
        if row.get("read_until"):
            deadline = min(
                deadline, datetime.fromisoformat(row["read_until"].replace("Z", "+00:00"))
            )
        signed = self.db.signed(BUCKET, row["canonical_path"])
        return {
            **public_asset(row),
            "url": signed["url"],
            "expires_at": deadline.isoformat(),
            "bytes": row["canonical_bytes"],
            "pixel_sha256": row["pixel_sha256"],
            "preprocessor": row["preprocessor"],
        }

    def delete_unused(self, owner, asset_id):
        if not self.db.rpc(
            "zils_image_delete", {"p_owner": identifier(owner), "p_asset": identifier(asset_id)}
        ):
            raise not_found()

    def cleanup(self, now=None, limit=100):
        # Database time is authoritative; caller's clock never advances retention.
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("cleanup limit must be between 1 and 100")
        rows = self.db.rpc("zils_image_cleanup_claim", {"p_limit": limit})
        removed = 0
        for row in rows:
            self.db.remove(BUCKET, [row["source_path"], row["canonical_path"]])
            removed += bool(
                self.db.rpc(
                    "zils_image_cleanup_finish",
                    {
                        "p_asset": row["id"],
                        "p_token": row["cleanup_token"],
                    },
                )
            )
        return {"claimed": len(rows), "removed": removed}
