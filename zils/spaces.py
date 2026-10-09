"""Private Spaces objects; only trusted services complete immutable multipart uploads."""

import hashlib
import re
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from . import settings
from .cloud import APIError, download, upload
from .storage import LIMITS, expired, fingerprint, object_path, timestamp


def configured_client():
    import boto3
    from botocore.config import Config

    endpoint = settings.required("ZILS_SPACES_ENDPOINT")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(r"[a-z][a-z0-9]+\.digitaloceanspaces\.com", parsed.hostname or "")
    ):
        raise ValueError("Set ZILS_SPACES_ENDPOINT to the exact regional HTTPS origin")
    bucket = settings.required("ZILS_SPACES_BUCKET")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", bucket):
        raise ValueError("Set ZILS_SPACES_BUCKET to a valid private bucket name")
    # A lost CreateMultipartUpload reply must never allocate another upload on the same key.
    client = boto3.client(
        "s3",
        endpoint_url=endpoint.rstrip("/"),
        region_name=settings.required("ZILS_SPACES_REGION"),
        aws_access_key_id=settings.required("ZILS_SPACES_ACCESS_KEY_ID"),
        aws_secret_access_key=settings.required("ZILS_SPACES_SECRET_ACCESS_KEY"),
        config=Config(
            signature_version="s3v4",
            connect_timeout=10,
            read_timeout=60,
            retries={"total_max_attempts": 1},
            s3={"addressing_style": "virtual"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    return client, bucket


class SpacesStorage:
    def __init__(self, catalog, client, physical_bucket):
        self.catalog, self.client, self.bucket = catalog, client, physical_bucket

    def _call(self, method, **values):
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            return getattr(self.client, method)(**values)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code in ("NoSuchKey", "NoSuchUpload", "404", "NotFound"):
                raise APIError(404, "Storage object is unavailable.") from None
            raise APIError(503, "Storage is unavailable; please retry.") from None
        except BotoCoreError:
            raise APIError(503, "Storage is unavailable; please retry.") from None

    def _location(self, row):
        if row["provider"] != "spaces" or row["physical_bucket"] != self.bucket:
            raise APIError(503, "Storage location is not configured.")
        expected = f"objects/{row['generation']}/{row['bucket']}/{row['path']}"
        if row["physical_key"] != expected:
            raise APIError(503, "Storage location could not be verified.")
        return {"Bucket": self.bucket, "Key": expected}

    def _change(self, row, action, token=None, **values):
        return self.catalog.change(
            row["bucket"],
            row["path"],
            action,
            token or row["token"],
            {"generation": row["generation"], **values},
        )

    def _allocate(self, bucket, path):
        row = self.catalog.get(bucket, path)
        if row and row["state"] in ("ready", "sealing", "deleting", "deleted"):
            raise APIError(409, "Uploaded files cannot be replaced.")
        if row and row["legacy_readable"]:
            raise APIError(409, "Storage copy is in progress; retry shortly.")
        if row and row["state"] == "uploading":
            if not (expired(row["lease_until"]) and expired(row["grant_expires_at"])):
                return row
            if self.exists(bucket, path):
                raise APIError(409, "Uploaded files cannot be replaced.")
        if row and not expired(row["lease_until"]):
            raise APIError(503, "Upload allocation is in progress; retry shortly.")
        row = self.catalog.change(
            bucket,
            path,
            "allocate",
            str(uuid.uuid4()),
            {
                "provider": "spaces",
                "physical_bucket": self.bucket,
                "max_bytes": LIMITS[bucket],
            },
        )
        created = self._call(
            "create_multipart_upload",
            **self._location(row),
            ACL="private",
            ContentType="application/octet-stream",
            Metadata={"zils-generation": row["generation"]},
        )
        return self._change(row, "bind", upload_id=created["UploadId"])

    def signed(self, bucket, path, *, upload=False):
        object_path(bucket, path)
        if upload:
            row = self._allocate(bucket, path)
            row = self._change(row, "grant", seconds=600)
            url = self.client.generate_presigned_url(
                "upload_part",
                Params={
                    **self._location(row),
                    "UploadId": row["upload_id"],
                    "PartNumber": 1,
                },
                ExpiresIn=600,
                HttpMethod="PUT",
            )
            return {
                "url": url,
                "method": "PUT",
                "provider": "spaces",
                "expires_at": row["grant_expires_at"],
                "headers": {"Content-Type": "application/octet-stream"},
            }
        row = self._ready(bucket, path)
        if row is None:
            raise APIError(404, "Storage object is unavailable.")
        url = self.client.generate_presigned_url(
            "get_object", Params=self._location(row), ExpiresIn=600
        )
        return {
            "url": url,
            "provider": "spaces",
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=600)).isoformat(),
        }

    def _head(self, row):
        try:
            return self._call("head_object", **self._location(row))
        except APIError as error:
            if error.status == 404:
                return None
            raise

    def _verified_head(self, row, head):
        part_etag = row.get("part_etag") or ""
        if not re.fullmatch(r'"[a-fA-F0-9]{32}"', part_etag):
            raise APIError(503, "Storage completion could not be verified.")
        expected = (
            hashlib.md5(bytes.fromhex(part_etag.strip('"')), usedforsecurity=False).hexdigest()
            + "-1"
        )
        if (
            not head
            or head.get("Metadata", {}).get("zils-generation") != row["generation"]
            or head.get("ContentLength") != row["part_size"]
            or head.get("ETag", "").strip('"').lower() != expected
        ):
            raise APIError(503, "Storage completion could not be verified.")

    def _ready(self, bucket, path):
        row = self.catalog.get(bucket, path)
        if not row or row["state"] in ("allocating", "deleting", "deleted"):
            return None
        self._location(row)
        if row["state"] == "ready":
            return row
        if row["legacy_readable"]:
            raise APIError(409, "Storage copy is awaiting verification.")
        if row["state"] == "sealing":
            head = self._head(row)
            if head:
                self._verified_head(row, head)
                if expired(row["lease_until"]):
                    row = self._change(
                        row,
                        "seal",
                        token=str(uuid.uuid4()),
                        part_etag=row["part_etag"],
                        part_size=row["part_size"],
                    )
                return self._change(row, "commit", size_bytes=row["part_size"])
            if not expired(row["lease_until"]):
                raise APIError(503, "Storage completion is in progress; please retry.")
        try:
            listing = self._call("list_parts", **self._location(row), UploadId=row["upload_id"])
        except APIError as error:
            if (
                error.status == 404
                and expired(row["grant_expires_at"])
                and expired(row["lease_until"])
                and not self._head(row)
            ):
                return None
            raise APIError(503, "Upload could not be verified; please retry.") from None
        parts = listing.get("Parts", [])
        if not parts and not listing.get("IsTruncated"):
            return None
        if (
            listing.get("IsTruncated")
            or len(parts) != 1
            or parts[0].get("PartNumber") != 1
            or not 0 < parts[0].get("Size", 0) <= row["max_bytes"]
        ):
            self._call("abort_multipart_upload", **self._location(row), UploadId=row["upload_id"])
            raise APIError(422, "Uploaded object exceeds its permitted size or part layout.")
        part = parts[0]
        row = self._change(
            row, "seal", token=str(uuid.uuid4()), part_etag=part["ETag"], part_size=part["Size"]
        )
        try:
            self._call(
                "complete_multipart_upload",
                **self._location(row),
                UploadId=row["upload_id"],
                MultipartUpload={"Parts": [{"PartNumber": 1, "ETag": row["part_etag"]}]},
            )
        except APIError:
            # Completion may have succeeded even when the reply was lost.
            head = self._head(row)
            if not head:
                raise APIError(503, "Storage completion is uncertain; please retry.") from None
        else:
            head = self._head(row)
        self._verified_head(row, head)
        return self._change(row, "commit", size_bytes=row["part_size"])

    def exists(self, bucket, path):
        return self._ready(bucket, path) is not None

    def download(self, bucket, path, destination, limit, *, max_seconds=600):
        return download(
            self.signed(bucket, path)["url"], destination, limit, max_seconds=max_seconds
        )

    def upload(self, bucket, path, source):
        descriptor = self.signed(bucket, path, upload=True)
        upload(descriptor["url"], source, descriptor["headers"], descriptor["method"])
        if not self.exists(bucket, path):
            raise APIError(503, "Upload has not completed; please retry.")

    @contextmanager
    def _copy_lease(self, row):
        stop, failures = threading.Event(), []

        def renew():
            while not stop.wait(60):
                try:
                    self._change(row, "renew")
                except Exception:
                    failures.append(True)
                    return

        def check():
            if failures:
                raise APIError(409, "Storage copy lease was lost.")
            self._change(row, "renew")

        check()
        thread = threading.Thread(target=renew, daemon=True)
        thread.start()
        try:
            yield check
        finally:
            stop.set()
            thread.join()

    def copy_verified(self, bucket, path, source: Path, sha256: str, token: str) -> dict:
        row = self.catalog.get(bucket, path)
        if (
            not row
            or row["token"] != token
            or not row["legacy_readable"]
            or row["sha256"] != sha256
        ):
            raise APIError(409, "Storage copy changed; retry.")
        with self._copy_lease(row) as check:
            size, actual = fingerprint(source)
            if actual != sha256 or size != row["size_bytes"] or not 0 < size <= row["max_bytes"]:
                raise APIError(422, "Storage source changed.")
            if row["state"] == "allocating":
                created = self._call(
                    "create_multipart_upload",
                    **self._location(row),
                    ACL="private",
                    ContentType="application/octet-stream",
                    Metadata={"zils-generation": row["generation"]},
                )
                row = self._change(row, "bind", upload_id=created["UploadId"])
            if row["state"] == "uploading":
                with source.open("rb") as stream:
                    part = self._call(
                        "upload_part",
                        **self._location(row),
                        UploadId=row["upload_id"],
                        PartNumber=1,
                        Body=stream,
                    )
                check()
                row = self._change(row, "seal", part_etag=part["ETag"], part_size=size)
            if row["state"] != "sealing":
                raise APIError(409, "Storage copy changed; retry.")
            head = self._head(row)
            if not head:
                try:
                    self._call(
                        "complete_multipart_upload",
                        **self._location(row),
                        UploadId=row["upload_id"],
                        MultipartUpload={"Parts": [{"PartNumber": 1, "ETag": row["part_etag"]}]},
                    )
                except APIError:
                    if not self._head(row):
                        raise
                head = self._head(row)
            self._verified_head(row, head)
            with tempfile.TemporaryDirectory(prefix="zils-copy-verify-") as directory:
                target = Path(directory) / "object"
                url = self.client.generate_presigned_url(
                    "get_object", Params=self._location(row), ExpiresIn=600
                )
                download(url, target, size, max_seconds=600)
                if fingerprint(target) != (size, sha256):
                    raise APIError(422, "Storage copy checksum mismatch.")
            check()
        return self._change(row, "commit", size_bytes=size, sha256=sha256)

    def _cleanup(self, row):
        if not expired(row["cleanup_after"]):
            return False
        if row.get("upload_id"):
            try:
                self._call(
                    "abort_multipart_upload", **self._location(row), UploadId=row["upload_id"]
                )
            except APIError as error:
                if error.status != 404:
                    raise
        self._call("delete_object", **self._location(row))
        self._change(row, "deleted")
        return True

    def remove(self, bucket, paths):
        if not isinstance(paths, list) or not 1 <= len(paths) <= 100:
            raise ValueError("remove requires 1..100 exact object paths")
        for path in paths:
            row = self.catalog.change(bucket, path, "delete", str(uuid.uuid4()))
            if row["state"] == "deleting":
                self._cleanup(row)

    def reap(self, limit=100, *, dry_run=False, eligible=None):
        removed, abandoned = 0, 0
        for row in self.catalog.pending(limit):
            if row["provider"] == "spaces" and row["state"] == "deleting":
                if expired(row["cleanup_after"]):
                    if not dry_run:
                        row = self._change(row, "delete", token=str(uuid.uuid4()))
                        self._cleanup(row)
                    removed += 1
            elif row["provider"] == "spaces" and eligible and eligible(row):
                now = datetime.now(timezone.utc)
                if (
                    timestamp(row["updated_at"]) > now - timedelta(hours=24)
                    or max(timestamp(row["lease_until"]), timestamp(row["grant_expires_at"]))
                    > now - timedelta(seconds=300)
                    or self._head(row)
                ):
                    continue
                if not dry_run:
                    self._change(row, "expire")
                    if row["upload_id"]:
                        try:
                            self._call(
                                "abort_multipart_upload",
                                **self._location(row),
                                UploadId=row["upload_id"],
                            )
                        except APIError as error:
                            if error.status != 404:
                                raise
                abandoned += 1
        orphaned = 0
        for method, field, date_field in (
            ("list_multipart_uploads", "Uploads", "Initiated"),
            ("list_objects_v2", "Contents", "LastModified"),
        ):
            cursor = {}
            while True:
                result = self._call(method, Bucket=self.bucket, Prefix="objects/", **cursor)
                for item in result.get(field, []):
                    if orphaned >= limit:
                        break
                    if not self._orphan_safe(item["Key"], item[date_field]):
                        continue
                    location = {"Bucket": self.bucket, "Key": item["Key"]}
                    if dry_run:
                        orphaned += 1
                        continue
                    if field == "Uploads":
                        try:
                            self._call(
                                "abort_multipart_upload", **location, UploadId=item["UploadId"]
                            )
                        except APIError as error:
                            if error.status != 404:
                                raise
                    else:
                        self._call("delete_object", **location)
                    orphaned += 1
                if orphaned >= limit or not result.get("IsTruncated"):
                    break
                if field == "Uploads":
                    cursor = {
                        "KeyMarker": result["NextKeyMarker"],
                        "UploadIdMarker": result["NextUploadIdMarker"],
                    }
                else:
                    cursor = {"ContinuationToken": result["NextContinuationToken"]}
        return {
            "removed": removed,
            "orphaned": orphaned,
            "abandoned": abandoned,
            "dry_run": dry_run,
        }

    def _orphan_safe(self, key, modified):
        if modified >= datetime.now(timezone.utc) - timedelta(hours=24):
            return False
        try:
            prefix, generation, bucket, path = key.split("/", 3)
            if prefix != "objects" or str(uuid.UUID(generation)) != generation:
                return False
            object_path(bucket, path)
        except ValueError:
            return False
        row = self.catalog.get(bucket, path)
        if row and row["generation"] == generation:
            return row["state"] == "deleted" and expired(row["cleanup_after"])
        retired = self.catalog.db.rows(
            "zils_storage_retired", f"generation=eq.{generation}&limit=1"
        )
        return not retired or timestamp(retired[0]["safe_after"]) < datetime.now(timezone.utc)
