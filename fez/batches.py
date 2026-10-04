"""Private JSONL bulk jobs, processed by a trusted Zils worker (not training miners)."""

import argparse
import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, quote, urlsplit

from .api_store import identifier
from .cloud import APIError, Supabase
from .decisions import MAX_BODY, MAX_DEPTH, DecisionError, decode_body, validate_request

BUCKET = "zils-api-batches"
MAX_FILE = 25 * 1024 * 1024
MAX_RECORDS = 10000
PUBLIC_FIELDS = (
    "id",
    "status",
    "created_at",
    "deadline",
    "finished_at",
    "total",
    "completed",
    "failed",
    "error",
    "purged_at",
)
TERMINAL = {"completed", "cancelled", "failed", "expired"}


def public(row):
    return {key: row.get(key) for key in PUBLIC_FIELDS}


def record_error(custom_id, error):
    return {
        "custom_id": custom_id,
        "error": {"status": error.status, "code": error.code, "message": str(error)},
    }


def parse_input(raw, catalog, *, max_records=MAX_RECORDS):
    if not raw or len(raw) > MAX_FILE:
        raise DecisionError(413, "batch_size", "Upload must contain 1 byte to 25 MiB of JSONL.")
    rows, seen = [], set()
    for line, encoded in enumerate(raw.splitlines(), 1):
        if line > max_records:
            raise DecisionError(
                413, "batch_size", "Split the file into smaller jobs; record limit exceeded."
            )
        row = decode_body(encoded, limit=MAX_BODY + 1024, max_depth=MAX_DEPTH + 1)
        if (
            not isinstance(row, dict)
            or set(row) != {"custom_id", "body"}
            or not isinstance(row["custom_id"], str)
            or not 1 <= len(row["custom_id"]) <= 128
        ):
            raise DecisionError(
                422, "invalid_record", "Each JSONL row needs custom_id (1–128 characters) and body."
            )
        custom_id = row["custom_id"]
        if "\x00" in custom_id:
            raise DecisionError(422, "invalid_record", "custom_id cannot contain U+0000.")
        if custom_id in seen:
            raise DecisionError(
                422, "duplicate_custom_id", "Every custom_id must be unique within the job."
            )
        seen.add(custom_id)
        result, frozen, body = None, None, row["body"]
        try:
            validate_request(body)
            pending = [body]
            while pending:
                value = pending.pop()
                if isinstance(value, str) and "\x00" in value:
                    raise DecisionError(
                        422, "unsupported_character", "Bulk records cannot contain U+0000."
                    )
                if isinstance(value, dict):
                    pending.extend(value)
                    pending.extend(value.values())
                elif isinstance(value, list):
                    pending.extend(value)
            frozen = catalog.get(body["model"])
            if frozen is None:
                raise DecisionError(
                    404, "model_not_found", "Model was unavailable when the batch was submitted."
                )
        except DecisionError as error:
            result, body = record_error(custom_id, error), None
        rows.append(
            {"line": line, "custom_id": custom_id, "body": body, "frozen": frozen, "result": result}
        )
    if not rows:
        raise DecisionError(422, "empty_batch", "Upload at least one record.")
    return rows


class Batches:
    def __init__(self, db):
        self.db = db

    def row(self, owner, batch_id):
        rows = self.db.rows(
            "zils_api_batches", f"id=eq.{identifier(batch_id)}&owner_id=eq.{identifier(owner)}"
        )
        if not rows:
            raise DecisionError(404, "not_found", "Batch not found.")
        return rows[0]

    def get(self, owner, batch_id):
        return public(self.row(owner, batch_id))

    def dispatch(self, method, path, owner, body, registry):
        parsed = urlsplit(path)
        parts = parsed.path.strip("/").split("/")
        if parts == ["v1", "batches"] and method == "POST" and not parsed.query:
            if (
                not isinstance(body, dict)
                or set(body) != {"idempotency_key"}
                or not isinstance(body["idempotency_key"], str)
                or not 1 <= len(body["idempotency_key"]) <= 128
            ):
                raise DecisionError(
                    422, "invalid_request", "Provide an idempotency_key of 1–128 characters."
                )
            row = self.db.rpc(
                "zils_api_batch_create", {"p_owner": owner, "p_key": body["idempotency_key"]}
            )
            if not row or not row.get("id"):
                raise DecisionError(
                    429,
                    "batch_capacity",
                    "Account batch capacity reached; wait for jobs to finish or retained inputs to expire.",
                    retry_after=60,
                )
            result = public(row)
            if row["status"] == "uploading":
                try:
                    result["upload"] = self.db.signed(BUCKET, row["input_path"], upload=True)
                except APIError as error:
                    # Supabase refuses immutable upload URLs once the object exists.
                    # Recover the batch only after confirming its private input arrived.
                    if error.status != 409 or not self.db.exists(BUCKET, row["input_path"]):
                        raise
                result["limits"] = {"max_bytes": MAX_FILE, "max_records": MAX_RECORDS}
            return 200, result
        if len(parts) not in (3, 4) or parts[:2] != ["v1", "batches"]:
            raise DecisionError(404, "not_found", "Resource not found.")
        row = self.row(owner, parts[2])
        if len(parts) == 3 and method == "GET" and not parsed.query:
            return 200, public(row)
        action = parts[3] if len(parts) == 4 else ""
        if action in ("submit", "cancel") and method == "POST" and not parsed.query and body == {}:
            if (
                action == "submit"
                and row["status"] == "uploading"
                and not self.db.exists(BUCKET, row["input_path"])
            ):
                raise DecisionError(
                    409, "upload_missing", "Complete the immutable upload before submitting."
                )
            row = self.db.rpc(
                "zils_api_batch_action",
                {
                    "p_owner": owner,
                    "p_id": row["id"],
                    "p_action": action,
                    "p_catalog": registry.snapshot(owner) if action == "submit" else None,
                },
            )
            if not row:
                raise DecisionError(404, "not_found", "Batch not found.")
            return (
                202
                if action == "submit" and row["status"] in ("queued", "validating", "running")
                else 200
            ), public(row)
        if action == "results" and method == "GET":
            if row["purged_at"] or (
                row["finished_at"]
                and datetime.fromisoformat(row["finished_at"])
                < datetime.now(timezone.utc) - timedelta(days=7)
            ):
                raise DecisionError(410, "results_expired", "Batch results have expired.")
            if row["status"] not in TERMINAL:
                raise DecisionError(
                    409,
                    "batch_pending",
                    "Wait for a terminal batch status before downloading results.",
                )
            query = parse_qs(parsed.query, keep_blank_values=True)
            if set(query) - {"after"} or len(query.get("after", ["0"])) != 1:
                raise DecisionError(422, "invalid_cursor", "Use the returned integer cursor.")
            cursor = query.get("after", ["0"])[0]
            if not cursor.isascii() or not cursor.isdigit() or len(cursor) > 6:
                raise DecisionError(422, "invalid_cursor", "Use the returned integer cursor.")
            items = self.db.rows(
                "zils_api_batch_items",
                f"batch_id=eq.{row['id']}&line=gt.{int(cursor)}&result=not.is.null&select=line,result&order=line&limit=20",
            )
            return 200, {
                "data": [x["result"] for x in items],
                "next_cursor": items[-1]["line"] if len(items) == 20 else None,
                "status": row["status"],
            }
        raise DecisionError(404, "not_found", "Resource not found.")


class Worker:
    def __init__(self, db, gateway):
        self.db, self.gateway = db, gateway

    def work(self, batch, action, data=None):
        return self.db.rpc(
            "zils_api_batch_work",
            {
                "p_id": batch["id"],
                "p_lease": batch["lease_token"],
                "p_action": action,
                "p_data": data if data is not None else {},
            },
        )

    def cleanup(self):
        now = datetime.now(timezone.utc)
        cutoff = quote((now - timedelta(days=7)).isoformat(), safe="")
        for row in self.db.rows(
            "zils_api_batches", f"finished_at=lt.{cutoff}&purged_at=is.null&limit=100"
        ):
            self.db.request(
                "DELETE", "/storage/v1/object/" + BUCKET, {"prefixes": [row["input_path"]]}
            )
            self.db.rpc("zils_api_batch_purge", {"p_id": row["id"]})
        metadata_cutoff = quote((now - timedelta(days=30)).isoformat(), safe="")
        self.db.request(
            "DELETE",
            f"/rest/v1/zils_api_batches?finished_at=lt.{metadata_cutoff}&purged_at=not.is.null",
        )
        self.db.request("DELETE", f"/rest/v1/zils_api_usage?created_at=lt.{metadata_cutoff}")

    def once(self):
        batch = self.db.rpc("zils_api_batch_claim", {})
        if not batch or not batch.get("id"):
            return False
        stop, lost = threading.Event(), threading.Event()

        def heartbeat():
            while not stop.wait(30):
                try:
                    self.work(batch, "renew")
                except (APIError, DecisionError):
                    lost.set()
                    return

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            if batch["status"] == "validating":
                with TemporaryDirectory(prefix="zils-batch-") as tmp:
                    path = Path(tmp) / "input.jsonl"
                    self.db.download(BUCKET, batch["input_path"], path, MAX_FILE)
                    rows = parse_input(path.read_bytes(), batch["catalog"])
                # Validate every ID before persisting any row or executing any model call.
                chunk, size = [], 0
                for row in rows:
                    if lost.is_set():
                        return True
                    row = {
                        **row,
                        "body": json.dumps(row["body"], ensure_ascii=False)
                        if row["body"] is not None
                        else None,
                    }
                    encoded_size = len(json.dumps(row).encode())
                    if chunk and size + encoded_size > 512 * 1024:
                        self.work(batch, "load", chunk)
                        chunk, size = [], 0
                    chunk.append(row)
                    size += encoded_size
                if chunk:
                    self.work(batch, "load", chunk)
                self.work(batch, "ready", {"total": len(rows)})
                if all(row["result"] is not None for row in rows):
                    return True
                self.work(batch, "release")
                return True
            items = self.db.rows(
                "zils_api_batch_items",
                f"batch_id=eq.{batch['id']}&result=is.null&order=line&limit=1",
            )
            if not items:
                self.work(batch, "release")
                return True
            item = items[0]
            request_id = str(uuid.uuid4())
            delay = 0
            try:
                if lost.is_set():
                    return True
                response = self.gateway.evaluate(
                    batch["owner_id"],
                    None,
                    decode_body(item["body"].encode()),
                    request_id,
                    lane="bulk",
                    frozen=item["frozen"],
                )
            except DecisionError as error:
                if error.status in (429, 529):
                    delay = error.retry_after or 1
                elif error.status >= 500 and item["attempts"] < 2:
                    self.work(batch, "retry", {"line": item["line"]})
                    delay = 5 * 2 ** item["attempts"]
                else:
                    self.work(
                        batch,
                        "result",
                        {"line": item["line"], "result": record_error(item["custom_id"], error)},
                    )
            else:
                self.work(
                    batch,
                    "result",
                    {
                        "line": item["line"],
                        "request_id": request_id,
                        "input_tokens": response["usage"]["input_tokens"],
                        "result": {"custom_id": item["custom_id"], "response": response},
                    },
                )
            # Completion/cancellation may have cleared the lease; release is safely fenced.
            try:
                self.work(batch, "release", {"delay": delay})
            except APIError as error:
                if error.status != 409:
                    raise
            return True
        except (DecisionError, ValueError) as error:
            safe = (
                error
                if isinstance(error, DecisionError)
                else DecisionError(
                    422, "invalid_input", "Input file is invalid or exceeds its transfer limit."
                )
            )
            self.work(batch, "fail", {"code": safe.code, "message": str(safe)})
            return True
        except APIError as error:
            if error.status not in (409, 503):
                raise
            # Lost leases fence writes. Infrastructure outages leave durable state for recovery.
            return True
        finally:
            stop.set()
            thread.join(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    from .api import Gateway, Registry
    from .api_store import Store

    db = Supabase()
    worker = Worker(
        db, Gateway(Store(db), Registry(json.loads(args.registry.read_text())["models"]))
    )
    next_cleanup = 0
    while True:
        try:
            active = worker.once()
            if time.monotonic() >= next_cleanup:
                worker.cleanup()
                next_cleanup = time.monotonic() + 60
        except APIError:
            print("Bulk storage unavailable; retrying after 5 seconds.", flush=True)
            active = False
        if args.once:
            break
        if not active:
            time.sleep(5)


if __name__ == "__main__":
    main()
