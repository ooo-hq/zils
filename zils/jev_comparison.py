"""Cache one supplemental TypeSafe Jev evaluation per accepted text model."""

import argparse
import json
import math
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

import zils

from . import benchmark, models
from .cloud import DATA_BUCKET, MAX_DATA_BYTES, APIError, Supabase
from .coordinator import JOBS, identifier, prepared_path
from .runtime import locked

MODEL = "jev-1.13.0"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MAX_CASES = 2000
MAX_INPUT_BYTES = 16 * 1024 * 1024


def now():
    return datetime.now(timezone.utc)


def predict(case, key):
    # Labels, case IDs, family/group metadata and training examples never leave Zils.
    payload = {"model": MODEL, "state": case["state"], "questions": {"decision": case["question"]}}
    started = time.monotonic()
    for attempt in range(3):
        with requests.post(
            ENDPOINT,
            json=payload,
            headers={"Authorization": "Bearer " + key},
            timeout=(10, 45),
            allow_redirects=False,
        ) as response:
            if response.status_code in (429, 529) and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            response.raise_for_status()
            body = response.json()
        if body.get("model") != MODEL:
            raise ValueError("Jev version changed")
        answer = body["answers"]["decision"]
        if answer["type"] != case["question"]["type"]:
            raise ValueError("Answer type changed")
        probabilities = (
            {"false": 1 - answer["noul"], "true": answer["noul"]}
            if answer["type"] == "noul"
            else answer["probabilities"]
        )
        row = {
            "id": case["id"],
            "probabilities": probabilities,
            "elapsed_ms": (time.monotonic() - started) * 1000,
        }
        zils.score([case], [row])
        tokens = body["usage"]["input_tokens"]
        if type(tokens) is not int or tokens < 0:
            raise ValueError("Invalid token usage")
        return {**row, "input_tokens": tokens}
    raise ValueError("Jev request failed")


def eligible(job):
    result = job.get("result") or {}
    return (
        job.get("status") == "completed"
        and result.get("delivery", {}).get("status") == "accepted"
        and result.get("workflow", {}).get("state") == "ready"
        and models.job_model(job) != models.IMAJEV
    )


class Comparisons:
    def __init__(self, store, state, key):
        if not key:
            raise ValueError("Set TYPESAFE_API_KEY for the comparison worker")
        self.store, self.state, self.key = store, Path(state), key
        self.offset = 0
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)

    def tick(self, job_id=None):
        query = (
            f"id=eq.{identifier(job_id)}"
            if job_id
            else "status=eq.completed&result->delivery->>status=eq.accepted&"
            "result->workflow->>state=eq.ready&"
            "or=(jev_comparison->>status.eq.pending,and(jev_comparison->>status.eq.running,"
            f"jev_comparison->>lease_until.lt.{now().isoformat()}))&order=created_at.asc,id.asc&limit=100&offset={self.offset}"
        )
        rows = self.store.rows(JOBS, query)
        self.offset = self.offset + 100 if len(rows) == 100 else 0
        for job in rows:
            if not eligible(job):
                continue
            previous = job.get("jev_comparison")
            if not previous or previous["status"] not in ("pending", "running"):
                continue
            if (
                previous["status"] == "running"
                and datetime.fromisoformat(previous["lease_until"]) > now()
            ):
                continue
            record = {
                "status": "running",
                "model": MODEL,
                "run_id": str(uuid.uuid4()),
                "started_at": (previous or {}).get("started_at", now().isoformat()),
                "lease_until": (now() + timedelta(minutes=5)).isoformat(),
                "completed_cases": (previous or {}).get("completed_cases", 0),
                "authorized_at": previous["authorized_at"],
            }
            condition = f"jev_comparison->>run_id=eq.{identifier(previous['run_id'])}"
            owner = identifier(job["owner_id"])
            job_id = identifier(job["id"])
            scope = f"id=eq.{job_id}&owner_id=eq.{owner}&status=eq.completed"
            if not self.store.patch(JOBS, scope + "&" + condition, {"jev_comparison": record}):
                continue
            claim = scope + f"&jev_comparison->>run_id=eq.{record['run_id']}"

            def save(**values):
                record.update(values, lease_until=(now() + timedelta(minutes=5)).isoformat())
                if not self.store.patch(JOBS, claim, {"jev_comparison": record}):
                    raise APIError(409, "Comparison lease changed")

            try:
                self.evaluate(job, record, save, resumed=previous["status"] == "running")
            except (APIError, OSError, ValueError, KeyError, TypeError, requests.RequestException):
                # Provider errors can contain submitted text or credentials; expose neither.
                save(status="failed", finished_at=now().isoformat())
            return True
        return False

    def evaluate(self, job, record, save, *, resumed):
        manifest = job["manifest"]
        test_hash = manifest["files"]["test.jsonl"]
        checkpoint = job["result"]["delivery"]["sha256"]
        if any(
            not re.fullmatch(r"[a-f0-9]{64}", value)
            for value in (test_hash, checkpoint, job["job_sha256"])
        ):
            raise ValueError("Invalid comparison provenance")
        count = manifest["counts"]["test"]
        if type(count) is not int or not 1 <= count <= MAX_CASES:
            save(status="skipped", reason="test_limit", finished_at=now().isoformat())
            return
        root = self.state / job["id"] / (MODEL + "-" + test_hash)
        # Losing durable state must never silently repeat previously charged requests.
        if resumed and not root.is_dir():
            raise ValueError("Comparison cache is missing")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in ("manifest.json", "test.jsonl"):
            path = root / name
            if not path.exists():
                self.store.download(DATA_BUCKET, prepared_path(job, name), path, MAX_DATA_BYTES)
        if (
            benchmark.file_hash(root / "manifest.json") != job["job_sha256"]
            or json.loads((root / "manifest.json").read_text()) != manifest
        ):
            raise ValueError("Frozen manifest changed")
        if benchmark.file_hash(root / "test.jsonl") != test_hash:
            raise ValueError("Test examples changed")
        cases = benchmark.read_jsonl(root / "test.jsonl")
        zils.validate_cases(cases)
        if len(cases) != count or any("image" in case for case in cases):
            raise ValueError("Test set differs from training evaluation")
        if (
            sum(len(json.dumps([c["state"], c["question"]]).encode()) for c in cases)
            > MAX_INPUT_BYTES
        ):
            save(status="skipped", reason="test_limit", finished_at=now().isoformat())
            return
        selected = next(
            (
                row
                for row in job["result"]["miners"]
                if row["uid"] == job["result"]["delivery"]["uid"]
            ),
            {},
        )
        accuracy = selected["accuracy"]
        if (
            type(accuracy) not in (float, int)
            or not math.isfinite(accuracy)
            or not 0 <= accuracy <= 1
        ):
            raise ValueError("Selected model accuracy is missing")
        if selected.get("cases", selected.get("count", count)) != count:
            raise ValueError("Selected model used another test set")
        save(count=count, test_sha256=test_hash, checkpoint_sha256=checkpoint)
        predictions = []
        for index, case in enumerate(cases):
            path = root / f"{index:05d}.json"
            marker = root / f"{index:05d}.started"
            if path.exists():
                row = json.loads(path.read_text())
                zils.score([case], [row])
            else:
                # An interrupted request may already be charged. Require operator review.
                benchmark.write_private(marker, "started\n")
                save(completed_cases=index)
                row = predict(case, self.key)
                temporary = path.with_suffix(".tmp")
                benchmark.write_private(temporary, json.dumps(row, allow_nan=False))
                temporary.replace(path)
            predictions.append(row)
        metrics = zils.score(cases, predictions)
        save(
            status="completed",
            completed_cases=count,
            evaluated_at=now().isoformat(),
            accuracy=metrics["accuracy"],
            brier=metrics["brier"],
            trained_accuracy=accuracy,
            input_tokens=sum(row["input_tokens"] for row in predictions),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--job")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    worker = Comparisons(Supabase(), args.state, os.environ.get("TYPESAFE_API_KEY"))
    with locked(args.state / ".comparison.lock"):
        while True:
            try:
                worker.tick(args.job)
            except (APIError, OSError, ValueError, requests.RequestException):
                print("Comparison worker unavailable; retrying later.", flush=True)
                if args.once or args.job:
                    raise
            if args.once or args.job:
                return
            time.sleep(15)


if __name__ == "__main__":
    main()
