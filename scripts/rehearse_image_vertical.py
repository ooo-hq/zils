"""Run an explicitly configured isolated image pipeline; retain redacted evidence."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from zils import models
from zils.image_jobs import validate_policy


def create_run_directory(destination: str | Path) -> Path:
    root = Path(destination)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    return root


def verify_health_identity(reply, expected):
    entries = reply.get("models", [])
    if isinstance(entries, dict):
        entries = list(entries.values())
    for row in [reply, *entries]:
        if row.get("fingerprint") == expected:
            return {key: row.get(key) for key in ("fingerprint", "release_id")}
    raise ValueError("Runtime identity changed")


def rehearse(config: dict, dataset: Path, out: Path) -> dict:
    root = create_run_directory(out)
    started = time.monotonic()
    report = {"version": "zils-image-rehearsal/v1", "status": "running", "steps": []}
    deadline = started + min(float(config.get("max_seconds", 300)), 1800)
    created = []
    job_id = None
    text_before = None

    def record(name, **values):
        report["steps"].append(
            {"name": name, "elapsed_seconds": time.monotonic() - started, **values}
        )
        save()

    def save():
        path = root / "report.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        path.chmod(0o600)

    def token(name):
        return os.environ[config[name]]

    def call(
        method, url, *, who="owner_token_env", body=None, statuses=(200,), timeout=30, retry=False
    ):
        # Reads and explicit idempotent operations may recover a lost response.
        # Never replay creation, submission, or a billable prediction automatically.
        attempts = 3 if method == "GET" or retry else 1
        for attempt in range(attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Rehearsal deadline")
            status = None
            try:
                response = requests.request(
                    method,
                    url,
                    headers={"Authorization": "Bearer " + token(who)},
                    json=body,
                    timeout=min(timeout, remaining),
                    allow_redirects=False,
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt + 1 == attempts:
                    raise
            else:
                try:
                    status = response.status_code
                    if status in statuses:
                        return response.json() if response.content else {}
                    if status not in (502, 503, 504) or attempt + 1 == attempts:
                        record("request_failed", method=method, status=status)
                        raise RuntimeError("Unexpected rehearsal response")
                finally:
                    response.close()
            record("transient_retry", method=method, status=status, attempt=attempt + 1)
            time.sleep(min(2**attempt, max(0, deadline - time.monotonic())))

    def health(which):
        value = config[which]
        reply = call("GET", value["url"], who=value["token_config"])
        return verify_health_identity(reply, value["fingerprint"])

    def upload(slot, data):
        parsed = urlsplit(slot["url"])
        if (parsed.scheme, parsed.netloc) != storage_origin or parsed.username or parsed.password:
            raise ValueError("Unexpected storage destination")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Rehearsal deadline")
        response = requests.put(
            slot["url"],
            headers=slot["headers"],
            data=data,
            timeout=min(30, remaining),
            allow_redirects=False,
        )
        if not response.ok:
            raise RuntimeError("Private upload failed")

    def photo(path, purpose, jid=None):
        raw = path.read_bytes()
        slot = call(
            "POST",
            api + "/v1/image-assets",
            body={
                "purpose": purpose,
                **({"job_id": jid} if jid else {}),
                "filename": path.name,
                "source_bytes": len(raw),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
            },
            statuses=(200, 201),
        )
        aid = slot["asset"]["id"]
        created.append(aid)
        upload(slot["upload"], raw)
        ready = call("POST", api + f"/v1/image-assets/{aid}/complete", body={}, retry=True)
        if ready["state"] != "ready":
            raise ValueError("Photo was not finalized")
        call(
            "POST",
            api + f"/v1/image-assets/{aid}/complete",
            who="other_token_env",
            body={},
            statuses=(404,),
        )
        return aid, ready

    try:
        if config.get("isolated") is not True or config.get("owner_id") == config.get(
            "other_owner_id"
        ):
            raise ValueError("Explicit isolated services and two owners required")
        api, coordinator = config["api_url"].rstrip("/"), config["coordinator_url"].rstrip("/")
        from zils.cloud import trusted_url

        for endpoint in (api, coordinator, config["storage_url"]):
            trusted_url(endpoint)
        storage = urlsplit(config["storage_url"])
        storage_origin = storage.scheme, storage.netloc
        policy = config["acceptance"]
        validate_policy(policy)
        rows, catalog, total = {}, {}, 0
        snapshots = root / "sources"
        snapshots.mkdir(mode=0o700)
        for split, maximum in (("train", 1024), ("calibration", 256), ("test", 512), ("fresh", 2)):
            rows[split] = [
                json.loads(line)
                for line in (Path(dataset) / (split + ".jsonl")).read_text().splitlines()
                if line.strip()
            ]
            if not 1 <= len(rows[split]) <= maximum:
                raise ValueError("Invalid rehearsal split size")
            for row in rows[split]:
                source = (Path(dataset) / row["image"]).resolve()
                if (
                    not source.is_relative_to(Path(dataset).resolve())
                    or source.stat().st_size > 10 * 1024**2
                ):
                    raise ValueError("Invalid source path or size")
                raw = source.read_bytes()
                sha = hashlib.sha256(raw).hexdigest()
                if sha not in catalog:
                    total += len(raw)
                    if total > 1024**3:
                        raise ValueError("Rehearsal image bound exceeded")
                    (snapshots / sha).write_bytes(raw)
                    catalog[sha] = len(raw)
                row["image"] = sha
        frozen = json.dumps({"rows": rows, "acceptance": policy})
        (root / "frozen.json").write_text(frozen)
        report.update(
            dataset_sha256=hashlib.sha256(frozen.encode()).hexdigest(),
            policy_sha256=hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
        )
        record(
            "freeze", counts={name: len(value) for name, value in rows.items()}, source_bytes=total
        )
        text_before = health("text_runtime")
        health("image_runtime")
        record("runtime_identities", text=text_before, image=config["image_runtime"]["fingerprint"])
        created_job = call(
            "POST",
            coordinator + "/v1/jobs",
            body={
                "name": "image-rehearsal",
                "model": models.IMAJEV,
                "acceptance": policy,
                "allow_training_data_export": True,
            },
        )
        job_id = created_job["job"]["id"]
        report["job_id"] = job_id
        call("GET", coordinator + "/v1/jobs/" + job_id, who="other_token_env", statuses=(404,))
        for split in ("train", "calibration", "test"):
            data = []
            for row in rows[split]:
                aid, _ = photo(snapshots / row["image"], "training", job_id)
                data.append({**row, "image": {"asset_id": aid}})
            upload(
                created_job["uploads"][split],
                ("".join(json.dumps(row) + "\n" for row in data)).encode(),
            )
            record("uploaded_" + split, count=len(data), other_owner_denied=True)
        call("POST", coordinator + f"/v1/jobs/{job_id}/submit", body={})
        record("submitted")
        while True:
            job = call("GET", coordinator + "/v1/jobs/" + job_id)["job"]
            if job["status"] in ("completed", "failed"):
                if job["status"] == "failed":
                    raise RuntimeError("Training did not complete")
                if job["result"]["delivery"]["status"] != "accepted" or (
                    job.get("workflow") or {}
                ).get("state") in ("ready", "activation_failed", "needs_review"):
                    break
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
        report["evaluation"] = job["result"]
        report["workflow"] = job.get("workflow")
        record("evaluated", outcome=job["result"]["delivery"]["status"])
        accepted = job["result"]["delivery"]["status"] == "accepted"
        ready = (job.get("workflow") or {}).get("state") == "ready"
        if accepted and not ready:
            report["serving"] = {"verified": False, "reason": "activation_not_ready"}
        if accepted and ready:
            model_id = job["workflow"]["model_id"]
            mine = call("GET", api + "/v1/image-models")["models"]
            theirs = call("GET", api + "/v1/image-models", who="other_token_env")["models"]
            if any(row["name"] == model_id for row in theirs):
                raise ValueError("Private model leaked")
            model = next(row for row in mine if row["name"] == model_id)
            image_id, _ = photo(snapshots / rows["fresh"][0]["image"], "prediction")
            body = {
                "model": model_id,
                "state": rows["fresh"][0]["state"],
                "questions": {"inspection": model["task"]["question"]},
                "images": [{"asset_id": image_id}],
            }
            call(
                "POST",
                api + "/v1/image-decisions",
                who="other_token_env",
                body=body,
                statuses=(404,),
            )
            prediction = call("POST", api + "/v1/image-decisions", body=body, timeout=60)
            report["serving"] = {
                "verified": True,
                "model_id": model_id,
                "other_owner_denied": True,
                "prediction": prediction,
            }
            record("private_prediction")
        elif not accepted:
            report["serving"] = {"verified": False, "reason": "no_qualifying_model"}
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=type(error).__name__)
        raise
    finally:
        # Never touch an existing job or a live service. Keep training evidence for inspection.
        report["created_assets"] = created
        report["total_seconds"] = time.monotonic() - started
        if text_before:
            try:
                report["text_unchanged"] = health("text_runtime") == text_before
                if not report["text_unchanged"]:
                    report["status"] = "failed"
            except Exception:
                report.update(status="failed", text_unchanged=False)
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "dataset", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = rehearse(json.loads(args.config.read_text()), args.dataset, args.out)
    print(json.dumps({key: result[key] for key in ("status", "total_seconds")}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
