"""Supabase-backed customer jobs and approved-worker queue. No chain writes."""

import argparse
import json
import re
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests
from bittensor_wallet import Keypair

import zils

from . import benchmark, image_jobs, jobs, models, queue_protocol, settings, version_selection
from .access import require_access
from .cloud import DATA_BUCKET, MAX_DATA_BYTES, MODEL_BUCKET, APIError, Supabase, trusted_url
from .decisions import DecisionError
from .runtime import digest, gpu_ready, locked

JOBS = "fez_training_jobs"
ASSIGNMENTS = "fez_training_assignments"
PUBLIC_FIELDS = ("id", "name", "status", "created_at", "updated_at", "error", "result")


def public_job(job):
    result = {name: job.get(name) for name in PUBLIC_FIELDS}
    data = job.get("result") or {}
    result["workflow"] = data.get("workflow")
    result["result"] = (
        {k: v for k, v in data.items() if k != "workflow"} if "delivery" in data else None
    )
    result["model"] = (
        models.spec(models.job_model(job))
        if job.get("job_sha256") or job.get("model_profile")
        else None
    )
    if models.job_model(job) == models.IMAJEV:
        from .image_metrics import public_metrics

        result["acceptance"] = job.get("acceptance")
        if result["result"] is not None:

            def aggregate(row):
                return {
                    **public_metrics(row),
                    **{k: row[k] for k in ("uid", "status") if k in row},
                    **({"count": row["cases"]} if "cases" in row else {}),
                }

            result["result"] = {
                "image_metrics_version": "zils-image-metrics/v1",
                "baseline": aggregate(data["baseline"]),
                "miners": [aggregate(row) for row in data.get("miners", [])],
                "delivery": data["delivery"],
                "weights": data.get("weights", {}),
            }
        result["image_intake"] = job.get("image_intake")
        date = (
            job.get("updated_at")
            if job["status"] in ("completed", "failed")
            else job.get("created_at")
            if job["status"] == "uploading"
            else None
        )
        result["data_expires_at"] = (
            (
                datetime.fromisoformat(date.replace("Z", "+00:00"))
                + (
                    timedelta(days=30)
                    if job["status"] in ("completed", "failed")
                    else timedelta(hours=24)
                )
            ).isoformat()
            if date
            else None
        )
    result["selection"] = (job.get("manifest") or {}).get("selection")
    return result


def identifier(value):
    try:
        if str(uuid.UUID(value)) != value:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise APIError(400, "Invalid job or lease identifier.") from None
    return value


def prepared_path(job, name):
    return f"{job['id']}/prepared/{job['job_sha256']}/{name}"


def artifact_path(assignment, name):
    return f"{assignment['job_id']}/candidates/{assignment['hotkey']}/{assignment['lease_token']}/{name}"


@contextmanager
def lease_heartbeat(renew, interval=60):
    stopped, errors = threading.Event(), []

    def keep_alive():
        while not stopped.wait(interval):
            try:
                renew()
            except Exception as error:
                errors.append(error)
                return

    renew()
    thread = threading.Thread(target=keep_alive, daemon=True)
    thread.start()

    def check_lease():
        if errors:
            raise APIError(409, "Lease renewal failed; this attempt must be retried.")

    try:
        yield check_lease
        check_lease()
    finally:
        stopped.set()
        thread.join(timeout=35)


class Service:
    def __init__(self, store, audience, model=models.KEV):
        self.store = store
        self.audience = trusted_url(audience)
        self.model = models.spec(model)

    def job(self, job_id, owner=None):
        query = f"id=eq.{identifier(job_id)}"
        if owner is not None:
            query += f"&owner_id=eq.{identifier(owner)}"
        rows = self.store.rows(JOBS, query)
        if not rows:
            raise APIError(404, "Job not found.")
        return rows[0]

    def uploads(self, job):
        if job["status"] != "uploading":
            raise APIError(409, "This job is no longer accepting dataset uploads.")
        urls = {}
        for split in ("train", "calibration", "test"):
            path = f"{job['id']}/inputs/{split}.jsonl"
            urls[split] = (
                {"uploaded": True}
                if self.store.exists(DATA_BUCKET, path)
                else self.store.signed(DATA_BUCKET, path, upload=True)
            )
        return {"job": public_job(job), "uploads": urls}

    def customer(self, method, path, token, body):
        owner = identifier(self.store.user(token))
        require_access(self.store, owner)
        if path == "/v1/jobs":
            if method == "GET":
                rows = self.store.rows(JOBS, f"owner_id=eq.{owner}&order=created_at.desc&limit=100")
                return {"jobs": [public_job(row) for row in rows]}
            if method == "POST":
                if not isinstance(body.get("name"), str) or not re.fullmatch(
                    r"[a-z0-9][a-z0-9-]{0,63}", body["name"]
                ):
                    raise APIError(
                        400, "Use a job name with 1..64 lowercase letters, digits or hyphens."
                    )
                requested = body.get("model", self.model["id"])
                selected = (
                    models.validate_spec(requested) if isinstance(requested, dict) else requested
                )
                if selected not in (self.model["id"], models.IMAJEV):
                    raise APIError(400, "This model is not enabled for new jobs.")
                profile = models.spec(selected)
                if selected == models.IMAJEV:
                    from .image_api import enabled

                    if not enabled() or not enabled("ZILS_IMAGE_TRAINING_ENABLED"):
                        raise APIError(503, "Image training is not enabled yet.")
                    image_jobs.validate_policy(body.get("acceptance"))
                else:
                    jobs.validate_policy(body.get("acceptance"))
                if "previous_job_id" in body["acceptance"]:
                    version_selection.freeze(
                        self.store,
                        {
                            "id": str(uuid.uuid4()),
                            "owner_id": owner,
                            "acceptance": body["acceptance"],
                            "model_profile": profile,
                        },
                    )
                if body.get("allow_training_data_export") is not True:
                    raise APIError(
                        400, "Permission to share training data with approved workers is required."
                    )
                intake = body.get("image_intake")
                if intake is not None:
                    if selected != models.IMAJEV:
                        raise APIError(400, "Image intake requires an image model.")
                    image_jobs.validate_intake(intake)
                job = self.store.rpc(
                    "zils_create_image_job" if intake is not None else "zils_create_profile_job",
                    {
                        "p_owner": owner,
                        "p_name": body["name"],
                        "p_acceptance": body["acceptance"],
                        "p_model": profile,
                        **({"p_intake": intake} if intake is not None else {}),
                    },
                )
                return self.uploads(job)
        match = re.fullmatch(
            r"/v1/jobs/([a-f0-9-]+)(?:/(uploads|submit|downloads|cancel|image-assets))?", path
        )
        if not match:
            raise APIError(404, "Unknown route.")
        job = self.job(match[1], owner)
        action = match[2]
        if method == "GET" and action is None:
            return {"job": public_job(job)}
        if method == "GET" and action == "image-assets":
            if models.job_model(job) != models.IMAJEV or job["status"] != "uploading":
                raise APIError(409, "This job is no longer accepting image uploads.")
            rows = self.store.rows(
                "zils_image_assets",
                f"job_id=eq.{job['id']}&owner_id=eq.{owner}&state=in.(uploading,verifying,ready)&order=created_at.desc&limit=1792",
            )
            return {
                "assets": [{k: row[k] for k in ("id", "filename", "source_sha256")} for row in rows]
            }
        if method == "POST" and action == "uploads":
            return self.uploads(job)
        if method == "POST" and action == "submit":
            if job["status"] != "uploading":
                return {"job": public_job(job)}
            if not all(
                self.store.exists(DATA_BUCKET, f"{job['id']}/inputs/{s}.jsonl")
                for s in ("train", "calibration", "test")
            ):
                raise APIError(409, "Upload all three dataset splits before submitting.")
            if models.job_model(job) == models.IMAJEV:
                ids = set()
                with tempfile.TemporaryDirectory(prefix="zils-image-submit-") as temp:
                    for split, limit in image_jobs.SPLITS.items():
                        file = Path(temp) / (split + ".jsonl")
                        self.store.download(
                            DATA_BUCKET, f"{job['id']}/inputs/{split}.jsonl", file, MAX_DATA_BYTES
                        )
                        cases = benchmark.read_jsonl(file)
                        if not 1 <= len(cases) <= limit:
                            raise APIError(400, "Image split exceeds the allowed example count.")
                        ids.update(identifier(c["image"]["asset_id"]) for c in cases)
                submitted = self.store.rpc(
                    "zils_submit_image_job",
                    {"p_owner": owner, "p_job": job["id"], "p_assets": sorted(ids)},
                )
                return {"job": public_job(submitted)}
            rows = self.store.patch(
                JOBS,
                f"id=eq.{job['id']}&status=eq.uploading",
                {"status": "validating", "updated_at": datetime.now(timezone.utc).isoformat()},
            )
            return {"job": public_job(rows[0] if rows else self.job(job["id"], owner))}
        if method == "POST" and action == "cancel":
            rows = self.store.patch(
                JOBS,
                f"id=eq.{job['id']}&status=not.in.(completed,failed)",
                {
                    "status": "failed",
                    "error": "Cancelled by customer.",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "lease_token": None,
                    "lease_until": None,
                },
            )
            return {"job": public_job(rows[0] if rows else self.job(job["id"], owner))}
        if method == "GET" and action == "downloads":
            if job["status"] != "completed" or not job.get("release_prefix"):
                raise APIError(409, "No accepted model is available for download.")
            return {
                "downloads": {
                    name: self.store.signed(MODEL_BUCKET, job["release_prefix"] + "/" + name)
                    for name in (*models.candidate_files(models.job_model(job)), "release.json")
                }
            }
        raise APIError(404, "Unknown route.")

    def assignment(self, hotkey, body, *, submitted=False):
        job_id, token = identifier(body.get("job_id")), identifier(body.get("lease_token"))
        rows = self.store.rows(
            ASSIGNMENTS, f"job_id=eq.{job_id}&hotkey=eq.{hotkey}&lease_token=eq.{token}"
        )
        if not rows:
            raise APIError(409, "Assignment is no longer available.")
        row = rows[0]
        if submitted and row["state"] == "submitted":
            return row
        if row["state"] != "leased":
            raise APIError(409, "Assignment is not accepting submissions.")
        self.store.rpc(
            "fez_renew_training", {"p_job": job_id, "p_hotkey": hotkey, "p_token": token}
        )
        return row

    def worker(self, path, message):
        hotkey, body = queue_protocol.verify(message, self.audience, path, self.store)
        if path == "/v1/workers/profiles":
            rows = self.store.rows("zils_worker_profiles", f"hotkey=eq.{hotkey}&enabled=eq.true")
            return {
                "profiles": [
                    {
                        key: row[key]
                        for key in (
                            "profile_id",
                            "profile_sha256",
                            "runtime_sha256",
                            "min_free_mib",
                        )
                    }
                    for row in rows
                    if row["profile_id"] in models.SPECS
                    and all(
                        row[key] == value
                        for key, value in models.profile_identity(row["profile_id"]).items()
                    )
                ]
            }
        if path == "/v1/workers/claim":
            profiles = body.get("supported_profiles")
            if profiles is None:
                if body:
                    raise APIError(400, "Invalid legacy claim request.")
                row = self.store.rpc("fez_claim_training", {"p_hotkey": hotkey})
            else:
                if (
                    set(body) != {"supported_profiles"}
                    or not isinstance(profiles, list)
                    or not 1 <= len(profiles) <= len(models.SPECS)
                    or any(not isinstance(x, str) or x not in models.SPECS for x in profiles)
                    or len(set(profiles)) != len(profiles)
                ):
                    raise APIError(400, "Advertise distinct installed model profiles.")
                row = self.store.rpc(
                    "zils_claim_profile_training",
                    {"p_hotkey": hotkey, "p_supported_profiles": profiles},
                )
            if not row:
                return {"assignment": None}
            job = self.job(row["job_id"])
            model = models.job_model(job)
            if model not in (profiles if profiles is not None else [models.KEV, models.JEVK5]):
                raise APIError(409, "Worker assignment profile differs from its claim.")
            resources = {}
            if model == models.IMAJEV:
                qualified = self.store.rows(
                    "zils_worker_profiles",
                    f"hotkey=eq.{hotkey}&profile_id=eq.{model}&enabled=eq.true",
                )
                if len(qualified) != 1 or any(
                    qualified[0][k] != v for k, v in models.profile_identity(model).items()
                ):
                    raise APIError(409, "Worker profile qualification changed.")
                resources = {
                    "min_free_mib": qualified[0]["min_free_mib"],
                    "max_seconds": qualified[0]["evidence"]["max_seconds"],
                }
            return {
                "assignment": {
                    **row,
                    **resources,
                    "model": models.spec(models.job_model(job)),
                    "base_revision": models.spec(models.job_model(job))["base_revision"],
                    "initial_sha256": job["initial_sha256"],
                    "job_sha256": job["job_sha256"],
                    "training_sha256": job["manifest"]["files"]["miner-training.jsonl"],
                    "training": self.store.signed(
                        DATA_BUCKET, prepared_path(job, "miner-training.jsonl")
                    ),
                }
            }
        if path not in (
            "/v1/workers/image-downloads",
            "/v1/workers/renew",
            "/v1/workers/uploads",
            "/v1/workers/submit",
            "/v1/workers/fail",
            "/v1/workers/defer",
        ):
            raise APIError(404, "Unknown worker route.")
        if path.endswith("/fail"):
            self.store.rpc(
                "fez_fail_training",
                {
                    "p_job": identifier(body.get("job_id")),
                    "p_hotkey": hotkey,
                    "p_token": identifier(body.get("lease_token")),
                },
            )
            return {"status": "released"}
        row = self.assignment(hotkey, body, submitted=path.endswith("/submit"))
        if path.endswith("/image-downloads"):
            from .image_store import ImageStore

            job = self.job(row["job_id"])
            manifest = image_jobs.verify_binding(job)
            requested = body.get("asset_ids")
            if (
                not isinstance(requested, list)
                or not 1 <= len(requested) <= 100
                or any(not isinstance(aid, str) for aid in requested)
                or len(set(requested)) != len(requested)
            ):
                raise APIError(400, "Request 1–100 distinct training image IDs.")
            allowed = manifest["assets"]
            if any(aid not in allowed or allowed[aid]["split"] != "train" for aid in requested):
                raise APIError(404, "Training image is unavailable.")
            images = ImageStore(self.store)
            # Validate the entire batch before the first signed grant.
            for aid in requested:
                live = images._get(job["owner_id"], aid, "training")
                if live["state"] != "ready" or any(
                    live[k] != allowed[aid][k] for k in image_jobs.ASSET_FIELDS
                ):
                    raise APIError(404, "Training image is unavailable.")
            return {
                "images": [
                    images.read_reference(job["owner_id"], aid, "training") for aid in requested
                ]
            }
        if path.endswith("/defer"):
            # The signed worker can release only its own live lease. Capacity waiting is
            # not a failed training attempt; the job's original deadline remains in force.
            self.store.patch(
                ASSIGNMENTS,
                f"job_id=eq.{row['job_id']}&hotkey=eq.{hotkey}&lease_token=eq.{row['lease_token']}&state=eq.leased",
                {
                    "state": "ready",
                    "lease_token": None,
                    "lease_until": None,
                    "attempts": max(0, row["attempts"] - 1),
                },
            )
            return {"status": "deferred"}
        if path.endswith("/renew"):
            return {"status": "renewed"}
        if path.endswith("/uploads"):
            return {
                "uploads": {
                    name: (
                        {"uploaded": True}
                        if self.store.exists(MODEL_BUCKET, artifact_path(row, name))
                        else self.store.signed(MODEL_BUCKET, artifact_path(row, name), upload=True)
                    )
                    for name in models.candidate_files(models.job_model(self.job(row["job_id"])))
                }
            }
        sha = body.get("sha256")
        if not isinstance(sha, str) or not re.fullmatch("[a-f0-9]{64}", sha):
            raise APIError(400, "Invalid checkpoint hash.")
        if row["state"] != "submitted":
            with tempfile.TemporaryDirectory(prefix="zils-submission-") as tmp:
                self.fetch(row, Path(tmp), sha)
        self.store.rpc(
            "fez_submit_training",
            {
                "p_job": row["job_id"],
                "p_hotkey": hotkey,
                "p_token": row["lease_token"],
                "p_sha256": sha,
            },
        )
        return {"status": "submitted"}

    def fetch(self, row, destination, expected):
        total = 0
        model = models.job_model(self.job(row["job_id"]))
        for name in models.candidate_files(model):
            total += self.store.download(
                MODEL_BUCKET,
                artifact_path(row, name),
                destination / name,
                zils.MAX_ARTIFACT_BYTES - total,
            )
        if zils.checkpoint_hash(destination) != expected:
            raise ValueError("uploaded checkpoint does not match its signed hash")
        if models.checkpoint_model(destination) != model or (
            model in (models.JEVK5, models.IMAJEV)
            and models.metadata(destination)["kind"] != "adapter"
        ):
            raise ValueError("candidate differs from the job's model contract")


def handler(service, origin):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # URLs and tokens must never appear in access logs.

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def send(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            if self.headers.get("Origin") == origin:
                self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            if self.headers.get("Origin") != origin:
                self.send(403, {"error": "Origin is not allowed."})
                return
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Vary", "Origin")
            self.end_headers()

        def dispatch(self):
            try:
                if self.headers.get("Origin") not in (None, origin):
                    raise APIError(403, "Origin is not allowed.")
                body = {}
                if self.command == "POST":
                    length = int(self.headers.get("Content-Length", "-1"))
                    if not 0 <= length <= 64 * 1024 or self.headers.get("Transfer-Encoding"):
                        raise APIError(400, "Invalid request size.")
                    body = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(body, dict):
                        raise APIError(400, "Request must be a JSON object.")
                if self.path == "/v1/config" and self.command == "GET":
                    result = {"model": service.model}
                    if settings.get("ZILS_IMAGES_ENABLED") == "1":
                        from .image_contract import IMAGE_CAPABILITIES

                        result["image"] = {
                            "model": models.spec(models.IMAJEV),
                            "capabilities": IMAGE_CAPABILITIES,
                            "training_enabled": settings.get("ZILS_IMAGE_TRAINING_ENABLED") == "1",
                            "limits": {
                                "train": 1024,
                                "calibration": 256,
                                "test": 512,
                                "max_source_bytes": 10 * 1024**2,
                                "max_input_tokens": 4096,
                            },
                        }
                elif self.path.startswith("/v1/workers/") and self.command == "POST":
                    result = service.worker(self.path, body)
                else:
                    auth = self.headers.get("Authorization", "")
                    if not auth.startswith("Bearer "):
                        raise APIError(401, "Sign in to continue.")
                    result = service.customer(self.command, self.path, auth[7:], body)
                self.send(200, result)
            except (APIError, DecisionError) as error:
                self.send(error.status, {"error": str(error)})
            except (ValueError, TypeError, KeyError):
                self.send(400, {"error": "Invalid request or dataset/checkpoint content."})
            except (OSError, requests.RequestException):
                self.send(503, {"error": "Service temporarily unavailable; please retry."})
            except Exception:
                self.send(500, {"error": "Unexpected service error."})

        do_GET = dispatch
        do_POST = dispatch

    return Handler


class Processor:
    def __init__(self, service, root, reference, args):
        self.service, self.store = service, service.store
        self.root, self.reference, self.args = Path(root), Path(reference), args
        self.references = {
            models.checkpoint_model(path): Path(path)
            for path in [reference, *getattr(args, "additional_reference", [])]
        }
        if models.spec(models.checkpoint_model(self.reference)) != service.model:
            raise ValueError("processor reference differs from the active service model")
        for path in self.references.values():
            zils.checkpoint_hash(path)
            if models.checkpoint_model(path) == models.IMAJEV:
                from .imajev import verify_starting_checkpoint

                verify_starting_checkpoint(path)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def renew(self, job):
        until = (datetime.now(timezone.utc) + timedelta(minutes=20)).isoformat()
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        rows = self.store.patch(
            JOBS,
            f"id=eq.{job['id']}&lease_token=eq.{job['lease_token']}&status=eq.{job['status']}&lease_until=gt.{now}",
            {"lease_until": until},
        )
        if not rows:
            raise APIError(409, "Processing lease expired.")

    def tick(self):
        for stage in ("validating", "evaluating"):
            if stage == "evaluating" and not gpu_ready(self.args.device):
                continue
            job = self.store.rpc("fez_claim_processing", {"p_stage": stage})
            if not job or not job.get("id"):
                continue
            work = self.root / job["id"] / job["lease_token"]
            work.mkdir(mode=0o700, parents=True, exist_ok=False)
            try:
                with lease_heartbeat(lambda: self.renew(job)):
                    values = (
                        self.prepare(job, work)
                        if stage == "validating"
                        else self.evaluate(job, work)
                    )
                status = "awaiting_approval" if stage == "validating" else "completed"
            except (APIError, requests.RequestException) as error:
                # Infrastructure outages must not become a model-quality failure.
                if not isinstance(error, APIError) or error.status != 409:
                    self.store.patch(
                        JOBS,
                        f"id=eq.{job['id']}&lease_token=eq.{job['lease_token']}",
                        {"lease_until": datetime.now(timezone.utc).isoformat()},
                    )
                print(
                    f"job {job['id']} processing interrupted; retry after lease recovery.",
                    file=sys.stderr,
                    flush=True,
                )
                return True
            except Exception as error:
                # Local logs use categories only; do not leak dataset contents or signed URLs.
                print(
                    f"job {job['id']} {stage} failed ({type(error).__name__})",
                    file=sys.stderr,
                    flush=True,
                )
                values = {
                    "error": (
                        "Dataset validation failed. Check fields, labels and split overlap. JevK5 supports up to 16 outcomes and 2,048 tokens per example; shorten long examples."
                        if stage == "validating"
                        and isinstance(error, (ValueError, KeyError, TypeError))
                        else "Processing failed. An operator must inspect the job before retrying."
                    )
                }
                status = "failed"
            self.store.rpc(
                "fez_finish_processing",
                {
                    "p_job": job["id"],
                    "p_token": job["lease_token"],
                    "p_status": status,
                    "p_values": values,
                },
            )
            return True
        return False

    def prepare(self, job, work):
        model = (
            models.job_model(job)
            if job.get("model_profile")
            else models.checkpoint_model(self.reference)
        )
        reference = self.references.get(model)
        if reference is None:
            raise APIError(503, "The frozen model reference is not available yet.")
        splits = {}
        for split in ("train", "calibration", "test"):
            path = work / (split + ".jsonl")
            self.store.download(
                DATA_BUCKET, f"{job['id']}/inputs/{split}.jsonl", path, MAX_DATA_BYTES
            )
            splits[split] = benchmark.read_jsonl(path)
        if model == models.JEVK5:
            from .jevk5 import validate_inputs

            validate_inputs(splits)
        data = work / "benchmark"
        if model == models.IMAJEV:
            from .image_store import ImageStore

            images = ImageStore(self.store)
            assets = {
                case["image"]["asset_id"]: images._get(
                    job["owner_id"], case["image"]["asset_id"], "training"
                )
                for rows in splits.values()
                for case in rows
            }
            manifest = image_jobs.build(
                data,
                {**job, "selection": version_selection.freeze(self.store, job)},
                splits,
                assets,
                job["acceptance"],
            )
            previous = manifest["selection"]["previous"]
            if previous:
                parent = self.service.job(previous["job_id"], job["owner_id"])
                image_jobs.verify_binding(parent)
                image_jobs.validate_predecessor(manifest, parent["manifest"])
        else:
            manifest = jobs.build(
                data,
                job["id"],
                splits,
                job["acceptance"],
                allow_training_data_export=True,
                model=model,
                selection=version_selection.freeze(self.store, job)
                if model == models.JEVK5
                else None,
            )
        initial = zils.checkpoint_hash(reference)
        sha = digest(data / "manifest.json")
        snapshot = {**job, "job_sha256": sha}
        for name in (*benchmark.FILES, "manifest.json"):
            path = prepared_path(snapshot, name)
            if not self.store.exists(DATA_BUCKET, path):
                self.store.upload(DATA_BUCKET, path, data / name)
        return {"manifest": manifest, "job_sha256": sha, "initial_sha256": initial}

    def evaluate(self, job, work):
        from .validator import evaluate_round

        model = models.job_model(job)
        if model == models.IMAJEV and not gpu_ready(self.args.device, model=model):
            raise APIError(503, "Waiting for image evaluation capacity.")
        data = work / "benchmark"
        data.mkdir(mode=0o700)
        for name in (*benchmark.FILES, "manifest.json"):
            self.store.download(DATA_BUCKET, prepared_path(job, name), data / name, MAX_DATA_BYTES)
        if digest(data / "manifest.json") != job["job_sha256"]:
            raise ValueError("job manifest changed")
        if model == models.IMAJEV:
            from .image_store import BUCKET as IMAGE_BUCKET, ImageStore

            manifest = image_jobs.verify_binding(job)
            benchmark.audit(data)
            cache = work / "images"
            cache.mkdir(mode=0o700)
            image_store = ImageStore(self.store)
            for aid, asset in manifest["assets"].items():
                if asset["split"] not in ("calibration", "test"):
                    continue
                live = image_store._get(job["owner_id"], aid, "training")
                if live["state"] != "ready" or any(
                    live[k] != asset[k] for k in image_jobs.ASSET_FIELDS
                ):
                    raise ValueError("Evaluation image binding changed")
                target = cache / (asset["canonical_sha256"] + ".png")
                if not target.exists():
                    self.store.download(IMAGE_BUCKET, live["canonical_path"], target, 10 * 1024**2)
                if (
                    digest(target) != asset["canonical_sha256"]
                    or target.stat().st_size != asset["canonical_bytes"]
                ):
                    raise ValueError("Evaluation image bytes changed")
        reference = self.references[model]
        if zils.checkpoint_hash(reference) != job["initial_sha256"]:
            raise ValueError("reference changed")
        zils.stage(zils.submission(reference, 0), work / "reference")
        selection = job["manifest"].get("selection")
        if selection and selection["previous"]:
            from .adapter_releases import publish

            previous = selection["previous"]
            # Re-verify ownership and immutable job provenance before staging any weights.
            if version_selection.freeze(self.store, job) != selection:
                raise ValueError("Previous model identity changed")
            release = publish(self.store, previous["job_id"], work / "previous-releases")
            if (
                release is None
                or release["release_id"] != previous["model_id"]
                or release["checkpoint_sha256"] != previous["sha256"]
            ):
                raise ValueError("Previous model checkpoint changed")
            zils.stage(
                zils.submission(work / "previous-releases" / release["release_id"], 0),
                work / "comparison",
            )
        assignments = self.store.rows(ASSIGNMENTS, f"job_id=eq.{job['id']}")
        members = {str(a["uid"]): {"hotkey": a["hotkey"]} for a in assignments}
        registry = {
            a["uid"]: {
                "claim": {
                    "uid": a["uid"],
                    "sha256": a["sha256"],
                    "endpoint": "storage",
                    "assignment": a,
                }
            }
            for a in assignments
            if a["state"] == "submitted"
        }
        config = {
            "job_id": job["id"],
            "job_sha256": job["job_sha256"],
            "benchmark_sha256": job["job_sha256"],
            "initial_sha256": job["initial_sha256"],
            "base_revision": models.spec(models.job_model(job))["base_revision"],
            "members": members,
        }

        def fetch(claim, destination, **kwargs):
            destination.mkdir(mode=0o700)
            self.service.fetch(claim["assignment"], destination, claim["sha256"])

        report = evaluate_round(config, work, work, registry, self.args, fetch_checkpoint=fetch)
        fields = (
            "uid",
            "status",
            "accuracy",
            "brier",
            "skill",
            "confident_errors",
            "median_ms",
            "p95_ms",
        )
        if model == models.IMAJEV:
            from .image_metrics import PUBLIC_FIELDS

            fields += PUBLIC_FIELDS
        result = {
            "model": models.spec(model),
            "delivery": {k: v for k, v in report["delivery"].items() if k != "checkpoint"},
            "baseline": {k: report["baseline"][k] for k in fields if k in report["baseline"]},
            "miners": [{k: r[k] for k in fields if k in r} for r in report["miners"]],
            "weights": report["weights"],
        }
        if selection:
            result["selection"] = report["selection"]
            result["baseline_reference_sha256"] = report["baseline_reference_sha256"]
        prefix = None
        if result["delivery"]["status"] == "accepted":
            release = work / report["delivery"]["checkpoint"]
            prefix = f"{job['id']}/releases/{job['lease_token']}"
            for name in (*models.artifact_files(release), "release.json"):
                self.store.upload(MODEL_BUCKET, prefix + "/" + name, release / name)
        return {"result": result, "release_prefix": prefix}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    server = sub.add_parser("serve")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8910)
    processor = sub.add_parser("process")
    processor.add_argument("--state", required=True)
    processor.add_argument("--reference", default="models/reference")
    processor.add_argument("--additional-reference", action="append", default=[])
    processor.add_argument("--runtime-python", default=sys.executable)
    processor.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    processor.add_argument("--once", action="store_true")
    worker = sub.add_parser("worker")
    worker.add_argument("--hotkey", required=True)
    worker.add_argument("--uid", type=int, required=True)
    worker.add_argument("--disable", action="store_true")
    approve = sub.add_parser("approve")
    approve.add_argument("--job", required=True)
    approve.add_argument("--hotkeys", nargs="+", required=True)
    args = parser.parse_args()
    try:
        service = Service(
            Supabase(),
            settings.required("ZILS_TRAINING_API_URL"),
            settings.get("ZILS_TRAINING_MODEL", models.KEV),
        )
        if args.command == "serve":
            origin = trusted_url(settings.required("ZILS_WEB_ORIGIN"))
            with ThreadingHTTPServer((args.host, args.port), handler(service, origin)) as server:
                print(
                    "Zils training API listening; deploy behind HTTPS for remote access.",
                    flush=True,
                )
                server.serve_forever()
        elif args.command == "process":
            state = Path(args.state)
            state.mkdir(mode=0o700, parents=True, exist_ok=True)
            with locked(state / "processor.lock"):
                engine = Processor(service, state, args.reference, args)
                while True:
                    try:
                        engine.tick()
                    except (APIError, requests.RequestException):
                        if args.once:
                            raise
                        print(
                            "Processor connection interrupted; retrying.",
                            file=sys.stderr,
                            flush=True,
                        )
                    if args.once:
                        break
                    time.sleep(5)
        elif args.command == "worker":
            Keypair(ss58_address=args.hotkey)
            zils.weight_vector([{"uid": args.uid, "skill": 0.0}])
            service.store.request(
                "POST",
                "/rest/v1/fez_training_workers?on_conflict=hotkey",
                {"hotkey": args.hotkey, "uid": args.uid, "enabled": not args.disable},
                {"Prefer": "resolution=merge-duplicates"},
            )
            print("Worker registration updated.")
        else:
            service.store.rpc(
                "fez_approve_training_job",
                {"p_job": identifier(args.job), "p_hotkeys": args.hotkeys},
            )
            print("Job assigned to the approved workers.")
    except KeyboardInterrupt:
        pass
    except (APIError, ValueError, OSError, KeyError) as error:
        parser.exit(1, f"zils coordinator: {error}\n")


if __name__ == "__main__":
    main()
