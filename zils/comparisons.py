"""One consented, immutable Jev comparison per accepted model; never selects a model."""

import json
import math
import re
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

import requests

import zils

from . import benchmark, models, settings
from .cloud import DATA_BUCKET, MAX_DATA_BYTES, APIError
from .comparison_metrics import summarize
from .runtime import digest, gpu_ready, run_child

TABLE = "zils_training_comparisons"
CONSENT = "typesafe-evaluation-v1"
JEV = "jev-1.13.0"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MAX_BYTES = 5 * 1024 * 1024
MAX_CASES = 500
PUBLIC = (
    "status",
    "created_at",
    "finished_at",
    "model_id",
    "checkpoint_sha256",
    "jev_model",
    "input_sha256",
    "result",
    "error",
)


def enabled():
    return settings.get("ZILS_JEV_COMPARISON_ENABLED") == "1"


def identity(job):
    delivery = (job.get("result") or {}).get("delivery") or {}
    sha = delivery.get("sha256", "")
    if (
        job["status"] != "completed"
        or delivery.get("status") != "accepted"
        or models.job_model(job) != models.JEVK5
        or not job.get("release_prefix")
        or not re.fullmatch(r"[a-f0-9]{64}", sha)
    ):
        raise APIError(409, "An accepted JevK5 model is required before comparing with Jev.")
    return f"zils-adapter-{job['id']}-{sha}", sha


def customer(service, job, method, action, body):
    store = service.store
    query = f"job_id=eq.{job['id']}&owner_id=eq.{job['owner_id']}"
    rows = store.rows(TABLE, query)
    row = rows[0] if rows else None
    if method == "GET" and action is None:
        return {"available": enabled(), "comparison": public(row)}
    if method != "POST" or action not in (None, "submit"):
        raise APIError(404, "Unknown comparison route.")
    if not enabled():
        raise APIError(503, "Jev comparisons are not enabled on this service yet.")
    model_id, sha = identity(job)
    if action is None:
        if (
            body.get("consent_version") != CONSENT
            or body.get("allow_typesafe_export") is not True
            or body.get("unseen_examples") is not True
        ):
            raise APIError(
                400, "Confirm new, unseen examples and permission to send inputs to TypeSafe."
            )
        if row is None:
            # Ignore duplicate creates: consent, model, file and result cannot be replaced.
            store.request(
                "POST",
                f"/rest/v1/{TABLE}?on_conflict=job_id",
                {
                    "job_id": job["id"],
                    "owner_id": job["owner_id"],
                    "consent_version": CONSENT,
                    "model_id": model_id,
                    "checkpoint_sha256": sha,
                    "jev_model": JEV,
                },
                {"Prefer": "resolution=ignore-duplicates"},
            )
            row = store.rows(TABLE, query)[0]
        result = {"comparison": public(row)}
        if row["status"] == "uploading":
            path = input_path(job["id"])
            result["upload"] = (
                {"uploaded": True}
                if store.exists(DATA_BUCKET, path)
                else store.signed(DATA_BUCKET, path, upload=True)
            )
        return result
    if row is None:
        raise APIError(409, "Confirm permission before uploading comparison examples.")
    if row["status"] == "uploading":
        if not store.exists(DATA_BUCKET, input_path(job["id"])):
            raise APIError(409, "Upload the new test file first.")
        store.patch(TABLE, query + "&status=eq.uploading", {"status": "queued"})
        row = store.rows(TABLE, query)[0]
    return {"comparison": public(row)}


def public(row):
    return {key: row.get(key) for key in PUBLIC} if row else None


def input_path(job_id):
    return f"{job_id}/comparison/input.jsonl"


def prompt(case):
    return json.dumps([case["state"], case["question"]], sort_keys=True, allow_nan=False)


def validate_fresh(cases, original):
    zils.validate_cases(cases)
    if len(cases) > MAX_CASES:
        raise ValueError("Use at most 500 new test examples.")
    ids = {c["id"] for c in original}
    groups = {c["group_id"] for c in original}
    prompts = {prompt(c) for c in original}
    fresh_prompts = set()
    for case in cases:
        group = case.get("group_id")
        if not isinstance(group, str) or not group:
            raise ValueError("Every example needs a source group.")
        fingerprint = prompt(case)
        if (
            case["id"] in ids
            or group in groups
            or fingerprint in prompts
            or fingerprint in fresh_prompts
        ):
            raise ValueError("Use new examples, source groups and prompts, without duplicates.")
        fresh_prompts.add(fingerprint)
        # Keep the exact question. Never silently rewrite it for one competitor.
        question = case["question"]
        if "instructions" not in question or (
            question["type"] == "score" and len(zils.options(question)) > 10
        ):
            raise ValueError("Questions need instructions; scores support at most 10 levels.")


def jev_prediction(case, key):
    start = time.monotonic()
    # Labels, source IDs and groups are deliberately absent. No redirects or automatic retries.
    with requests.post(
        ENDPOINT,
        json={"model": JEV, "state": case["state"], "questions": {"decision": case["question"]}},
        headers={"Authorization": "Bearer " + key},
        timeout=(10, 45),
        allow_redirects=False,
    ) as response:
        if response.status_code != 200 or len(response.content) > 1024 * 1024:
            raise ValueError("Jev did not return a complete result.")
        data = response.json()
    if data.get("model") != JEV:
        raise ValueError("Jev returned a different model version.")
    answer = data["answers"]["decision"]
    kind = case["question"]["type"]
    if answer.get("type") != kind:
        raise ValueError("Jev returned a different answer type.")
    if kind == "noul":
        p = answer["noul"]
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid Jev probability.")
        probabilities = {"true": p, "false": 1 - p}
    else:
        probabilities = answer["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(
        zils.options(case["question"])
    ):
        raise ValueError("Incomplete Jev probabilities.")
    values = list(probabilities.values())
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Invalid Jev probabilities.")
    total = sum(values)
    if not 0.98 <= total <= 1.02:
        raise ValueError("Invalid Jev probability sum.")
    row = {
        "id": case["id"],
        "probabilities": {k: p / total for k, p in probabilities.items()},
        "elapsed_ms": (time.monotonic() - start) * 1000,
    }
    if kind == "choice":
        row["choice"] = answer["choice"]
        if row["choice"] not in probabilities:
            raise ValueError("Invalid Jev choice.")
    return row


def evaluate(processor, row, work):
    from .adapter_releases import publish
    from .coordinator import prepared_path
    from .jevk5 import validate_inputs

    job = processor.service.job(row["job_id"], row["owner_id"])
    if identity(job) != (row["model_id"], row["checkpoint_sha256"]):
        raise ValueError("The accepted checkpoint changed.")
    if row["consent_version"] != CONSENT or row["jev_model"] != JEV:
        raise ValueError("Comparison consent or model version changed.")
    store = processor.store
    path = work / "comparison.jsonl"
    store.download(DATA_BUCKET, input_path(job["id"]), path, MAX_BYTES)
    cases = benchmark.read_jsonl(path)
    # Check this run and its known ancestor models; attestation covers outside data and tuning.
    visited, ancestor = set(), job
    validate_fresh(cases, [])
    while ancestor:
        if ancestor["id"] in visited or len(visited) >= 32:
            raise ValueError("Invalid model ancestry.")
        visited.add(ancestor["id"])
        for split in ("train", "calibration", "test"):
            name = split + ".jsonl"
            source = work / f"{ancestor['id']}-{name}"
            store.download(DATA_BUCKET, prepared_path(ancestor, name), source, MAX_DATA_BYTES)
            if digest(source) != ancestor["manifest"]["files"][name]:
                raise ValueError("Original examples changed.")
            validate_fresh(cases, benchmark.read_jsonl(source))
        previous = (ancestor["manifest"].get("selection") or {}).get("previous")
        ancestor = processor.service.job(previous["job_id"], row["owner_id"]) if previous else None
    validate_inputs({"comparison": cases})
    release = publish(store, job["id"], work / "releases")
    if not release or release["release_id"] != row["model_id"]:
        raise ValueError("Accepted model could not be verified.")
    manifest = work / "submissions.json"
    benchmark.write_private(
        manifest, json.dumps([zils.submission(work / "releases" / row["model_id"], 0)])
    )
    report = work / "trained.json"
    run_child(
        [
            sys.executable,
            "-m",
            "zils",
            "evaluate",
            "--submissions",
            str(manifest),
            "--cases",
            str(path),
            "--base-revision",
            models.spec(models.JEVK5)["base_revision"],
            "--runner-python",
            processor.args.runtime_python,
            "--device",
            processor.args.device,
            "--timeout",
            "900",
            "--report",
            str(report),
        ],
        work / "trained.log",
        processor.args.device,
        timeout=1020,
    )
    trained = json.loads(report.read_text())["miners"][0]
    if trained["status"] != "evaluated" or trained["sha256"] != row["checkpoint_sha256"]:
        raise ValueError("Your model did not return a complete result.")
    zils.score(cases, trained["predictions"])
    comparator, deadline = [], time.monotonic() + 1200
    key = settings.required("TYPESAFE_API_KEY")
    for case in cases:
        if time.monotonic() >= deadline or datetime.now(timezone.utc) >= datetime.fromisoformat(
            row["lease_until"]
        ):
            raise ValueError("Comparison exceeded its time limit.")
        comparator.append(jev_prediction(case, key))
    result = summarize(cases, trained["predictions"], comparator)
    # A private audit copy includes full predictions; owner-facing results contain only outcomes.
    benchmark.write_private(work / "jev.json", json.dumps(comparator, allow_nan=False))
    return {"input_sha256": digest(path), "result": result}


def process_one(processor):
    if not enabled() or not settings.get("TYPESAFE_API_KEY"):
        return False
    now = datetime.now(timezone.utc)
    # An interrupted paid attempt is never silently replayed; partial results cannot declare a win.
    processor.store.patch(
        TABLE,
        "status=eq.running&lease_until=lt." + now.isoformat().replace("+00:00", "Z"),
        {
            "status": "failed",
            "finished_at": now.isoformat(),
            "error": "Comparison interrupted. No complete comparison is available; contact support.",
        },
    )
    if not gpu_ready(processor.args.device):
        return False
    rows = processor.store.rows(TABLE, "status=eq.queued&order=created_at.asc&limit=1")
    if not rows:
        return False
    token = str(uuid.uuid4())
    claimed = processor.store.patch(
        TABLE,
        f"job_id=eq.{rows[0]['job_id']}&status=eq.queued",
        {
            "status": "running",
            "lease_token": token,
            "lease_until": (now + timedelta(hours=1)).isoformat(),
        },
    )
    if not claimed:
        return False
    row = claimed[0]
    work = processor.root / row["job_id"] / ("comparison-" + token)
    try:
        work.mkdir(mode=0o700, parents=True, exist_ok=False)
        values = {**evaluate(processor, row, work), "status": "completed", "error": None}
    except Exception as error:
        print(
            f"comparison {row['job_id']} failed ({type(error).__name__})",
            file=sys.stderr,
            flush=True,
        )
        values = {
            "status": "failed",
            "error": (
                "Comparison could not finish. Check that examples are new, have valid labels and fit "
                "the model’s input limits. Contact support if the file is valid. No winner was declared."
            ),
        }
    processor.store.patch(
        TABLE,
        f"job_id=eq.{row['job_id']}&status=eq.running&lease_token=eq.{token}",
        {
            **values,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "lease_token": None,
            "lease_until": None,
        },
    )
    return True
