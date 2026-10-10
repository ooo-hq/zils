"""Deterministic miner selection from operator/validator evidence, never self grades."""

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median

from . import models

POLICY = "zils-miner-routing/v1"
CONTEXT_KEYS = frozenset(
    (
        "model",
        "profile_sha256",
        "runtime_sha256",
        "trainer_sha256",
        "rubric",
        "benchmark_sha256",
        "band",
    )
)
WINDOW = timedelta(days=30)
FAILURES = {"invalid_artifact", "worker_failure", "abandoned"}
NEUTRAL = {
    "capacity_deferred",
    "validator_error",
    "infrastructure_error",
    "cancelled",
    "pending_review",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def timestamp(value):
    if not isinstance(value, (datetime, str)):
        raise ValueError("timestamp must include a timezone")
    result = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(value.replace("Z", "+00:00"))
    )
    if result.tzinfo is None:
        raise ValueError("timezone required")
    return result.astimezone(UTC)


def finite(value, minimum=0, maximum=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum


def trainer_identity():
    root = Path(__file__).resolve().parents[1]
    return digest(
        {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("zils/jevk5.py", "requirements/model.txt", "requirements/jevk5-source.txt")
        }
    )


def workload_profile(token_lengths, *, model):
    if (
        model != models.JEVK5
        or not token_lengths
        or any(type(n) is not int or not 1 <= n <= 2048 for n in token_lengths)
    ):
        raise ValueError("workload requires encoded JevK5 training inputs of 1..2048 tokens")
    maximum = max(token_lengths)
    return {
        "model": model,
        **models.profile_identity(model),
        "examples": len(token_lengths),
        "total_tokens": sum(token_lengths),
        "max_tokens": maximum,
        "band": next(f"tokens-{n}" for n in (512, 1024, 2048) if maximum <= n),
    }


def validate_workload(value, *, model, examples):
    if not isinstance(value, dict):
        raise ValueError("missing workload")
    if set(value) != {
        "model",
        "profile_sha256",
        "runtime_sha256",
        "examples",
        "total_tokens",
        "max_tokens",
        "band",
    }:
        raise ValueError("invalid workload fields")
    if value["model"] != model or any(
        value[k] != v for k, v in models.profile_identity(model).items()
    ):
        raise ValueError("workload model mismatch")
    count, total, maximum = (value[k] for k in ("examples", "total_tokens", "max_tokens"))
    if any(type(n) is not int or n < 1 for n in (count, total, maximum)) or count != examples:
        raise ValueError("invalid workload counts")
    if maximum > 2048 or not maximum + count - 1 <= total <= count * maximum:
        raise ValueError("invalid workload tokens")
    if value["band"] != workload_profile([maximum], model=model)["band"]:
        raise ValueError("invalid workload band")
    return value


def valid_context(context):
    if (
        not isinstance(context, dict)
        or set(context) != CONTEXT_KEYS
        or context["model"] != models.JEVK5
    ):
        return False
    return (
        all(
            isinstance(context[k], str)
            and len(context[k]) == 64
            and all(c in "0123456789abcdef" for c in context[k])
            for k in ("profile_sha256", "runtime_sha256", "trainer_sha256", "benchmark_sha256")
        )
        and isinstance(context["rubric"], str)
        and 1 <= len(context["rubric"]) <= 128
        and context["band"] in {"tokens-512", "tokens-1024", "tokens-2048"}
    )


def usable_reports(reports, context, now):
    """Caller reads these only from the private, append-only evidence table."""
    found = {}
    if not valid_context(context):
        return []
    for report in reports:
        try:
            if report["context"] != context or not now - WINDOW <= timestamp(
                report["verified_at"]
            ) <= now < timestamp(report["expires_at"]):
                continue
            capacity = report["capacity"]
            if (
                report["artifact_valid"] is not True
                or not isinstance(report["id"], str)
                or not finite(report["seconds_per_token"], minimum=1e-9)
                or any(
                    type(capacity[k]) is not int or capacity[k] <= 0
                    for k in ("examples", "total_tokens", "max_tokens")
                )
            ):
                continue
            if report.get("kind") != "capacity":
                if (
                    not all(
                        finite(report[k], maximum=2) for k in ("baseline_brier", "candidate_brier")
                    )
                    or not finite(report["uniform_brier"], minimum=1e-9, maximum=2)
                    or not finite(report["accuracy"], maximum=1)
                    or not finite(report["min_accuracy"], maximum=1)
                    or not finite(report["quality_floor"], minimum=-2, maximum=2)
                ):
                    continue
            found[report["id"]] = report
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return sorted(
        found.values(), key=lambda r: (timestamp(r["verified_at"]), r["id"]), reverse=True
    )


def qualification_grade(reports, context, now):
    current = [r for r in usable_reports(reports, context, now) if r.get("kind") != "capacity"][:5]
    improvements = [
        (r["baseline_brier"] - r["candidate_brier"]) / r["uniform_brier"] for r in current
    ]
    # Decimal-looking measurements must not lose a whole band to binary rounding.
    band = math.floor(round(100 * median(improvements), 10)) if improvements else None
    passes_floor = (
        bool(current)
        and median(improvements) >= max(r["quality_floor"] for r in current)
        and median(r["accuracy"] for r in current) >= max(r["min_accuracy"] for r in current)
    )
    expired = (
        any(r.get("context") == context and r.get("kind") != "capacity" for r in reports)
        and not current
    )
    return {
        "status": "qualified"
        if len(current) >= 3 and passes_floor
        else "expired"
        if expired
        else "provisional",
        "quality_band": band,
        "quality_runs": len(current),
        "evidence_ids": [r["id"] for r in current],
        "expires_at": min((r["expires_at"] for r in current), default=None),
    }


def performance_grade(attempts, context, now):
    recent = {}
    for attempt in attempts:
        try:
            ctx = attempt["context"]
            if any(
                ctx[k] != context[k]
                for k in ("model", "profile_sha256", "runtime_sha256", "trainer_sha256", "band")
            ):
                continue
            if now - WINDOW <= timestamp(attempt["finished_at"]) <= now and attempt[
                "outcome"
            ] in FAILURES | NEUTRAL | {"valid"}:
                recent[attempt["id"]] = attempt
        except (KeyError, TypeError, ValueError):
            continue
    rows = sorted(
        recent.values(), key=lambda a: (timestamp(a["finished_at"]), a["id"]), reverse=True
    )[:50]
    successes = sum(a["outcome"] == "valid" for a in rows)
    failures = sum(a["outcome"] in FAILURES for a in rows)
    rates = []
    for a in rows:
        if a["outcome"] == "valid":
            try:
                duration = (
                    timestamp(a["submitted_at"]) - timestamp(a["started_at"])
                ).total_seconds()
                if finite(duration, minimum=1e-9) and finite(a["total_tokens"], minimum=1):
                    rates.append(duration / a["total_tokens"])
            except (KeyError, TypeError, ValueError):
                pass
    rates.sort()
    return {
        "reliability": (successes + 1) / (successes + failures + 2),
        "successes": successes,
        "failures": failures,
        "neutral": len(rows) - successes - failures,
        "p90_seconds_per_token": rates[math.ceil(0.9 * len(rates)) - 1] if rates else None,
    }


def rank_candidates(job, workers, now):
    now = timestamp(now)
    context = job.get("context") or {}
    workload = job.get("workload")
    candidates, exclusions = [], []
    invalid_job = None
    try:
        validate_workload(workload, model=context["model"], examples=workload["examples"])
        if (
            not valid_context(context)
            or workload["band"] != context["band"]
            or job.get("consent") is not True
        ):
            raise ValueError("invalid job")
        remaining = (timestamp(job["deadline"]) - now).total_seconds()
    except (ValueError, TypeError, KeyError):
        invalid_job, remaining = "unverified_workload_or_consent", 0
    for worker in workers:
        reason = invalid_job
        hotkey = worker["hotkey"]
        reports = usable_reports(worker.get("reports", []), context, now)
        quality = qualification_grade(worker.get("reports", []), context, now)
        performance = performance_grade(worker.get("attempts", []), context, now)
        checks = [
            (worker.get("enabled") is True, "disabled"),
            (worker.get("approved") is True, "not_in_pool"),
            (bool(worker.get("resource_id")), "unmapped_resource"),
            (
                worker.get("profile")
                == {k: context.get(k) for k in ("profile_sha256", "runtime_sha256")},
                "profile_mismatch",
            ),
        ]
        try:
            fresh = (
                timedelta(0) <= now - timestamp(worker.get("received_at")) < timedelta(seconds=45)
            )
            cooling = worker.get("cooldown_until") and timestamp(worker["cooldown_until"]) > now
        except (ValueError, TypeError, AttributeError):
            fresh, cooling = False, True
        checks += [
            (fresh, "offline"),
            (not worker.get("reserved"), "busy"),
            (worker.get("ready") is True, "not_ready"),
            (not cooling, "cooldown"),
        ]
        qualification = job.get("purpose") == "qualification"
        checks += [
            (
                bool(reports) if qualification else quality["status"] == "qualified",
                quality["status"],
            )
        ]
        if not reason:
            reason = next((message for passed, message in checks if not passed), None)
        if not reason and qualification and quality["status"] == "qualified":
            reason = "already_qualified"
        envelope = [
            r
            for r in reports
            if workload
            and all(
                r["capacity"][k] >= workload[k] for k in ("examples", "total_tokens", "max_tokens")
            )
        ]
        if not reason and not envelope:
            reason = "capacity"
        estimate = None
        if not reason:
            rate = max(
                max(r["seconds_per_token"] for r in envelope),
                performance["p90_seconds_per_token"] or 0,
            )
            estimate = rate * workload["total_tokens"]
            if estimate > 3600 or estimate + 60 > remaining:
                reason = "deadline"
        if reason:
            exclusions.append({"hotkey": hotkey, "reason": reason})
        else:
            candidates.append(
                {
                    "hotkey": hotkey,
                    "resource_id": worker["resource_id"],
                    **quality,
                    **performance,
                    "estimated_seconds": estimate,
                    "evidence_ids": sorted({r["id"] for r in reports}),
                    "received_at": worker["received_at"],
                    "last_assigned_at": worker.get("last_assigned_at") or "",
                    "last_qualification_at": worker.get("last_qualification_at") or "",
                }
            )
    if job.get("purpose") == "qualification":
        candidates.sort(
            key=lambda c: (c["status"] == "qualified", c["last_qualification_at"], c["hotkey"])
        )
    else:
        candidates.sort(
            key=lambda c: (
                -c["quality_band"],
                -c["reliability"],
                c["estimated_seconds"],
                c["last_assigned_at"],
                c["hotkey"],
            )
        )
    result = {
        "evaluated_at": now.isoformat(),
        "policy_version": POLICY,
        "job_id": job["id"],
        "job_sha256": job["job_sha256"],
        "context": context,
        "purpose": job.get("purpose", "customer"),
        "selected_hotkey": candidates[0]["hotkey"] if candidates else None,
        "candidates": candidates,
        "exclusions": sorted(exclusions, key=lambda c: c["hotkey"]),
    }
    return {**result, "snapshot_sha256": digest(result)}


def evaluation_observations(job, attempts, assignments, report, *, qualification=None):
    """Bind evaluator validity to the uploaded artifact, before calibration changes it."""
    if (
        report.get("job_sha256") != job["job_sha256"]
        or report.get("baseline", {}).get("status") != "evaluated"
    ):
        raise ValueError("Evaluation does not match the frozen job and baseline")
    if qualification is not None:
        if benchmark_binding(job) != qualification["benchmark_sha256"]:
            raise ValueError("Qualification dataset differs from its approved benchmark")
    observations = []
    for attempt in attempts:
        assignment = next(
            (
                a
                for a in assignments
                if a["hotkey"] == attempt["hotkey"]
                and a.get("lease_token") == attempt["lease_token"]
                and a["state"] == "submitted"
            ),
            None,
        )
        if assignment is None or attempt.get("finished_at") is not None:
            continue
        uid = assignment["uid"]
        submitted = report.get("submitted", {})
        if (
            submitted.get(uid, submitted.get(str(uid))) != attempt["sha256"]
            or assignment["sha256"] != attempt["sha256"]
        ):
            raise ValueError("Evaluation artifact does not match the attempt")
        row = next((r for r in report["miners"] if r["uid"] == uid), {})
        observations.append(
            {
                "hotkey": attempt["hotkey"],
                "lease_token": attempt["lease_token"],
                "job_sha256": job["job_sha256"],
                "sha256": attempt["sha256"],
                "outcome": "valid" if row.get("status") == "evaluated" else "pending_review",
            }
        )
        if qualification is not None and observations[-1]["outcome"] == "valid":
            if (
                not all(
                    finite(value, maximum=2)
                    for value in (report["baseline"]["brier"], row["brier"])
                )
                or not finite(row["uniform_brier"], minimum=1e-9, maximum=2)
                or not finite(row["accuracy"], maximum=1)
                or row["cases"] != job["manifest"]["counts"]["test"]
            ):
                raise ValueError("Invalid qualification metrics")
            observations[-1]["qualification"] = {
                "baseline_brier": report["baseline"]["brier"],
                "candidate_brier": row["brier"],
                "uniform_brier": row["uniform_brier"],
                "accuracy": row["accuracy"],
                "cases": row["cases"],
                "calibration_sha256": job["manifest"]["files"]["calibration.jsonl"],
                "evaluated_sha256": row["sha256"],
            }
    return observations


def benchmark_binding(job):
    manifest = job["manifest"]
    if models.job_model(job) != models.JEVK5 or manifest.get("selection", {}).get("previous"):
        raise ValueError("Qualification requires the pinned base, not a customer predecessor")
    return digest(
        {k: manifest[k] for k in ("model", "files", "counts", "workload")}
        | {"initial_sha256": job["initial_sha256"]}
    )
