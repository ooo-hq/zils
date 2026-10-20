"""Frozen public reference releases for the adapter regression suite, never customer data."""

import json
import math
from pathlib import Path

from . import benchmark, jobs, models, releases


def references(path, suite_jobs, incumbent_model):
    """Verify pinned releases and old splits before any GPU work; reject reused test groups."""
    path = Path(path).resolve()
    entries = json.loads(path.read_text())
    if not isinstance(entries, dict) or set(entries) != {p.name for p in suite_jobs}:
        raise ValueError("Migration references must cover every suite job exactly")
    result = {}
    for job in suite_jobs:
        entry = entries[job.name]
        if not isinstance(entry, dict) or set(entry) != {"release", "fingerprint", "splits"}:
            raise ValueError("Migration reference requires release, fingerprint and splits")
        root = (path.parent / entry["release"]).resolve()
        release = releases.read_release(root)
        if release["fingerprint"] != entry["fingerprint"]:
            raise ValueError("Frozen migration release fingerprint changed")
        if release["model"] != models.spec(incumbent_model):
            raise ValueError("Frozen migration release must use the incumbent profile")
        manifest = json.loads((root / "manifest.json").read_text())
        old = Path(path.parent / entry["splits"])
        previous = {}
        for split in ("train", "calibration", "test"):
            file = old / f"{split}.jsonl"
            if benchmark.file_hash(file) != manifest["files"][file.name]:
                raise ValueError("Frozen migration source split changed")
            previous[split] = benchmark.read_jsonl(file)
        jobs.validate_splits(previous)
        current = {s: benchmark.read_jsonl(job / f"{s}.jsonl") for s in previous}
        jobs.validate_splits(current)
        groups = {r["group_id"] for rows in previous.values() for r in rows}
        prompts = {
            json.dumps([r["state"], r["question"]], sort_keys=True)
            for rows in previous.values()
            for r in rows
        }
        ids = {r["id"] for rows in previous.values() for r in rows}
        if any(
            r["id"] in ids
            or r["group_id"] in groups
            or json.dumps([r["state"], r["question"]], sort_keys=True) in prompts
            for r in current["test"]
        ):
            raise ValueError("Migration test must be fresh to every frozen-release split")
        result[job.name] = (root, release)
    return result


def compare(baseline, candidate, release, policy):
    """Use the production gate for both strict upgrades and no-worse migrations."""
    for scored in (baseline, candidate):
        for key in ("brier", "accuracy", "skill"):
            value = scored[key]
            maximum = 2 if key == "brier" else 1
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= maximum
            ):
                raise ValueError("Migration scores must be finite and in range")
    policy = {
        **policy,
        "previous_job_id": release["job_id"],
        "base_change": True,
    }
    row = {**candidate, "status": "evaluated", "uid": 1, "sha256": candidate["adapter_sha256"]}
    baseline = {**baseline, "status": "evaluated"}
    return {
        "strict_upgrade": jobs.select(baseline, [row], policy)["status"] == "accepted",
        "migration": jobs.select(baseline, [row], {**policy, "migration": "no_regression"})[
            "status"
        ]
        == "accepted",
        "brier_improvement": baseline["brier"] - row["brier"],
    }


def summary(outcomes):
    """Every job must pass every seed; a mean cannot hide a stranded reference adapter."""
    if not outcomes or any(not seeds for seeds in outcomes.values()):
        raise ValueError("Migration report requires jobs and seeds")
    seeds = next(iter(outcomes.values())).keys()
    if any(row.keys() != seeds for row in outcomes.values()):
        raise ValueError("Every migration job must use the same seeds")
    return {
        "seed_rule": "every-job-every-seed",
        "stable_eligible": len(seeds) >= 3
        and all(r["migration"] for rows in outcomes.values() for r in rows.values()),
        "strict_upgrade_yield": sum(
            r["strict_upgrade"] for rows in outcomes.values() for r in rows.values()
        )
        / sum(len(rows) for rows in outcomes.values()),
        "jobs": outcomes,
    }
