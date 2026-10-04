"""Freeze customer decision datasets for one job per approved-worker fleet."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import fez

from . import benchmark, models

VERSION = "fez-customer-job/v1"


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {"min_accuracy", "min_brier_improvement"}:
        raise ValueError("acceptance requires min_accuracy and min_brier_improvement")
    for name, maximum in (("min_accuracy", 1), ("min_brier_improvement", 2)):
        value = policy[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"invalid {name}")


def validate_splits(splits):
    if set(splits) != {"train", "calibration", "test"}:
        raise ValueError("require train, calibration and test splits")
    ids, groups, prompts = set(), {}, {}
    for split, cases in splits.items():
        fez.validate_cases(cases)
        for case in cases:
            group = case.get("group_id")
            if not isinstance(group, str) or not group:
                raise ValueError("every case needs a group_id identifying related source records")
            if case["id"] in ids:
                raise ValueError("case id overlaps splits")
            ids.add(case["id"])
            if groups.setdefault(group, split) != split:
                raise ValueError("related source group overlaps splits")
            fingerprint = json.dumps([case["state"], case["question"]], sort_keys=True)
            if prompts.setdefault(fingerprint, split) != split:
                raise ValueError("duplicate prompt overlaps splits")
    families = {split: {c["family"] for c in cases} for split, cases in splits.items()}
    if families["calibration"] != families["test"] or not families["test"] <= families["train"]:
        raise ValueError("calibration and test families must match and be present in training")


def build(root, job_id, splits, policy, *, allow_training_data_export=False, model=None):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", job_id):
        raise ValueError("job ID must be 1..64 lowercase letters, digits or hyphens")
    if allow_training_data_export is not True:
        raise ValueError("confirm permission to copy training data to approved workers")
    validate_policy(policy)
    validate_splits(splits)
    root = Path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    for split, cases in splits.items():
        benchmark.write_private(
            root / f"{split}.jsonl", "".join(json.dumps(c, allow_nan=False) + "\n" for c in cases)
        )
    benchmark.write_private(
        root / "miner-training.jsonl",
        "".join(json.dumps(c) + "\n" for c in benchmark.training_rows(splits["train"])),
    )
    manifest = {
        "version": VERSION,
        "job_id": job_id,
        "data_access": "approved-workers-training-export",
        "acceptance": policy,
        "files": {name: benchmark.file_hash(root / name) for name in benchmark.FILES},
        "counts": {split: len(cases) for split, cases in splits.items()},
    }
    if model is not None:
        manifest["model"] = models.spec(model)
    benchmark.write_private(root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def audit(root, manifest):
    if (
        manifest.get("version") != VERSION
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", manifest.get("job_id", ""))
        or manifest.get("data_access") != "approved-workers-training-export"
        or set(manifest.get("files", {})) != set(benchmark.FILES)
    ):
        raise ValueError("invalid customer job manifest")
    validate_policy(manifest["acceptance"])
    if "model" in manifest:
        models.validate_spec(manifest["model"])
    root = Path(root)
    for name in benchmark.FILES:
        if benchmark.file_hash(root / name) != manifest["files"][name]:
            raise ValueError(f"job data changed: {name}")
    splits = {
        s: benchmark.read_jsonl(root / f"{s}.jsonl") for s in ("train", "calibration", "test")
    }
    validate_splits(splits)
    if benchmark.read_jsonl(root / "miner-training.jsonl") != benchmark.training_rows(
        splits["train"]
    ):
        raise ValueError("miner export differs from job training split")
    if manifest.get("counts") != {s: len(c) for s, c in splits.items()}:
        raise ValueError("job counts differ from data")
    return splits


def verify_report(root, report, split):
    cases = benchmark.audit(root)[split]
    digest = hashlib.sha256(json.dumps(cases, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if report["dataset_sha256"] != digest:
        raise ValueError("report dataset does not match frozen job split")
    for row in report["miners"]:
        if row["status"] == "evaluated":
            fez.score(cases, row["predictions"])
    return cases


def select(baseline, rows, policy):
    """Delivery gate only; miner rewards remain governed by the round rubric."""
    validate_policy(policy)
    if baseline.get("status") != "evaluated":
        raise ValueError("a valid baseline evaluation is required for delivery")
    eligible = [
        row
        for row in rows
        if row["status"] == "evaluated"
        and row["skill"] > 0
        and row["accuracy"] >= policy["min_accuracy"]
        and baseline["brier"] - row["brier"] > 0
        and baseline["brier"] - row["brier"] >= policy["min_brier_improvement"]
    ]
    if not eligible:
        return {"status": "no_qualifying_model", "acceptance": policy}
    winner = min(eligible, key=lambda row: (row["brier"], -row["accuracy"], row["uid"]))
    return {
        "status": "accepted",
        "uid": winner["uid"],
        "sha256": winner["sha256"],
        "brier_improvement": baseline["brier"] - winner["brier"],
        "acceptance": policy,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-accuracy", type=float, required=True)
    parser.add_argument("--min-brier-improvement", type=float, default=0.0)
    parser.add_argument("--allow-training-data-export", action="store_true")
    args = parser.parse_args()
    try:
        manifest = build(
            args.out,
            args.job_id,
            {s: benchmark.read_jsonl(getattr(args, s)) for s in ("train", "calibration", "test")},
            {
                "min_accuracy": args.min_accuracy,
                "min_brier_improvement": args.min_brier_improvement,
            },
            allow_training_data_export=args.allow_training_data_export,
        )
        print(json.dumps({"job_id": manifest["job_id"], "counts": manifest["counts"]}))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"fez jobs: {error}\n")


if __name__ == "__main__":
    main()
