"""Fit a fresh checkpoint copy on Fez's frozen calibration split using Kev's temperature fitter."""

import argparse
import json
import math
from pathlib import Path

import fez

from . import benchmark, jobs


def fit(root, report, uid, checkpoint, destination):
    # Verifies the report's dataset and distributions before any checkpoint is written.
    jobs.verify_report(root, report, "calibration")
    matches = [m for m in report["miners"] if m["uid"] == uid]
    if len(matches) != 1 or matches[0]["status"] != "evaluated":
        raise ValueError("require exactly one evaluated checkpoint for this uid")
    miner = matches[0]
    if miner["runtime"].get("temperature") != 1.0:
        raise ValueError("calibration predictions must use temperature 1.0")
    entry = fez.submission(checkpoint, uid)
    if entry["sha256"] != miner["sha256"]:
        raise ValueError("calibration report belongs to a different checkpoint")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"checkpoint destination already exists: {destination}")

    from kev.checkpoint import read_meta, write_meta
    from kev.metrics import fit_temperature, probabilities_at_temperature

    meta = read_meta(checkpoint)
    if meta.temperature != 1.0:
        raise ValueError("source checkpoint must have temperature 1.0")
    cases = benchmark.read_jsonl(Path(root) / "calibration.jsonl")
    predictions = {p["id"]: p for p in miner["predictions"]}
    rows = []
    for case in cases:
        keys = fez.options(case["question"])
        rows.append(
            {
                "p": [predictions[case["id"]]["probabilities"][k] for k in keys],
                "label": keys.index(case["label"]),
                "task": case["family"],
                "keys": keys,
                "source": case["family"],
                "inference_temperature": 1.0,
                # Kev filters on this eligibility marker. Our declared fit population includes all variants.
                "variant": "clean",
                "fez_variant": case.get("variant", "customer"),
            }
        )
    temperature = fit_temperature(rows, aggregation="macro", points=81)
    if not math.isfinite(temperature) or not 0.25 <= temperature <= 4:
        raise ValueError("temperature fitter returned an invalid value")
    adjusted = [
        {
            **predictions[c["id"]],
            "probabilities": dict(
                zip(row["keys"], probabilities_at_temperature(row, temperature).tolist())
            ),
        }
        for c, row in zip(cases, rows)
    ]
    result = {
        "raw_checkpoint_sha256": entry["sha256"],
        "dataset_sha256": report["dataset_sha256"],
        "benchmark_manifest_sha256": benchmark.file_hash(Path(root) / "manifest.json"),
        "temperature": temperature,
        "fit_cases": len(rows),
        "fit_split": "calibration",
        "method": "Kev macro-family NLL; 81 log-spaced temperatures in [0.25, 4]; all calibration variants",
        "probability_floor": 1e-9,
        "at_grid_boundary": temperature in (0.25, 4.0),
        "before": fez.score(cases, miner["predictions"]),
        "after": fez.score(cases, adjusted),
    }
    # A single positive temperature preserves ranking; runtime timing is measured again in the real test run.
    if result["before"]["accuracy"] != result["after"]["accuracy"]:
        raise ValueError(
            "temperature unexpectedly changed accuracy; inspect numerical ties before saving"
        )
    fez.stage(entry, destination)
    meta.temperature = temperature
    meta.extra["fez_temperature_fit"] = {
        k: result[k]
        for k in (
            "raw_checkpoint_sha256",
            "dataset_sha256",
            "benchmark_manifest_sha256",
            "fit_cases",
            "fit_split",
            "method",
        )
    }
    head = destination / "head.pt"
    head.chmod(0o600)
    write_meta(destination, meta)
    head.chmod(0o444)
    result["checkpoint_sha256"] = fez.checkpoint_hash(destination)
    benchmark.write_private(
        destination / "calibration.json", json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--uid", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        result = fit(
            args.benchmark,
            json.loads(Path(args.report).read_text()),
            args.uid,
            args.checkpoint,
            args.out,
        )
        print(
            json.dumps(
                {
                    "checkpoint": str(Path(args.out).resolve()),
                    "temperature": result["temperature"],
                    "fit_cases": result["fit_cases"],
                    "sha256": result["checkpoint_sha256"],
                }
            )
        )
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f"fez calibration: {error}\n")


if __name__ == "__main__":
    main()
