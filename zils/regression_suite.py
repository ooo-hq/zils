"""Adapter regression suite: retrain fixed reference jobs on the incumbent and on a candidate base, then compare.

A candidate that wins the sealed exam is promoted only if adapters trained on it are no worse than adapters
trained on the incumbent on every job (within `tolerance` Brier) and better on at least one. Jobs are public or
Zils-owned case files under ``<suite>/jobs/<name>/{train,calibration,test}.jsonl``; customer data never enters.
Training and inference use the production recipe through ``zils.h2o train`` and ``zils evaluate``.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import zils

from . import base_round, benchmark, models

DEFAULT_TOLERANCE = 0.005


def jobs(suite):
    root = Path(suite) / "jobs"
    found = sorted(p for p in root.iterdir() if (p / "train.jsonl").is_file())
    if not found:
        raise ValueError("suite has no jobs")
    for job in found:
        for split in ("train", "calibration", "test"):
            zils.validate_cases(benchmark.read_jsonl(job / f"{split}.jsonl"))
    return found


def verdict(results, tolerance=DEFAULT_TOLERANCE):
    """results: {job: {"incumbent": brier, "candidate": brier}} on calibrated test Brier (lower is better)."""
    deltas = {job: r["incumbent"] - r["candidate"] for job, r in results.items()}
    worse = {job: d for job, d in deltas.items() if d < -tolerance}
    better = {job: d for job, d in deltas.items() if d > 0}
    return {
        "passed": not worse and bool(better),
        "deltas": deltas,
        "worse": worse,
        "better": better,
        "tolerance": tolerance,
    }


def _run(command, env, cwd):
    result = subprocess.run(command, text=True, capture_output=True, env=env, cwd=cwd)
    if result.returncode:
        raise RuntimeError(f"{command[2:5]} failed: {result.stderr[-2000:]}")
    return result


def train_and_score(job, base_dir, label, out, runtime_python, env, device):
    """Train the production LoRA on `job` against the base that `env` selects; return calibrated test Brier."""
    work = Path(out) / job.name / label
    work.mkdir(parents=True, exist_ok=False)
    rows = benchmark.training_rows(benchmark.read_jsonl(job / "train.jsonl"))
    export = work / "training-export.jsonl"
    export.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    adapter = work / "adapter"
    _run(
        [
            runtime_python,
            "-m",
            "zils.h2o",
            "train",
            "--data",
            str(export),
            "--reference",
            str(base_dir),
            "--out",
            str(adapter),
            "--seed",
            "553",
            "--device",
            device,
        ],
        env,
        str(Path(__file__).resolve().parents[1]),
    )
    submissions = work / "submissions.json"
    submissions.write_text(json.dumps([zils.submission(str(adapter), 1)]))
    reports = {}
    for split in ("calibration", "test"):
        report = work / f"{split}-report.json"
        _run(
            [
                runtime_python,
                "-m",
                "zils",
                "evaluate",
                "--submissions",
                str(submissions),
                "--cases",
                str(job / f"{split}.jsonl"),
                "--base-revision",
                models.spec(models.H2O)["base_revision"],
                "--runner-python",
                runtime_python,
                "--device",
                device,
                "--timeout",
                "7200",
                "--report",
                str(report),
            ],
            env,
            str(Path(__file__).resolve().parents[1]),
        )
        reports[split] = json.loads(report.read_text())
    cases = {s: benchmark.read_jsonl(job / f"{s}.jsonl") for s in ("calibration", "test")}
    rows_cal = base_round.rows_for(reports["calibration"], cases["calibration"])
    rows_test = base_round.rows_for(reports["test"], cases["test"])
    if 1 not in rows_cal or 1 not in rows_test:
        raise RuntimeError(
            f"adapter evaluation rejected: {reports['test']['miners'][0].get('error')}"
        )
    temperature = base_round.fit_temperature(cases["calibration"], rows_cal[1]["predictions"])
    scored = zils.score(
        cases["test"],
        base_round.apply_temperature(cases["test"], rows_test[1]["predictions"], temperature),
    )
    return {
        "brier": scored["brier"],
        "accuracy": scored["accuracy"],
        "temperature": temperature,
        "adapter_sha256": zils.checkpoint_hash(adapter),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--incumbent", required=True, help="reference checkpoint directory of the pinned base"
    )
    parser.add_argument("--candidate", required=True, help="verified base candidate directory")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--runtime-python", default=os.environ.get("ZILS_H2O_RUNTIME_PYTHON", sys.executable)
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if models.metadata(args.candidate)["kind"] != "candidate":
        raise SystemExit("--candidate must be a base candidate checkpoint")
    base_env = {**os.environ, "ZILS_H2O_RUNTIME_PYTHON": args.runtime_python}
    base_env.pop("ZILS_H2O_CANDIDATE_BASE", None)
    candidate_env = {**base_env, "ZILS_H2O_CANDIDATE_BASE": str(Path(args.candidate).resolve())}
    # The candidate is loaded as the base through ZILS_H2O_CANDIDATE_BASE; its reference checkpoint is a base stub.
    stub = out / "candidate-reference"
    stub.mkdir()
    models.write_metadata(stub, kind="base", model=models.H2O)
    results, detail = {}, {}
    for job in jobs(args.suite):
        detail[job.name] = {
            "incumbent": train_and_score(
                job,
                Path(args.incumbent),
                "incumbent",
                out,
                args.runtime_python,
                base_env,
                args.device,
            ),
            "candidate": train_and_score(
                job, stub, "candidate", out, args.runtime_python, candidate_env, args.device
            ),
        }
        results[job.name] = {k: v["brier"] for k, v in detail[job.name].items()}
        (out / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
    decision = verdict(results, args.tolerance)
    (out / "suite-result.json").write_text(
        json.dumps(
            {"verdict": decision, "jobs": detail, "candidate": str(args.candidate)}, indent=2
        )
        + "\n"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
