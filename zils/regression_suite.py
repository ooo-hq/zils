"""Adapter regression suite: retrain fixed reference jobs on the incumbent and on a candidate base, then compare.

A candidate that wins the sealed exam is promoted only if adapters trained on it are no worse than adapters
trained on the incumbent on every job (within `tolerance` Brier) and better on at least one. Jobs are public or
Zils-owned case files under ``<suite>/jobs/<name>/{train,calibration,test}.jsonl``; customer data never enters.
Training and inference use the production recipe through the profile's trainer (``zils.h2o train`` or
``zils.d3 train``) and ``zils evaluate``. ``--model`` runs both sides on one pinned profile; with
``--incumbent-model`` and ``--candidate-model`` each side trains on its own profile, base revision and interpreter.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import zils

from . import base_round, benchmark, jobs as customer_jobs, migration_regression, models

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


def train_and_score(
    job, base_dir, label, out, runtime_python, env, device, module, base_revision, seed=553
):
    """Train the production LoRA with trainer `module` on `job` against the base that `env` selects; return
    calibrated test Brier."""
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
            module,
            "train",
            "--data",
            str(export),
            "--reference",
            str(base_dir),
            "--out",
            str(adapter),
            "--seed",
            str(seed),
            "--device",
            device,
        ],
        env,
        str(Path(__file__).resolve().parents[1]),
    )
    reports = {
        split: evaluate_adapter(
            adapter,
            job / f"{split}.jsonl",
            work / f"{split}-report.json",
            runtime_python,
            env,
            device,
            base_revision,
        )
        for split in ("calibration", "test")
    }
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
        "skill": scored["skill"],
        "temperature": temperature,
        "adapter_sha256": zils.checkpoint_hash(adapter),
    }


def evaluate_adapter(adapter, cases, report, runtime_python, env, device, base_revision):
    submissions = report.with_suffix(".submissions.json")
    submissions.write_text(json.dumps([zils.submission(str(adapter), 1)]))
    _run(
        [
            runtime_python,
            "-m",
            "zils",
            "evaluate",
            "--submissions",
            str(submissions),
            "--cases",
            str(cases),
            "--base-revision",
            base_revision,
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
    return json.loads(report.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    text_models = list(models.TEXT_RUNTIME_MODULES)
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--incumbent", required=True, help="reference checkpoint directory of the pinned base"
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="verified base candidate directory (cross-profile: reference directory of the candidate profile)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model", choices=text_models, help=f"same-profile mode, default {models.D3}"
    )
    parser.add_argument("--incumbent-model", choices=text_models, help="cross-profile mode")
    parser.add_argument("--candidate-model", choices=text_models, help="cross-profile mode")
    parser.add_argument(
        "--runtime-python", help="interpreter of the (incumbent) profile; default its env variable"
    )
    parser.add_argument(
        "--candidate-runtime-python",
        help="cross-profile only: interpreter of the candidate profile; default its env variable",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--seeds", nargs="+", type=int, help="training seeds; migration default: 553 771 991"
    )
    parser.add_argument(
        "--migration-references",
        help="JSON mapping suite jobs to frozen public releases and original splits",
    )
    parser.add_argument("--min-accuracy", type=float, default=0.8)
    parser.add_argument("--min-brier-improvement", type=float, default=0.01)
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("Tolerance must be finite and nonnegative")
    seeds = args.seeds or ([553, 771, 991] if args.migration_references else [553])
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        parser.error("Seeds must be distinct nonnegative integers")
    policy = {
        "min_accuracy": args.min_accuracy,
        "min_brier_improvement": args.min_brier_improvement,
    }
    customer_jobs.validate_policy(policy)
    cross = bool(args.incumbent_model or args.candidate_model)
    if cross and not (args.incumbent_model and args.candidate_model and not args.model):
        parser.error(
            "cross-profile mode takes --incumbent-model and --candidate-model, not --model"
        )
    if args.candidate_runtime_python and not cross:
        parser.error("--candidate-runtime-python applies to cross-profile mode only")
    incumbent_model = args.incumbent_model or args.model or models.D3
    candidate_model = args.candidate_model or incumbent_model

    def interpreter(model, given):
        return given or os.environ.get(models.runtime_python_env(model), sys.executable)

    incumbent_python = interpreter(incumbent_model, args.runtime_python)
    candidate_python = interpreter(candidate_model, args.candidate_runtime_python)
    if not cross:
        candidate_python = incumbent_python
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    meta = models.metadata(args.candidate)
    if cross:
        if (meta or {}).get("model") != candidate_model:
            raise SystemExit(f"--candidate must be a {candidate_model} checkpoint")
        if (models.metadata(args.incumbent) or {}).get("model") != incumbent_model:
            raise SystemExit(f"--incumbent must be a {incumbent_model} checkpoint")
    elif (meta or {}).get("kind") != "candidate" or meta["model"] != candidate_model:
        raise SystemExit(f"--candidate must be a {candidate_model} base candidate checkpoint")

    def side_env(model, python):
        env = {**os.environ, models.runtime_python_env(model): python}
        for name in models.CANDIDATE_BASE_ENV.values():
            env.pop(name, None)
        return env

    incumbent_env = side_env(incumbent_model, incumbent_python)
    if cross:
        candidate_env, candidate_reference = (
            side_env(candidate_model, candidate_python),
            args.candidate,
        )
    else:
        # The candidate is loaded as the base through the profile's candidate env; its reference checkpoint is a base stub.
        candidate_env = {
            **incumbent_env,
            models.candidate_base_env(candidate_model): str(Path(args.candidate).resolve()),
        }
        candidate_reference = out / "candidate-reference"
        candidate_reference.mkdir()
        models.write_metadata(candidate_reference, kind="base", model=candidate_model)
    sides = {
        "incumbent": (incumbent_model, Path(args.incumbent), incumbent_python, incumbent_env),
        "candidate": (candidate_model, Path(candidate_reference), candidate_python, candidate_env),
    }
    suite_jobs = jobs(args.suite)
    frozen = (
        migration_regression.references(args.migration_references, suite_jobs, incumbent_model)
        if args.migration_references
        else {}
    )
    results, detail, runs, migration = {}, {}, {}, {}
    for job in suite_jobs:
        baseline = None
        if frozen:
            adapter, release = frozen[job.name]
            report_path = out / f"{job.name}-frozen-test-report.json"
            report = evaluate_adapter(
                adapter,
                job / "test.jsonl",
                report_path,
                incumbent_python,
                incumbent_env,
                args.device,
                models.spec(incumbent_model)["base_revision"],
            )
            cases = benchmark.read_jsonl(job / "test.jsonl")
            rows = base_round.rows_for(report, cases)
            if 1 not in rows:
                raise RuntimeError("Frozen migration release evaluation rejected")
            # The runtime applies the release's stored temperature; never recalibrate it.
            baseline = zils.score(cases, rows[1]["predictions"])
            migration[job.name] = {}
        runs[job.name] = {}
        for seed in seeds:
            run_out = out if len(seeds) == 1 else out / f"seed-{seed}"
            run = {
                label: train_and_score(
                    job,
                    reference,
                    label,
                    run_out,
                    python,
                    env,
                    args.device,
                    "zils" + models.TEXT_RUNTIME_MODULES[model],
                    models.spec(model)["base_revision"],
                    seed=seed,
                )
                for label, (model, reference, python, env) in sides.items()
            }
            runs[job.name][str(seed)] = run
            if baseline is not None:
                migration[job.name][str(seed)] = migration_regression.compare(
                    baseline, run["candidate"], release, policy
                )
        detail[job.name] = (
            run
            if len(seeds) == 1
            else {
                side: {
                    key: sum(r[side][key] for r in runs[job.name].values()) / len(seeds)
                    for key in ("brier", "accuracy", "skill")
                }
                for side in sides
            }
        )
        results[job.name] = {k: v["brier"] for k, v in detail[job.name].items()}
        (out / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
    decision = verdict(results, args.tolerance)
    migration_report = migration_regression.summary(migration) if frozen else None
    if migration_report is not None:
        migration_report["stable_eligible"] &= decision["passed"]
    result = {
        "verdict": decision,
        "seeds": seeds,
        "regression_seed_rule": "mean-brier-per-job",
        "runs": runs,
        "migration": (
            {
                **migration_report,
                "policy": policy,
                "tolerance": customer_jobs.MIGRATION_TOLERANCE,
                "references": {
                    name: {
                        "fingerprint": release["fingerprint"],
                        "checkpoint_sha256": release["checkpoint_sha256"],
                        "temperature": release["temperature"],
                        "test_sha256": benchmark.file_hash(
                            Path(args.suite) / "jobs" / name / "test.jsonl"
                        ),
                    }
                    for name, (_, release) in frozen.items()
                },
            }
            if frozen
            else None
        ),
        "jobs": detail,
        "candidate": str(args.candidate),
        "incumbent_model": incumbent_model,
        "candidate_model": candidate_model,
        "candidate_sha256": zils.checkpoint_hash(args.candidate),
        "incumbent_sha256": zils.checkpoint_hash(args.incumbent),
    }
    (out / "suite-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
