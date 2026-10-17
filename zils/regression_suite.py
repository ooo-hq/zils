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


def train_and_score(job, base_dir, label, out, runtime_python, env, device, module, base_revision):
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
        "--model", choices=text_models, help=f"same-profile mode, default {models.H2O}"
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
    args = parser.parse_args()
    cross = bool(args.incumbent_model or args.candidate_model)
    if cross and not (args.incumbent_model and args.candidate_model and not args.model):
        parser.error(
            "cross-profile mode takes --incumbent-model and --candidate-model, not --model"
        )
    if args.candidate_runtime_python and not cross:
        parser.error("--candidate-runtime-python applies to cross-profile mode only")
    incumbent_model = args.incumbent_model or args.model or models.H2O
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
    results, detail = {}, {}
    for job in jobs(args.suite):
        detail[job.name] = {
            label: train_and_score(
                job,
                reference,
                label,
                out,
                python,
                env,
                args.device,
                "zils" + models.TEXT_RUNTIME_MODULES[model],
                models.spec(model)["base_revision"],
            )
            for label, (model, reference, python, env) in sides.items()
        }
        results[job.name] = {k: v["brier"] for k, v in detail[job.name].items()}
        (out / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
    decision = verdict(results, args.tolerance)
    result = {
        "verdict": decision,
        "jobs": detail,
        "candidate": str(args.candidate),
        "incumbent_model": incumbent_model,
        "candidate_model": candidate_model,
    }
    (out / "suite-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
