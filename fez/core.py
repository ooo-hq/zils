"""Fez decision-model subnet: local submissions -> evaluation -> dry-run weights."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from . import ROOT, models

ARTIFACT_FILES = ("adapter_config.json", "adapter_model.safetensors", "head.pt")
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
BASE = "Qwen/Qwen3.5-0.8B-Base"
RUBRIC = "fez-decisions/v1"


def options(question):
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict) or set(criteria) - {"false", "true"}
        ):
            raise ValueError("noul criteria must use false/true keys")
        return ["false", "true"]
    if kind == "choice" and isinstance(criteria, dict) and 2 <= len(criteria) <= 255:
        if all(isinstance(key, str) and key for key in criteria):
            return list(criteria)
    if kind == "score" and isinstance(criteria, list) and 2 <= len(criteria) <= 255:
        return [str(index) for index in range(len(criteria))]
    raise ValueError("question must be noul, or choice/score with 2..255 options")


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation cases must be a nonempty list")
    seen = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or not {"id", "family", "state", "question", "label"} <= case.keys()
        ):
            raise ValueError("case requires id, family, state, question and label")
        for key in ("id", "family", "label"):
            if not isinstance(case[key], str) or not case[key]:
                raise ValueError(f"case {key} must be a nonempty string")
        if case["id"] in seen:
            raise ValueError("duplicate case id")
        seen.add(case["id"])
        if not isinstance(case["question"], dict) or case["label"] not in options(case["question"]):
            raise ValueError("label must identify an available option")
        if len(json.dumps(case, allow_nan=False).encode()) > 128 * 1024:
            raise ValueError("case exceeds 128 KiB")


def score(cases, predictions):
    validate_cases(cases)
    if not isinstance(predictions, list) or len(predictions) != len(cases):
        raise ValueError("exactly one prediction per case is required")
    by_id = {}
    for row in predictions:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in by_id:
            raise ValueError("invalid or duplicate prediction id")
        by_id[row["id"]] = row
    if set(by_id) != {case["id"] for case in cases}:
        raise ValueError("prediction ids differ from evaluation cases")
    families = defaultdict(list)
    latency, correct, confident_errors = [], 0, 0
    for case in cases:
        row = by_id[case["id"]]
        keys = options(case["question"])
        probabilities = row.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
            raise ValueError("probabilities must cover exactly the available options")
        values = list(probabilities.values())
        if any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values
        ):
            raise ValueError("probabilities must be finite numbers in [0, 1]")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        elapsed = row.get("elapsed_ms")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid measured latency")
        loss = sum((probabilities[key] - (key == case["label"])) ** 2 for key in keys)
        families[case["family"]].append((loss, 1 - 1 / len(keys)))
        # Stable tie break; accuracy is diagnostic, Brier determines skill.
        chosen = max(sorted(keys), key=probabilities.get)
        correct += chosen == case["label"]
        confident_errors += chosen != case["label"] and probabilities[chosen] >= 0.9
        latency.append(elapsed)
    brier = statistics.mean(statistics.mean(loss for loss, _ in rows) for rows in families.values())
    baseline = statistics.mean(
        statistics.mean(base for _, base in rows) for rows in families.values()
    )
    return {
        "brier": brier,
        "uniform_brier": baseline,
        "skill": max(0.0, 1 - brier / baseline),
        "accuracy": correct / len(cases),
        "confident_errors": confident_errors,
        "median_ms": statistics.median(latency),
        "p95_ms": sorted(latency)[math.ceil(0.95 * len(latency)) - 1],
        "cases": len(cases),
    }


def weight_vector(rows):
    skills = {}
    for row in rows:
        uid, skill = row["uid"], row.get("skill", 0.0)
        if type(uid) is not int or not 0 <= uid <= 65535 or uid in skills:
            raise ValueError("uids must be unique integers in [0, 65535]")
        if type(skill) not in (int, float) or not math.isfinite(skill) or not 0 <= skill <= 1:
            raise ValueError("invalid skill")
        skills[uid] = skill
    total = sum(skills.values())
    return {uid: skill / total for uid, skill in skills.items() if skill > 0} if total else {}


def checkpoint_hash(path):
    path = Path(path)
    records, total = [], 0
    for name in models.artifact_files(path):
        artifact = path / name
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"checkpoint requires a regular, non-symlink {name}")
        total += artifact.stat().st_size
        if total > MAX_ARTIFACT_BYTES:
            raise ValueError("checkpoint exceeds 512 MiB adapter budget")
        with artifact.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append([name, digest])
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def submission(checkpoint, uid):
    if type(uid) is not int or not 0 <= uid <= 65535:
        raise ValueError("uid must be an integer in [0, 65535]")
    checkpoint = Path(checkpoint).resolve(strict=True)
    return {"uid": uid, "checkpoint": str(checkpoint), "sha256": checkpoint_hash(checkpoint)}


def validate_submissions(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError("submissions must be a nonempty JSON list")
    uids, hashes = set(), set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"uid", "checkpoint", "sha256"}:
            raise ValueError("submission requires exactly uid, checkpoint, sha256")
        uid = entry["uid"]
        if type(uid) is not int or not 0 <= uid <= 65535 or uid in uids:
            raise ValueError("duplicate or invalid uid")
        if not isinstance(entry["checkpoint"], str) or not Path(entry["checkpoint"]).is_absolute():
            raise ValueError("checkpoint must be an absolute local path")
        digest = entry["sha256"]
        if (
            not isinstance(digest, str)
            or not re.fullmatch("[a-f0-9]{64}", digest)
            or digest in hashes
        ):
            raise ValueError("duplicate or invalid checkpoint hash")
        uids.add(uid)
        hashes.add(digest)


def stage(entry, destination):
    source, destination = Path(entry["checkpoint"]), Path(destination)
    if checkpoint_hash(source) != entry["sha256"]:
        raise ValueError("checkpoint hash changed since submission")
    destination.mkdir()
    for name in models.artifact_files(source):
        shutil.copyfile(source / name, destination / name)
    if checkpoint_hash(destination) != entry["sha256"]:
        raise ValueError("checkpoint hash changed while copying")
    for name in models.artifact_files(destination):
        (destination / name).chmod(0o444)


def evaluate(args):
    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    validate_cases(cases)
    entries = json.loads(Path(args.submissions).read_text())
    validate_submissions(entries)
    if not re.fullmatch("[a-f0-9]{40}", args.base_revision):
        raise ValueError("base revision must be an immutable 40-character commit SHA")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    if Path(args.report).exists():
        raise ValueError("report already exists; choose a new round filename")
    # The inference process never receives labels or task-family weights.
    requests = [
        {"id": case["id"], "state": case["state"], "question": case["question"]} for case in cases
    ]
    model = (
        models.JEVK5
        if args.base_revision == models.spec(models.JEVK5)["base_revision"]
        else models.KEV
    )
    base = models.spec(model)["base"]
    runner = Path(__file__).with_name(
        "jevk5_runner.py" if model == models.JEVK5 else "kev_runner.py"
    )
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HOME": os.environ.get("HF_HOME", str(ROOT / ".cache/huggingface")),
    }
    rows = []
    for entry in entries:
        row = {"uid": entry["uid"], "sha256": entry["sha256"]}
        print(
            f"Evaluating miner {entry['uid']} on {len(cases)} cases ({args.device})",
            file=sys.stderr,
            flush=True,
        )
        try:
            with tempfile.TemporaryDirectory(prefix="fez-eval-") as tmp:
                checkpoint = Path(tmp) / "checkpoint"
                stage(entry, checkpoint)
                if models.checkpoint_model(checkpoint) != model:
                    raise ValueError("checkpoint belongs to a different model")
                try:
                    result = subprocess.run(
                        [
                            args.runner_python,
                            str(runner),
                            "--checkpoint",
                            str(checkpoint),
                            "--base",
                            base,
                            "--base-revision",
                            args.base_revision,
                            "--device",
                            args.device,
                        ],
                        input=json.dumps(requests, allow_nan=False),
                        text=True,
                        capture_output=True,
                        timeout=args.timeout,
                        env=environment,
                    )
                except OSError as error:
                    raise RuntimeError(f"cannot start evaluation runtime: {error}") from error
                if result.returncode == 78:
                    raise RuntimeError(
                        f"evaluation environment unavailable: {result.stderr[-1000:]}"
                    )
                if result.returncode:
                    raise ValueError(f"model runner failed: {result.stderr[-1000:]}")
                output = json.loads(result.stdout)
                row.update(score(cases, output["predictions"]))
                row.update(
                    status="evaluated", runtime=output["runtime"], predictions=output["predictions"]
                )
        except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
            row.update(status="rejected", skill=0.0, error=str(error)[:1200])
        rows.append(row)
    report = {
        "mode": "local-dry-run",
        "rubric": RUBRIC,
        "base": base,
        "base_revision": args.base_revision,
        "device": args.device,
        "timeout_s": args.timeout,
        "dataset_sha256": hashlib.sha256(
            json.dumps(cases, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "miners": rows,
        "weights": weight_vector(rows),
    }
    # Exclusive creation protects an earlier round from accidental overwrite.
    with Path(args.report).open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(
        json.dumps(
            {
                "report": str(Path(args.report).resolve()),
                "mode": report["mode"],
                "weights": report["weights"],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit", help="print a local submission; redirect to a JSON file")
    submit.add_argument("--checkpoint", required=True)
    submit.add_argument("--uid", type=int, required=True)
    run = commands.add_parser(
        "evaluate", help="evaluate local checkpoints; never submits chain weights"
    )
    run.add_argument("--submissions", required=True)
    run.add_argument("--cases", required=True)
    run.add_argument("--base-revision", required=True)
    run.add_argument("--runner-python", default=sys.executable)
    run.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--report", required=True)
    args = parser.parse_args()
    try:
        if args.command == "submit":
            print(json.dumps([submission(args.checkpoint, args.uid)], indent=2))
        else:
            evaluate(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"fez: {error}\n")


if __name__ == "__main__":
    main()
