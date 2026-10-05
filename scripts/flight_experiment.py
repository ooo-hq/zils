"""Run one frozen JevK5 flight experiment without changing any serving model."""

import argparse
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from scripts.flight_data import digest, history_probability, read_rows, write_json


def validate_run(environment, metadata):
    if (
        not environment.get("FEZ_COMPUTE_LOCK")
        or int(environment.get("FEZ_GPU_MIN_FREE_MIB", 0)) < 12288
    ):
        raise ValueError("Require the shared GPU lock and at least 12288 MiB free GPU memory")
    if metadata.get("kind") != "base" or metadata.get("temperature") != 1.22:
        raise ValueError("Reference must be the unchanged JevK5 base at temperature 1.22")


def execute(command, log_path, environment, cwd, timeout):
    """Reap the GPU child before the caller releases its shared compute lock."""

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, interrupted)
    process = None
    try:
        with log_path.open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status = process.wait(timeout=timeout)
            if status:
                raise subprocess.CalledProcessError(status, command)
    finally:
        # Repeated stop requests must not interrupt cleanup and unlock a busy GPU.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)


def probability(logit, temperature):
    value = logit / temperature
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def metrics(labels, probabilities):
    if not labels or len(labels) != len(probabilities) or any(y not in (0, 1) for y in labels):
        raise ValueError("Require matching nonempty binary labels and predictions")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilities must be finite and in [0,1]")
    n, positives = len(labels), sum(labels)
    negatives = n - positives
    tp = sum(y == 1 and p >= 0.5 for y, p in zip(labels, probabilities))
    tn = sum(y == 0 and p < 0.5 for y, p in zip(labels, probabilities))
    ranked = sorted(zip(probabilities, labels))
    rank_sum, start = 0.0, 0
    while start < n:
        end = start + 1
        while end < n and ranked[end][0] == ranked[start][0]:
            end += 1
        rank_sum += ((start + 1 + end) / 2) * sum(y for _, y in ranked[start:end])
        start = end
    return {
        "count": n,
        "late_rate": positives / n,
        "mean_predicted_late": statistics.mean(probabilities),
        "brier": statistics.mean((p - y) ** 2 for p, y in zip(probabilities, labels)),
        "log_loss": -statistics.mean(
            math.log(max(1e-12, p if y else 1 - p)) for y, p in zip(labels, probabilities)
        ),
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": (tp / positives + tn / negatives) / 2
        if positives and negatives
        else None,
        "late_recall": tp / positives if positives else None,
        "roc_auc": (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
        if positives and negatives
        else None,
        "confident_errors": sum(
            (p >= 0.9 and y == 0) or (p <= 0.1 and y == 1) for p, y in zip(probabilities, labels)
        ),
    }


def fit_temperature(labels, logits):
    candidates = [0.25 * 16 ** (i / 80) for i in range(81)]
    return min(
        candidates, key=lambda t: metrics(labels, [probability(x, t) for x in logits])["log_loss"]
    )


def interval(rows, reference, candidate):
    """Paired resampling by flight date keeps same-day correlated outcomes together."""
    groups = defaultdict(lambda: [0.0, 0])
    for row, a, b in zip(rows, reference, candidate):
        y = row["label"] == "late"
        groups[row["group_id"]][0] += (a - y) ** 2 - (b - y) ** 2
        groups[row["group_id"]][1] += 1
    values = list(groups.values())
    rng = random.Random(553)
    draws = []
    for _ in range(2000):
        chosen = rng.choices(values, k=len(values))
        draws.append(sum(x[0] for x in chosen) / sum(x[1] for x in chosen))
    draws.sort()
    return {
        "mean_improvement": sum(x[0] for x in values) / sum(x[1] for x in values),
        "ci95": [draws[49], draws[1949]],
        "date_clusters": len(values),
        "resamples": 2000,
    }


def audit(data):
    manifest = json.loads((data / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(data / name) != expected:
            raise ValueError(f"Frozen experiment file changed: {name}")
    splits = {s: read_rows(data / f"{s}.jsonl") for s in ("train", "calibration", "test")}
    ids, dates = set(), set()
    previous = ""
    for split, rows in splits.items():
        if len(rows) != manifest["splits"][split]["count"]:
            raise ValueError("Sample count mismatch")
        current_ids = {r["id"] for r in rows}
        current_dates = {r["group_id"] for r in rows}
        if len(current_ids) != len(rows) or ids & current_ids or dates & current_dates:
            raise ValueError("Flight identities or dates overlap splits")
        if min(current_dates) <= previous:
            raise ValueError("Splits are not chronological")
        previous = max(current_dates)
        ids.update(current_ids)
        dates.update(current_dates)
        expected_inputs = [{k: r[k] for k in ("id", "state", "question")} for r in rows]
        if read_rows(data / f"{split}-inputs.jsonl") != expected_inputs:
            raise ValueError("Inference inputs differ from frozen examples")
    return splits


def infer(checkpoint, inputs, output):
    import torch

    from fez.jevk5 import DecisionModel, encode

    rows = read_rows(inputs)
    if any(set(row) != {"id", "state", "question"} for row in rows):
        raise ValueError("Inference receives no labels")
    model = DecisionModel(checkpoint, "cuda")
    encoded = [encode(model.tokenizer, r["state"], r["question"]) for r in rows]
    if max(len(ids) for ids, _ in encoded) > 512:
        raise ValueError("Experiment input exceeds the fixed memory budget")
    torch.cuda.reset_peak_memory_stats()
    with Path(output).open("x") as stream, torch.inference_mode():
        # Warmup is outside recorded per-flight latency.
        model.logits(*[encoded[0][0], len(encoded[0][1])])
        torch.cuda.synchronize()
        for index, (row, (ids, keys)) in enumerate(zip(rows, encoded)):
            started = time.perf_counter()
            logits = model.logits(ids, len(keys)).double().cpu().tolist()
            torch.cuda.synchronize()
            value = logits[keys.index("late")] - logits[keys.index("not_late")]
            if not math.isfinite(value):
                raise ValueError("Nonfinite model logits")
            stream.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "logit": value,
                        "elapsed_ms": (time.perf_counter() - started) * 1000,
                    }
                )
                + "\n"
            )
            stream.flush()
            if (index + 1) % 128 == 0:
                print(f"Evaluated {index + 1}/{len(rows)}", flush=True)
    print(
        json.dumps(
            {
                "max_input_tokens": max(len(ids) for ids, _ in encoded),
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        ),
        flush=True,
    )


def prediction_values(path, rows):
    predictions = read_rows(path)
    if len(predictions) != len(rows) or [p["id"] for p in predictions] != [r["id"] for r in rows]:
        raise ValueError("Predictions differ from the frozen flight identities")
    if any(not math.isfinite(p["logit"]) for p in predictions):
        raise ValueError("Nonfinite predictions")
    return [p["logit"] for p in predictions]


def report(data, output):
    splits = audit(data)
    history = json.loads((data / "historical.json").read_text())
    labels = {s: [int(r["label"] == "late") for r in rows] for s, rows in splits.items()}
    temperatures, predictions, latency = {}, {}, {}
    for model in ("base", "adapter"):
        calibration_logits = prediction_values(
            output / f"{model}-calibration.jsonl", splits["calibration"]
        )
        temperature = fit_temperature(labels["calibration"], calibration_logits)
        temperatures[model] = temperature
        test_logits = prediction_values(output / f"{model}-test.jsonl", splits["test"])
        predictions[model + "_calibrated"] = [probability(x, temperature) for x in test_logits]
        raw_temperature = 1.22 if model == "base" else 1.0
        predictions[model + "_raw"] = [probability(x, raw_temperature) for x in test_logits]
        latency[model] = statistics.median(
            p["elapsed_ms"] for p in read_rows(output / f"{model}-test.jsonl")
        )
    predictions["historical_rate"] = [history_probability(history, r) for r in splits["test"]]
    comparisons = {
        name: interval(splits["test"], predictions[name], predictions["adapter_calibrated"])
        for name in ("base_calibrated", "historical_rate")
    }
    result = {
        "experiment": "flight-delay-001",
        "manifest_sha256": digest(data / "manifest.json"),
        "protocol_sha256": digest(data / "protocol.json"),
        "temperatures": temperatures,
        "metrics": {name: metrics(labels["test"], p) for name, p in predictions.items()},
        "adapter_brier_improvement": comparisons,
        "median_forward_ms": latency,
        "demonstrated_improvement": all(value["ci95"][0] > 0 for value in comparisons.values()),
        "training": json.loads((output / "adapter" / "training_metrics.json").read_text()),
        "artifacts": {p.name: digest(p) for p in (output / "adapter").iterdir() if p.is_file()},
        "prediction_hashes": {p.name: digest(p) for p in output.glob("*-*.jsonl")},
        "run": json.loads((output / "run.json").read_text()),
    }
    write_json(output / "results.json", result)
    return result


def run(data, output, reference, timeout):
    from transformers import AutoTokenizer

    from fez import models
    from fez.jevk5 import base_path, encode
    from fez.runtime import gpu_ready, locked

    splits = audit(data)
    if output.exists():
        raise FileExistsError("Preserve previous runs; use a new output directory")
    metadata = models.metadata(reference)
    validate_run(os.environ, metadata)
    reference_hash = digest(reference / "model.json")
    tokenizer = AutoTokenizer.from_pretrained(base_path(), local_files_only=True)
    max_tokens = max(
        len(encode(tokenizer, row["state"], row["question"])[0])
        for rows in splits.values()
        for row in rows
    )
    if max_tokens > 512:
        raise ValueError("Experiment input exceeds the fixed memory budget")
    with locked(os.environ["FEZ_COMPUTE_LOCK"], wait=False):
        if not gpu_ready("cuda"):
            raise RuntimeError("Insufficient free GPU capacity; serving remains unchanged")
        output.mkdir(parents=True, mode=0o700)
        environment = {
            k: v
            for k, v in os.environ.items()
            if k not in {"SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_DB_URL"}
        }
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        root = Path(__file__).resolve().parents[1]
        write_json(
            output / "run.json",
            {
                "model": models.spec(models.JEVK5),
                "reference_metadata": metadata,
                "reference_metadata_sha256": reference_hash,
                "max_input_tokens": max_tokens,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "code_sha256": {
                    name: digest(root / name)
                    for name in (
                        "scripts/flight_data.py",
                        "scripts/flight_experiment.py",
                        "fez/jevk5.py",
                    )
                },
                "hardware": subprocess.check_output(
                    [
                        os.environ.get("FEZ_NVIDIA_SMI", "nvidia-smi"),
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).strip(),
            },
        )
        deadline = time.monotonic() + timeout

        def child(name, command):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Experiment time budget exhausted")
            print(f"START {name}", flush=True)
            execute(
                [sys.executable, "-m", *command],
                output / f"{name}.log",
                environment,
                root,
                remaining,
            )
            print(f"DONE {name}", flush=True)

        child(
            "training",
            [
                "fez.jevk5",
                "train",
                "--data",
                str(data / "train-export.jsonl"),
                "--reference",
                str(reference),
                "--device",
                "cuda",
                "--seed",
                "553",
                "--out",
                str(output / "adapter"),
            ],
        )
        # Freeze the trained weights before either model receives final test inputs.
        adapter_hashes = {p.name: digest(p) for p in (output / "adapter").iterdir() if p.is_file()}
        write_json(output / "frozen-adapter.json", adapter_hashes)
        for model, checkpoint in (("base", reference), ("adapter", output / "adapter")):
            for split in ("calibration", "test"):
                child(
                    f"{model}-{split}",
                    [
                        "scripts.flight_experiment",
                        "infer",
                        "--checkpoint",
                        str(checkpoint),
                        "--inputs",
                        str(data / f"{split}-inputs.jsonl"),
                        "--out",
                        str(output / f"{model}-{split}.jsonl"),
                    ],
                )
        if adapter_hashes != {
            p.name: digest(p) for p in (output / "adapter").iterdir() if p.is_file()
        }:
            raise ValueError("Adapter changed after freeze")
        if digest(reference / "model.json") != reference_hash:
            raise ValueError("Base reference changed during the experiment")
        result = report(data, output)
        print(
            json.dumps(
                {"completed": True, "demonstrated_improvement": result["demonstrated_improvement"]}
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--data", type=Path, required=True)
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument("--reference", type=Path, required=True)
    runner.add_argument("--timeout", type=int, default=3600)
    inference = commands.add_parser("infer")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--inputs", type=Path, required=True)
    inference.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.data.resolve(), args.out.resolve(), args.reference.resolve(), args.timeout)
    else:
        infer(args.checkpoint, args.inputs, args.out)


if __name__ == "__main__":
    main()
