"""Qualify this host for the pinned H2O profile and write qualification.json.

Runs from .venv-kev; drives the isolated .venv-h2o runtime through subprocesses
so the pinned runtime check, a 2,048-token training step and an adapter reload
happen in the real environment. Mirrors the 10 October 2026 RTX 4090 check.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from zils import models
from zils.miner_grading import trainer_identity
from zils.runtime import gpu_ready

FLOOR_MIB = 12288
STATE = {"ticket": "Please help with payment.", "history": "payment issue " * 950}
QUESTIONS = [
    {
        "type": "choice",
        "instructions": "Route this ticket.",
        "criteria": {"billing": "payment", "support": "technical"},
    },
    {
        "type": "noul",
        "instructions": "Does the ticket mention payment?",
        "criteria": {"true": "payment", "false": "no payment"},
    },
    {"type": "score", "instructions": "Rate urgency.", "criteria": ["low", "medium", "high"]},
]
RELOAD_PREDICT = """# reload_predict probe
import json, sys
from zils.h2o import DecisionModel
adapter, state, questions = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
model = DecisionModel(adapter, "cuda")
lengths = [len(model.encode(state, q)[0]) for q in questions]
probes = [model.predict(state, q) for q in questions]
print(json.dumps({"ok": all(abs(sum(p.values()) - 1) < 1e-7 for p in probes), "max_prompt_tokens": max(lengths)}))
"""


def nvidia(run):
    result = run(
        ["nvidia-smi", "--id=0", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    name, driver = [x.strip() for x in result.stdout.strip().split(",", 1)]
    return {"name": name, "driver": driver}


def free_memory_mib(run):
    result = run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return int(result.stdout.strip().splitlines()[0])


def synthetic_rows(path):
    rows = []
    for question in QUESTIONS * 2:
        label = (
            next(iter(question["criteria"]))
            if question["type"] == "choice"
            else "true"
            if question["type"] == "noul"
            else "1"
        )
        rows.append({"state": STATE, "questions": {"decision": dict(question, label=label)}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def qualify(root, runtime_python, out, run=subprocess.run, free_mib=None):
    out.mkdir(parents=True, exist_ok=True)
    (out / "qualification.json").unlink(missing_ok=True)  # never leave a stale passing report
    report = {
        "model": models.H2O,
        **models.profile_identity(models.H2O),
        "trainer_sha256": trainer_identity(models.H2O),
        "completed_at": None,
    }

    def finish(code):
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        (out / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
        if code:
            sys.exit(code)
        return report

    adapter = out / "adapter"
    if adapter.exists():
        report["error"] = "adapter directory exists; use a fresh --out or remove it"
        return finish(5)
    stage = "gpu query"
    try:
        report["gpu"] = nvidia(run)
        free = free_memory_mib(run) if free_mib is None else free_mib
        report["gpu"]["free_mib_at_start"] = free
        report["capacity_ok"] = free >= FLOOR_MIB and (
            free_mib is not None or gpu_ready("cuda", model=models.H2O)
        )
        if not report["capacity_ok"]:
            report["error"] = f"free GPU memory {free} MiB is below the {FLOOR_MIB} MiB floor"
            return finish(2)
        env_root = str(root)
        stage = "runtime check"
        verify = run(
            [
                runtime_python,
                "-c",
                "from zils.h2o import verify_runtime; verify_runtime(); print('ok')",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=env_root,
        )
        report["runtime_verified"] = verify.returncode == 0 and "ok" in verify.stdout
        if not report["runtime_verified"]:
            report["error"] = verify.stderr[-2000:]
            return finish(3)
        stage = "reference"
        reference = out / "reference"
        if not (reference / "model.json").exists():
            shutil.rmtree(reference, ignore_errors=True)  # half-written reference
            run(
                [runtime_python, "-m", "zils.h2o", "reference", "--out", str(reference)],
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=env_root,
            )
        stage = "train"
        data = out / "train.jsonl"
        examples = synthetic_rows(data)
        run(
            [
                runtime_python,
                "-m",
                "zils.h2o",
                "train",
                "--data",
                str(data),
                "--reference",
                str(reference),
                "--out",
                str(adapter),
                "--seed",
                "771",
                "--device",
                "cuda",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=env_root,
        )
        metrics = json.loads((adapter / "training_metrics.json").read_text())
        stage = "reload probe"
        probe = run(
            [
                runtime_python,
                "-c",
                RELOAD_PREDICT,
                str(adapter),
                json.dumps(STATE),
                json.dumps(QUESTIONS),
            ],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=env_root,
        )
        if probe.returncode != 0:
            report["error"] = probe.stderr[-2000:]
        reload = (
            json.loads(probe.stdout)
            if probe.returncode == 0
            else {"ok": False, "max_prompt_tokens": None}
        )
        report["training"] = {
            "examples": examples,
            "optimizer_steps": metrics["optimizer_steps"],
            "seconds": metrics["training_seconds"],
            "peak_gpu_mib": metrics["peak_gpu_allocated_bytes"] // 2**20,
            "max_prompt_tokens": reload["max_prompt_tokens"],
        }
        report["reload_predict_ok"] = bool(reload["ok"])
        return finish(0 if report["reload_predict_ok"] else 4)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
        ValueError,  # includes json.JSONDecodeError
        KeyError,
    ) as exc:
        detail = getattr(exc, "stderr", None) or exc
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        report["error"] = f"{stage}: {str(detail)[-2000:]}"
        return finish(5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=".", help="zils checkout (runtime imports zils from here)"
    )
    parser.add_argument("--runtime-python", required=True, help=".venv-h2o/bin/python")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = qualify(Path(args.root).resolve(), args.runtime_python, Path(args.out).resolve())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
