"""Qualify this host for a pinned text profile and write qualification.json.

Runs from .venv-kev; drives the isolated .venv-h2o or .venv-d3 runtime through
subprocesses so the pinned runtime check, a training step at the profile's input
cap (2,048 tokens for H2O, 4,096 for d3) and an adapter reload happen in the
real environment. d3 also checks its readout against the vendor's. Mirrors the
10 October 2026 RTX 4090 check.
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
H2O_CAP = 2048
PHRASE = "payment issue "
HEADROOM_TOKENS = 200  # d3 rows target the cap minus this; the encoder refuses anything over
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
from MODULE import DecisionModel
adapter, state, questions = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
model = DecisionModel(adapter, "cuda")
lengths = [len(model.encode(state, q)[0]) for q in questions]
probes = [model.predict(state, q) for q in questions]
print(json.dumps({"ok": all(abs(sum(p.values()) - 1) < 1e-7 for p in probes), "max_prompt_tokens": max(lengths)}))
"""


SIZING_PROBE = """# sizing probe
import json, sys
from zils.d3 import Encoder, base_path
ticket, questions = json.loads(sys.argv[1]), json.loads(sys.argv[2])
encoder = Encoder(base_path())
def tokens(question, repeats):
    state = {"ticket": ticket, "history": "payment issue " * repeats}
    return len(encoder.encode(state, question)[0])
base = [tokens(q, 0) for q in questions]
per = [(tokens(q, 100) - b) / 100 for q, b in zip(questions, base)]
print(json.dumps({"overhead": max(base), "per_repeat": max(per)}))
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


def repeats_for(target_tokens, overhead, per_repeat):
    """Repeats of PHRASE that bring a row to about target_tokens, from measured token costs."""
    if per_repeat <= 0:
        raise ValueError(f"measured {per_repeat} tokens per repeat")
    return max(0, int((target_tokens - overhead) // per_repeat))


def synthetic_rows(path, state):
    rows = []
    for question in QUESTIONS * 2:
        label = (
            next(iter(question["criteria"]))
            if question["type"] == "choice"
            else "true"
            if question["type"] == "noul"
            else "1"
        )
        rows.append({"state": state, "questions": {"decision": dict(question, label=label)}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def parity_cases(path, rows, states):
    """The six training rows plus one choice row per extra state (about 1,000, 2,000, cap - 200)."""
    cases = [{"state": r["state"], "question": QUESTIONS[i % 3]} for i, r in enumerate(rows)]
    cases += [{"state": s, "question": QUESTIONS[0]} for s in states]
    path.write_text(
        "".join(json.dumps(dict(case, id=f"case-{n}")) + "\n" for n, case in enumerate(cases))
    )


def qualify(root, runtime_python, out, run=subprocess.run, free_mib=None, model=models.H2O):
    out.mkdir(parents=True, exist_ok=True)
    (out / "qualification.json").unlink(missing_ok=True)  # never leave a stale passing report
    module = "zils" + models.TEXT_RUNTIME_MODULES[model]
    cap = models.spec(model)["max_input_tokens"]
    report = {
        "model": model,
        **models.profile_identity(model),
        "trainer_sha256": trainer_identity(model),
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
            free_mib is not None or gpu_ready("cuda", model=model)
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
                f"from {module} import verify_runtime; verify_runtime(); print('ok')",
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
                [runtime_python, "-m", module, "reference", "--out", str(reference)],
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=env_root,
            )
        if model in models.d3_family():
            stage = "sizing"
            sized = run(
                [runtime_python, "-c", SIZING_PROBE, json.dumps(STATE["ticket"])]
                + [json.dumps(QUESTIONS)],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=env_root,
            )
            report["sizing"] = json.loads(sized.stdout)

            def history(target):
                return PHRASE * repeats_for(target, **report["sizing"])

            state = dict(STATE, history=history(cap - HEADROOM_TOKENS))
        else:
            # H2O's 950 repeats fill its 2,048-token cap.
            state = dict(STATE, history=STATE["history"] * (cap // H2O_CAP))
        stage = "train"
        data = out / "train.jsonl"
        rows = synthetic_rows(data, state)
        examples = len(rows)
        run(
            [
                runtime_python,
                "-m",
                module,
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
                RELOAD_PREDICT.replace("MODULE", module),
                str(adapter),
                json.dumps(state),
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
        if model in models.d3_family() and report["reload_predict_ok"]:
            stage = "parity"
            cases = out / "parity.jsonl"
            targets = (1000, 2000, cap - HEADROOM_TOKENS)
            parity_cases(cases, rows, [dict(STATE, history=history(n)) for n in targets])
            checked = run(
                [runtime_python, "-m", module, "parity"]
                + ["--reference", str(reference), "--cases", str(cases)],
                check=True,
                capture_output=True,
                text=True,
                timeout=900,
                cwd=env_root,
            )
            report["parity"] = json.loads(checked.stdout)
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
    parser.add_argument("--model", choices=(models.H2O, *models.d3_family()), default=models.H2O)
    parser.add_argument(
        "--runtime-python", required=True, help=".venv-h2o/bin/python or .venv-d3/bin/python"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = qualify(
        Path(args.root).resolve(),
        args.runtime_python,
        Path(args.out).resolve(),
        model=args.model,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
