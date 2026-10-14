# Fleet Nodes Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a bare DigitalOcean GPU droplet into a qualified Zils node (`miner` or `fleet-validator` role) with one script, prove it on one RTX 4000 Ada, and snapshot it.

**Architecture:** A pure-Python role module (`zils/node_role.py`) validates `/etc/zils-node/role.json` and names the systemd units to enable; a qualification runner (`scripts/qualify_node.py`) drives the pinned H2O runtime through the same checks the 10 October smoke test used and writes a report; an idempotent shell provisioner (`scripts/provision_node.sh`) installs accounts, pinned checkouts, environments, references and unit files, then runs the qualifier and writes `image.json`. Serving and hosted-validator roles are out of scope (platform add-on, later phase).

**Tech Stack:** Bash, Python 3.13 (`.venv-kev`), Python 3.12 pinned H2O runtime (`.venv-h2o`), `uv`, systemd, `doctl`, `unittest`, `ruff`.

**Spec:** `docs/fleet-nodes.md` (branch `codex/fleet-nodes`, PR ooo-hq/zils#32)

## Global Constraints

- Nothing in `zils/` or `miner/` may import `zils_platform`, `zils_sdk`, `stripe` or `boto3` (`tests/test_repository_boundary.py`).
- Python target 3.13, ruff line length 100, lint rules `E4,E7,E9,F,I`, isort `combine-as-imports`; `make check` must pass.
- H2O runtime pins are authoritative: Python 3.12, PyTorch 2.10.0 CUDA 13.0, Transformers 5.17.0, PEFT 0.21.0, FLA 0.5.2, causal-conv1d 1.7.0 (`zils/h2o-pins.json`, `requirements/h2o.txt`). The qualifier never relaxes them.
- Training floor: `gpu_ready('cuda', model=H2O)` requires at least 12,288 MiB free. The qualifier records free memory and fails below the floor.
- Secrets never land in the image: no hotkey, coldkey, token or credential is read or written by the provisioner; `role.json` and `/etc/zils-node/hotkey/` arrive at boot.
- Prices quoted for the trial: `gpu-4000adax1-20gb` at $0.76/GPU-hour, per-second billing, 60-second minimum; destroy the droplet when done.

## Review Focus

1. `role.json` present but a required field missing or the role unknown: `zils.node_role` must raise a clear `ValueError` naming the field and enable nothing (Task 1 test).
2. Role valid but the wallet hotkey file absent at boot: refuse to enable units with a message naming the expected path; never fall back to a weaker role (Task 1 test).
3. Driver older than CUDA 13.0 on the droplet: provisioner stops at the driver check with the `nvidia-smi` version it saw, before downloading anything (Task 4 dry-run test asserts the check precedes the install phases).
4. Free GPU memory below 12,288 MiB when qualification starts (another process resident): qualifier reports `capacity_ok: false` with the measured number and exits non-zero without training (Task 2 test).
5. Re-running `provision_node.sh` on an already-provisioned node: every phase must be a no-op when its marker exists; the test runs the dry-run twice and asserts identical output (Task 4 test).

---

### Task 1: Role configuration module

**Files:**
- Create: `zils/node_role.py`
- Create: `deploy/role-examples/miner.json`
- Create: `deploy/role-examples/fleet-validator.json`
- Test: `tests/test_node_role.py`

**Interfaces:**
- Consumes: nothing from this plan.
- Produces: `load_role(path: Path) -> dict`, `required_files(role: dict) -> list[Path]`, `units_for(role: dict) -> list[str]`, `check(path: Path, root: Path = Path("/etc/zils-node")) -> list[str]` (returns unit names or raises `ValueError`), and `python -m zils.node_role check [--path PATH] [--root ROOT]` printing one unit per line.

- [ ] **Step 1: Write the failing tests**

```python
"""Node role configuration selects units and refuses incomplete identities."""

import json
import tempfile
import unittest
from pathlib import Path

from zils import node_role


def write(path, value):
    path.write_text(json.dumps(value))
    return path


class NodeRoleTest(unittest.TestCase):
    def miner(self):
        return {
            "role": "miner",
            "coordinator": "https://training.zils.ai",
            "hotkey": "5GnLjWkMA6UhM4K7k94j8tWrxPGvECzwbRMdbEgPAUeMbJuj",
            "wallet": {"name": "fez-testnet", "hotkey": "gpu4090", "path": "/etc/zils-node/wallets"},
            "device": "cuda",
        }

    def test_miner_role_enables_queue_miner_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wallets/fez-testnet/hotkeys").mkdir(parents=True)
            (root / "wallets/fez-testnet/hotkeys/gpu4090").write_text("{}")
            role = self.miner()
            role["wallet"]["path"] = str(root / "wallets")
            path = write(root / "role.json", role)
            self.assertEqual(node_role.check(path, root=root), ["zils-queue-miner.service"])

    def test_fleet_validator_role_enables_validator_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wallets/fez-testnet/hotkeys").mkdir(parents=True)
            (root / "wallets/fez-testnet/hotkeys/validator").write_text("{}")
            (root / "fleet-config.json").write_text("{}")
            role = {
                "role": "fleet-validator",
                "hotkey": "5HTAVUuHxV81VbYoX7DYFzebdyA1SRk3zazdZiiBN7zCcpxy",
                "wallet": {"name": "fez-testnet", "hotkey": "validator", "path": str(root / "wallets")},
                "fleet_config": str(root / "fleet-config.json"),
                "device": "cuda",
            }
            path = write(root / "role.json", role)
            self.assertEqual(node_role.check(path, root=root), ["zils-fleet-validator.service"])

    def test_unknown_role_and_missing_field_are_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "role"):
                node_role.check(write(root / "a.json", {"role": "serve-h2o"}), root=root)
            role = self.miner()
            del role["coordinator"]
            with self.assertRaisesRegex(ValueError, "coordinator"):
                node_role.check(write(root / "b.json", role), root=root)

    def test_missing_hotkey_file_refuses_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            role = self.miner()
            role["wallet"]["path"] = str(root / "wallets")
            with self.assertRaisesRegex(ValueError, "hotkeys/gpu4090"):
                node_role.check(write(root / "role.json", role), root=root)

    def test_examples_validate_against_schema_fields(self):
        root = Path(__file__).resolve().parents[1] / "deploy/role-examples"
        for name in ("miner.json", "fleet-validator.json"):
            role = node_role.load_role(root / name)
            self.assertIn(role["role"], ("miner", "fleet-validator"))
            self.assertTrue(node_role.units_for(role))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-kev/bin/python -m unittest tests.test_node_role -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zils.node_role'`

- [ ] **Step 3: Write the module and the two examples**

```python
"""Select the systemd units a node runs from /etc/zils-node/role.json.

A node has exactly one role. Missing identity files refuse the role; there is
no fallback to a weaker role. Serving and hosted-validator roles are installed
by the platform repository and are not accepted here.
"""

import argparse
import json
import sys
from pathlib import Path

ROLES = {
    "miner": {
        "fields": ("coordinator", "hotkey", "wallet", "device"),
        "units": ["zils-queue-miner.service"],
    },
    "fleet-validator": {
        "fields": ("hotkey", "wallet", "fleet_config", "device"),
        "units": ["zils-fleet-validator.service"],
    },
}
WALLET_FIELDS = ("name", "hotkey", "path")


def load_role(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value.get("role") not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    for field in ROLES[value["role"]]["fields"]:
        if field not in value:
            raise ValueError(f"role.json is missing required field: {field}")
    wallet = value["wallet"]
    if not isinstance(wallet, dict) or any(k not in wallet for k in WALLET_FIELDS):
        raise ValueError(f"wallet must contain {WALLET_FIELDS}")
    if value["device"] != "cuda":
        raise ValueError("fleet nodes run on cuda")
    return value


def required_files(role):
    wallet = role["wallet"]
    files = [Path(wallet["path"]) / wallet["name"] / "hotkeys" / wallet["hotkey"]]
    if role["role"] == "fleet-validator":
        files.append(Path(role["fleet_config"]))
    return files


def units_for(role):
    return list(ROLES[role["role"]]["units"])


def check(path, root=Path("/etc/zils-node")):
    role = load_role(path)
    missing = [str(p) for p in required_files(role) if not p.is_file()]
    if missing:
        raise ValueError("refusing role; missing identity files: " + ", ".join(missing))
    return units_for(role)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("check")
    cmd.add_argument("--path", default="/etc/zils-node/role.json")
    cmd.add_argument("--root", default="/etc/zils-node")
    args = parser.parse_args(argv)
    try:
        for unit in check(Path(args.path), root=Path(args.root)):
            print(unit)
    except (OSError, ValueError) as error:
        print(f"role check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`deploy/role-examples/miner.json`:

```json
{
  "role": "miner",
  "coordinator": "https://training.zils.ai",
  "hotkey": "REPLACE_WITH_PUBLIC_SS58",
  "wallet": {"name": "fez-testnet", "hotkey": "node-1", "path": "/etc/zils-node/wallets"},
  "device": "cuda"
}
```

`deploy/role-examples/fleet-validator.json`:

```json
{
  "role": "fleet-validator",
  "hotkey": "REPLACE_WITH_PUBLIC_SS58",
  "wallet": {"name": "fez-testnet", "hotkey": "validator", "path": "/etc/zils-node/wallets"},
  "fleet_config": "/etc/zils-node/fleet/validator/config.json",
  "device": "cuda"
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-kev/bin/python -m unittest tests.test_node_role tests.test_repository_boundary -v`
Expected: PASS (boundary test still passes; the module imports only stdlib)

- [ ] **Step 5: Lint and commit**

```bash
make lint
git add zils/node_role.py deploy/role-examples tests/test_node_role.py
git commit -m "Add node role configuration for miner and fleet-validator nodes"
```

---

### Task 2: Qualification runner

**Files:**
- Create: `scripts/qualify_node.py`
- Test: `tests/test_qualify_node.py`

**Interfaces:**
- Consumes: `zils.models.profile_identity(models.H2O)`, `zils.miner_grading.trainer_identity(models.H2O)`, `zils.runtime.gpu_ready`, the `zils.h2o reference` and `zils.h2o train` CLIs, `training_metrics.json` written by `zils.jevk5.train` (field `peak_gpu_allocated_bytes`).
- Produces: `qualify(root: Path, runtime_python: str, out: Path, run=subprocess.run, free_mib=None) -> dict` and `python scripts/qualify_node.py --root REPO --runtime-python PATH --out DIR`. The report dict is the contract Task 4 and the operator read:

```json
{
  "model": "h2o-lightning-4b-v1.2.3",
  "profile_sha256": "...", "runtime_sha256": "...", "trainer_sha256": "...",
  "gpu": {"name": "...", "driver": "...", "free_mib_at_start": 19500},
  "capacity_ok": true,
  "runtime_verified": true,
  "training": {"examples": 6, "optimizer_steps": 2, "seconds": 27.6, "peak_gpu_mib": 9759, "max_prompt_tokens": 2048},
  "reload_predict_ok": true,
  "completed_at": "2026-10-14T12:00:00+00:00"
}
```

- [ ] **Step 1: Write the failing tests**

```python
"""Qualification records pinned identities, capacity and a 2,048-token training step."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts import qualify_node
from zils import models


class QualifyNodeTest(unittest.TestCase):
    def fake_run(self, calls):
        def run(command, **kwargs):
            calls.append(command)
            text = " ".join(map(str, command))
            if "--query-gpu=name,driver_version" in text:
                return SimpleNamespace(returncode=0, stdout="NVIDIA RTX 4000 Ada, 580.95\n", stderr="")
            if "verify_runtime" in text:
                return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
            if " train " in text or "zils.h2o train" in text:
                out = Path(command[command.index("--out") + 1])
                out.mkdir(parents=True)
                (out / "training_metrics.json").write_text(
                    json.dumps({"examples": 6, "optimizer_steps": 2, "training_seconds": 27.6,
                                "peak_gpu_allocated_bytes": 10233864192, "max_tokens": 2048})
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if "reload_predict" in text:
                return SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "max_prompt_tokens": 2031}), stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return run

    def test_report_binds_identities_and_measurements(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(__file__).resolve().parents[1]
            report = qualify_node.qualify(root, "python3.12", Path(tmp) / "q", run=self.fake_run(calls), free_mib=19500)
        self.assertEqual(report["model"], models.H2O)
        self.assertEqual(report["profile_sha256"], models.profile_identity(models.H2O)["profile_sha256"])
        self.assertTrue(report["capacity_ok"] and report["runtime_verified"] and report["reload_predict_ok"])
        self.assertEqual(report["training"]["peak_gpu_mib"], 9759)
        self.assertEqual(report["training"]["max_prompt_tokens"], 2031)
        self.assertTrue(any("verify_runtime" in " ".join(map(str, c)) for c in calls))

    def test_below_floor_stops_before_training(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(__file__).resolve().parents[1]
            with self.assertRaises(SystemExit):
                qualify_node.qualify(root, "python3.12", Path(tmp) / "q", run=self.fake_run(calls), free_mib=12000)
            report = json.loads((Path(tmp) / "q" / "qualification.json").read_text())
        self.assertFalse(report["capacity_ok"])
        self.assertEqual(report["gpu"]["free_mib_at_start"], 12000)
        self.assertFalse(any("train" in " ".join(map(str, c)) for c in calls))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-kev/bin/python -m unittest tests.test_qualify_node -v`
Expected: FAIL with `ImportError: cannot import name 'qualify_node'`

- [ ] **Step 3: Write the runner**

```python
"""Qualify this host for the pinned H2O profile and write qualification.json.

Runs from .venv-kev; drives the isolated .venv-h2o runtime through subprocesses
so the pinned runtime check, a 2,048-token training step and an adapter reload
happen in the real environment. Mirrors the 10 October 2026 RTX 4090 check.
"""

import argparse
import json
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
    {"type": "choice", "instructions": "Route this ticket.", "criteria": {"billing": "payment", "support": "technical"}},
    {"type": "noul", "instructions": "Does the ticket mention payment?", "criteria": {"true": "payment", "false": "no payment"}},
    {"type": "score", "instructions": "Rate urgency.", "criteria": ["low", "medium", "high"]},
]
RELOAD_PREDICT = """
import json, sys
from zils.h2o import DecisionModel
adapter, state, questions = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
model = DecisionModel(adapter, "cuda")
lengths = [len(model.encode(state, q)[0]) for q in questions]
probes = [model.predict(state, q) for q in questions]
print(json.dumps({"ok": all(abs(sum(p.values()) - 1) < 1e-7 for p in probes), "max_prompt_tokens": max(lengths)}))
"""


def nvidia(run):
    result = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    name, driver = [x.strip() for x in result.stdout.strip().split(",", 1)]
    return {"name": name, "driver": driver}


def free_memory_mib(run):
    result = run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    return int(result.stdout.strip().splitlines()[0])


def synthetic_rows(path):
    rows = []
    for question in QUESTIONS * 2:
        label = next(iter(question["criteria"])) if question["type"] == "choice" else "true" if question["type"] == "noul" else "1"
        rows.append({"state": STATE, "questions": {"decision": dict(question, label=label)}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def qualify(root, runtime_python, out, run=subprocess.run, free_mib=None):
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "model": models.H2O,
        **models.profile_identity(models.H2O),
        "trainer_sha256": trainer_identity(models.H2O),
        "gpu": nvidia(run),
        "completed_at": None,
    }
    free = free_memory_mib(run) if free_mib is None else free_mib
    report["gpu"]["free_mib_at_start"] = free
    report["capacity_ok"] = free >= FLOOR_MIB and (free_mib is not None or gpu_ready("cuda", model=models.H2O))

    def finish(code):
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        (out / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
        if code:
            sys.exit(code)
        return report

    if not report["capacity_ok"]:
        report["error"] = f"free GPU memory {free} MiB is below the {FLOOR_MIB} MiB floor"
        return finish(2)
    env_root = str(root)
    verify = run([runtime_python, "-c", "from zils.h2o import verify_runtime; verify_runtime(); print('ok')"],
                 capture_output=True, text=True, timeout=300, cwd=env_root)
    report["runtime_verified"] = verify.returncode == 0 and "ok" in verify.stdout
    if not report["runtime_verified"]:
        report["error"] = verify.stderr[-2000:]
        return finish(3)
    reference = out / "reference"
    if not reference.exists():
        run([runtime_python, "-m", "zils.h2o", "reference", "--out", str(reference)], check=True, timeout=3600, cwd=env_root)
    data = out / "train.jsonl"
    examples = synthetic_rows(data)
    adapter = out / "adapter"
    run([runtime_python, "-m", "zils.h2o", "train", "--data", str(data), "--reference", str(reference),
         "--out", str(adapter), "--seed", "771", "--device", "cuda"], check=True, timeout=3600, cwd=env_root)
    metrics = json.loads((adapter / "training_metrics.json").read_text())
    probe = run([runtime_python, "-c", RELOAD_PREDICT, str(adapter), json.dumps(STATE), json.dumps(QUESTIONS)],
                capture_output=True, text=True, timeout=900, cwd=env_root)
    reload = json.loads(probe.stdout) if probe.returncode == 0 else {"ok": False, "max_prompt_tokens": None}
    report["training"] = {
        "examples": examples,
        "optimizer_steps": metrics["optimizer_steps"],
        "seconds": metrics["training_seconds"],
        "peak_gpu_mib": round(metrics["peak_gpu_allocated_bytes"] / 2**20),
        "max_prompt_tokens": reload["max_prompt_tokens"],
    }
    report["reload_predict_ok"] = bool(reload["ok"])
    return finish(0 if report["reload_predict_ok"] else 4)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="zils checkout (runtime imports zils from here)")
    parser.add_argument("--runtime-python", required=True, help=".venv-h2o/bin/python")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = qualify(Path(args.root).resolve(), args.runtime_python, Path(args.out).resolve())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-kev/bin/python -m unittest tests.test_qualify_node -v`
Expected: PASS

- [ ] **Step 5: Lint and commit**

```bash
make lint
git add scripts/qualify_node.py tests/test_qualify_node.py
git commit -m "Add the node qualification runner for the pinned H2O profile"
```

---

### Task 3: Unit files and role apply script

**Files:**
- Create: `deploy/units/zils-queue-miner.service`
- Create: `deploy/units/zils-fleet-validator.service`
- Create: `deploy/units/zils-node.env.example`
- Create: `scripts/apply_role.sh`
- Test: `tests/test_node_units.py`

**Interfaces:**
- Consumes: `python -m zils.node_role check` (Task 1) printing unit names.
- Produces: unit files under `/etc/systemd/system/` with identical names; `apply_role.sh` enables exactly the units `node_role` returns and disables the others in `ROLES`.

- [ ] **Step 1: Write the failing test**

```python
"""Every unit a role can enable ships in deploy/units with the pinned command shape."""

import re
import unittest
from pathlib import Path

from zils import node_role

ROOT = Path(__file__).resolve().parents[1]


class NodeUnitsTest(unittest.TestCase):
    def test_every_role_unit_exists_and_runs_from_pinned_environment(self):
        for role, spec in node_role.ROLES.items():
            for unit in spec["units"]:
                text = (ROOT / "deploy/units" / unit).read_text()
                self.assertIn("EnvironmentFile=/etc/zils-node/node.env", text, unit)
                self.assertRegex(text, r"ExecStart=/opt/zils/\.venv-kev/bin/python -m (miner\.queue|zils\.fleet)", unit)
                self.assertIn("ProtectSystem=strict", text, unit)
                self.assertNotIn("SUPABASE", text, unit)

    def test_apply_script_uses_role_check_and_never_hardcodes_units(self):
        text = (ROOT / "scripts/apply_role.sh").read_text()
        self.assertIn("zils.node_role check", text)
        self.assertNotIn("systemctl enable zils-queue-miner", text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-kev/bin/python -m unittest tests.test_node_units -v`
Expected: FAIL with `FileNotFoundError` for `deploy/units/zils-queue-miner.service`

- [ ] **Step 3: Write the units and the script**

`deploy/units/zils-queue-miner.service`:

```ini
[Unit]
Description=Zils queue miner (customer adapter training)
After=network-online.target
Wants=network-online.target
[Service]
User=zilsminer
Group=zilsminer
SupplementaryGroups=zils-compute
WorkingDirectory=/opt/zils
EnvironmentFile=/etc/zils-node/node.env
ExecStart=/opt/zils/.venv-kev/bin/python -m miner.queue --config /etc/zils-node/queue-miner.json --state /var/lib/zils-node/queue-miner --reference /opt/zils/models/h2o-reference --runtime-python /opt/zils/.venv-h2o/bin/python --device cuda
Restart=on-failure
RestartSec=15
KillMode=control-group
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/zils-node /var/lock/zils-training
[Install]
WantedBy=multi-user.target
```

`deploy/units/zils-fleet-validator.service`:

```ini
[Unit]
Description=Zils fleet validator (chain rounds)
After=network-online.target
Wants=network-online.target
[Service]
User=zilsminer
Group=zilsminer
SupplementaryGroups=zils-compute
WorkingDirectory=/opt/zils
EnvironmentFile=/etc/zils-node/node.env
ExecStart=/opt/zils/.venv-kev/bin/python -m zils.fleet validator --config /etc/zils-node/fleet/validator/config.json --runtime-python /opt/zils/.venv-h2o/bin/python --device cuda
Restart=on-failure
RestartSec=15
KillMode=control-group
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/zils-node /var/lock/zils-training /etc/zils-node/fleet
[Install]
WantedBy=multi-user.target
```

`deploy/units/zils-node.env.example`:

```bash
ZILS_H2O_RUNTIME_PYTHON=/opt/zils/.venv-h2o/bin/python
ZILS_H2O_BASE_DIR=/opt/zils/models/h2o-base
HF_HOME=/var/lib/zils-node/cache/hf
TRITON_CACHE_DIR=/var/lib/zils-node/cache/triton
TORCHINDUCTOR_CACHE_DIR=/var/lib/zils-node/cache/inductor
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
ZILS_COMPUTE_LOCK=/var/lock/zils-training/cuda.lock
PYTHONUNBUFFERED=1
```

`scripts/apply_role.sh`:

```bash
#!/usr/bin/env bash
# Enable exactly the units /etc/zils-node/role.json selects; disable the rest.
set -euo pipefail
ROOT=/opt/zils
mapfile -t WANTED < <("$ROOT/.venv-kev/bin/python" -m zils.node_role check)
if [ "${#WANTED[@]}" -eq 0 ]; then echo "role check selected no units" >&2; exit 1; fi
ALL=$(ls "$ROOT/deploy/units"/*.service | xargs -n1 basename)
for unit in $ALL; do
  if printf '%s\n' "${WANTED[@]}" | grep -qx "$unit"; then
    systemctl enable --now "$unit"
  else
    systemctl disable --now "$unit" 2>/dev/null || true
  fi
done
systemctl --no-pager --no-legend list-units 'zils-*' || true
```

The miner unit reads `/etc/zils-node/queue-miner.json`; `provision_node.sh` (Task 4) writes it from `role.json` (`coordinator`, `hotkey`, `wallet`) so the queue miner's existing config format is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-kev/bin/python -m unittest tests.test_node_units -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/apply_role.sh
git add deploy/units scripts/apply_role.sh tests/test_node_units.py
git commit -m "Ship node unit files and the role apply script"
```

---

### Task 4: Provisioning script

**Files:**
- Create: `scripts/provision_node.sh`
- Test: `tests/test_provision_script.py`

**Interfaces:**
- Consumes: `scripts/qualify_node.py` (Task 2), `scripts/apply_role.sh` and `deploy/units` (Task 3), `deploy/role-examples` (Task 1).
- Produces: a provisioned host with `/opt/zils` (pinned checkout, `.venv-kev`, `.venv-h2o`, `models/h2o-reference`), unit files installed and disabled, `/var/lib/zils-node/qualification/qualification.json`, and `/etc/zils-node/image.json`. Environment variables: `ZILS_SUBNET_REF` (default: `main`), `ZILS_NODE_DRY_RUN=1` prints the phase plan and exits 0 without changing anything.

- [ ] **Step 1: Write the failing test**

```python
"""The provisioner is idempotent, checks the driver first and never touches secrets."""

import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/provision_node.sh"


class ProvisionScriptTest(unittest.TestCase):
    def dry_run(self):
        return subprocess.run(["bash", str(SCRIPT)], env={**os.environ, "ZILS_NODE_DRY_RUN": "1"},
                              capture_output=True, text=True, timeout=60)

    def test_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(SCRIPT)]).returncode, 0)

    def test_dry_run_lists_phases_in_order_and_is_stable(self):
        first, second = self.dry_run(), self.dry_run()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        phases = [line.split(":")[1].strip() for line in first.stdout.splitlines() if line.startswith("phase:")]
        self.assertEqual(phases, ["driver", "accounts", "source", "environments", "references", "units", "qualify", "image"])

    def test_script_never_reads_secret_locations(self):
        text = SCRIPT.read_text()
        for forbidden in ("coldkey", "SUPABASE_SERVICE_ROLE_KEY", "ZILS_RUNTIME_TOKEN", "/etc/zils-node/hotkey", "/etc/zils-node/wallets"):
            self.assertNotIn(forbidden, text, forbidden)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-kev/bin/python -m unittest tests.test_provision_script -v`
Expected: FAIL (script missing)

- [ ] **Step 3: Write the script**

```bash
#!/usr/bin/env bash
# Provision a Zils fleet node from a bare GPU droplet. Idempotent: every phase
# writes a marker under /var/lib/zils-node/markers and skips itself when the
# marker exists. ZILS_NODE_DRY_RUN=1 prints the plan and changes nothing.
set -euo pipefail

ZILS_SUBNET_REF="${ZILS_SUBNET_REF:-main}"
ROOT=/opt/zils
STATE=/var/lib/zils-node
MARKERS=$STATE/markers
DRY="${ZILS_NODE_DRY_RUN:-0}"

phase() { echo "phase: $1"; }
done_marker() { [ "$DRY" = 1 ] || touch "$MARKERS/$1"; }
skip() { [ "$DRY" = 0 ] && [ -f "$MARKERS/$1" ]; }

phase driver
if [ "$DRY" = 0 ]; then
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing; use the NVIDIA AI/ML Ready image" >&2; exit 10; }
  CUDA_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  # CUDA 13.0 needs driver >= 580 (NVIDIA release notes); refuse before downloading anything.
  [ "$CUDA_MAJOR" -ge 580 ] || { echo "driver $CUDA_MAJOR is older than the CUDA 13.0 minimum (580)" >&2; exit 11; }
fi

phase accounts
if ! skip accounts; then
  if [ "$DRY" = 0 ]; then
    getent group zils-compute >/dev/null || groupadd zils-compute
    id zilsminer >/dev/null 2>&1 || useradd --system --home-dir /var/lib/zils-node --shell /usr/sbin/nologin -G zils-compute zilsminer
    install -d -m 0750 -o zilsminer -g zilsminer $STATE $STATE/queue-miner $STATE/cache $STATE/qualification $MARKERS
    install -d -m 0770 -o root -g zils-compute /var/lock/zils-training
    install -d -m 0750 /etc/zils-node /etc/zils-node/fleet
  fi
  done_marker accounts
fi

phase source
if ! skip source; then
  if [ "$DRY" = 0 ]; then
    apt-get update -q && apt-get install -y -q git python3.13 python3.13-venv curl >/dev/null
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if [ ! -d $ROOT/.git ]; then git clone -q https://github.com/ooo-hq/zils.git $ROOT; fi
    git -C $ROOT fetch -q origin && git -C $ROOT checkout -q "$ZILS_SUBNET_REF"
    chown -R root:zils-compute $ROOT && chmod -R g+rX $ROOT
  fi
  done_marker source
fi

phase environments
if ! skip environments; then
  if [ "$DRY" = 0 ]; then
    export PATH="$HOME/.local/bin:$PATH"
    cd $ROOT
    [ -x .venv-kev/bin/python ] || uv venv --python 3.13 .venv-kev
    uv pip install -q --python .venv-kev/bin/python --torch-backend=cu128 -r requirements/model.txt -r requirements/rehearsal.txt -r requirements/testnet.txt
    .venv-kev/bin/python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()'
    [ -x .venv-h2o/bin/python ] || uv venv --python 3.12 .venv-h2o
    uv pip install -q --python .venv-h2o/bin/python --torch-backend=cu130 -r requirements/h2o.txt
  fi
  done_marker environments
fi

phase references
if ! skip references; then
  if [ "$DRY" = 0 ]; then
    cd $ROOT
    install -d -m 0750 -o root -g zils-compute $STATE/cache/hf
    [ -f models/h2o-reference/model.json ] || HF_HOME=$STATE/cache/hf .venv-h2o/bin/python -m zils.h2o reference --out models/h2o-reference
    chown -R root:zils-compute models && chmod -R g+rX models
  fi
  done_marker references
fi

phase units
if ! skip units; then
  if [ "$DRY" = 0 ]; then
    install -m 0644 $ROOT/deploy/units/*.service /etc/systemd/system/
    [ -f /etc/zils-node/node.env ] || install -m 0640 -g zils-compute $ROOT/deploy/units/zils-node.env.example /etc/zils-node/node.env
    sed -i "s#^HF_HOME=.*#HF_HOME=$STATE/cache/hf#" /etc/zils-node/node.env
    systemctl daemon-reload
    for unit in $ROOT/deploy/units/*.service; do systemctl disable "$(basename "$unit")" 2>/dev/null || true; done
  fi
  done_marker units
fi

phase qualify
if ! skip qualify; then
  if [ "$DRY" = 0 ]; then
    cd $ROOT
    set -a; . /etc/zils-node/node.env; set +a
    .venv-kev/bin/python scripts/qualify_node.py --root $ROOT --runtime-python $ROOT/.venv-h2o/bin/python --out $STATE/qualification
  fi
  done_marker qualify
fi

phase image
if [ "$DRY" = 0 ]; then
  cd $ROOT
  PINS=$(sha256sum zils/h2o-pins.json requirements/h2o.txt | sha256sum | cut -c1-16)
  Q=$(sha256sum $STATE/qualification/qualification.json | cut -c1-16)
  printf '{"subnet_ref":"%s","subnet_commit":"%s","pins_sha256_16":"%s","qualification_sha256_16":"%s","built_at":"%s"}\n' \
    "$ZILS_SUBNET_REF" "$(git rev-parse HEAD)" "$PINS" "$Q" "$(date -u +%FT%TZ)" > /etc/zils-node/image.json
  echo "image: $(cat /etc/zils-node/image.json)"
fi
echo "provisioned; apply a role with: scripts/apply_role.sh after writing /etc/zils-node/role.json"
```

Note for the miner role: `apply_role.sh` expects `/etc/zils-node/queue-miner.json`. Add this helper at the end of `scripts/apply_role.sh` before the enable loop, so the queue miner's existing config format is produced from `role.json`:

```bash
"$ROOT/.venv-kev/bin/python" - <<'PY'
import json, pathlib
role = json.loads(pathlib.Path("/etc/zils-node/role.json").read_text())
if role["role"] == "miner":
    cfg = {"coordinator": role["coordinator"], "hotkey": role["hotkey"], "wallet": role["wallet"]}
    p = pathlib.Path("/etc/zils-node/queue-miner.json"); p.write_text(json.dumps(cfg, indent=2) + "\n"); p.chmod(0o640)
PY
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-kev/bin/python -m unittest tests.test_provision_script tests.test_node_units -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/provision_node.sh
git add scripts/provision_node.sh scripts/apply_role.sh tests/test_provision_script.py
git commit -m "Add the idempotent node provisioner with a dry-run phase plan"
```

---

### Task 5: Documentation

**Files:**
- Modify: `docs/fleet-nodes.md` (section "Provisioning and rebuild")
- Modify: `docs/queue-miners.md` (after "## 1. Confirm access and hardware")
- Create: `docs/fleet-node-runbook.md`

- [ ] **Step 1: Write the runbook**

```markdown
# Fleet node runbook (DigitalOcean)

Create, qualify, snapshot and destroy a node. Prices are DigitalOcean list
as of August 2026; recheck before a long-lived node.

## Create the trial node

    doctl compute ssh-key list
    doctl compute droplet create zils-node-trial --region nyc2 \
      --size gpu-4000adax1-20gb --image gpu-h100x1-base \
      --ssh-keys <SSH_KEY_ID> --tag-names zils-fleet --wait
    doctl compute droplet get zils-node-trial --format PublicIPv4

If nyc2 rejects the size, try `tor1` then `atl1`.

## Provision and qualify

    ssh root@<ip> 'curl -fsSL https://raw.githubusercontent.com/ooo-hq/zils/main/scripts/provision_node.sh | ZILS_SUBNET_REF=main bash'
    ssh root@<ip> cat /var/lib/zils-node/qualification/qualification.json

Record `profile_sha256`, `runtime_sha256`, `trainer_sha256`,
`training.peak_gpu_mib` and `gpu.free_mib_at_start`; the operator registers
them in `zils_worker_profiles` for this image hash.

## Snapshot (no secrets on the node yet)

    doctl compute droplet-action snapshot <droplet-id> --snapshot-name "zils-node-$(ssh root@<ip> jq -r .pins_sha256_16 /etc/zils-node/image.json)" --wait

## Assign a role

Copy only the node's hotkey file into `/etc/zils-node/wallets/<wallet>/hotkeys/`,
write `/etc/zils-node/role.json` from `deploy/role-examples/`, then:

    ssh root@<ip> /opt/zils/scripts/apply_role.sh

## Destroy

    doctl compute droplet delete zils-node-trial --force

A powered-off droplet still bills; destroy, do not power off.
```

- [ ] **Step 2: Update the spec's provisioning section and the miner guide**

In `docs/fleet-nodes.md`, replace the sentence `scripts/provision_node.sh (new, this repository) is idempotent and runs on a bare droplet or an already-provisioned one:` with `scripts/provision_node.sh is idempotent and runs on a bare droplet or an already-provisioned one (see the [runbook](fleet-node-runbook.md)):`. In `docs/queue-miners.md`, after the "## 1. Confirm access and hardware" section, add:

```markdown
Operators standing up a rented node can run `scripts/provision_node.sh` instead
of steps 2–3; it performs the same installation and writes a qualification
report. See [fleet nodes](fleet-nodes.md) and the [runbook](fleet-node-runbook.md).
```

- [ ] **Step 3: Check links and commit**

Run: `grep -n "fleet-node-runbook.md" docs/fleet-nodes.md docs/queue-miners.md` (expect two hits) and `make check`.

```bash
git add docs/fleet-node-runbook.md docs/fleet-nodes.md docs/queue-miners.md
git commit -m "Document node provisioning and the DigitalOcean runbook"
```

---

### Task 6: Trial on one RTX 4000 Ada (operator-run, billed)

**Files:**
- Create (private, not committed): `skills/.private/fleet-trial-20261014/` with `create.json`, `provision.log`, `qualification.json`, `snapshot.json`, `destroy.json`

**Interfaces:**
- Consumes: everything above, pushed to `codex/fleet-nodes` so the `curl` in the runbook can fetch the script from that branch (`ZILS_SUBNET_REF=codex/fleet-nodes` and the raw URL on that branch until merged).

- [ ] **Step 1: Confirm the budget with the user**: one `gpu-4000adax1-20gb` for up to two hours, about $1.60, destroyed at the end.
- [ ] **Step 2: Create** with the runbook command; record `create.json` (id, region, ip; no keys).
- [ ] **Step 3: Provision** with `ZILS_SUBNET_REF=codex/fleet-nodes`; capture stdout to `provision.log`. Expected: eight `phase:` lines, then `image:`.
- [ ] **Step 4: Read `qualification.json`**. Expected: `capacity_ok`, `runtime_verified`, `reload_predict_ok` all true; `training.max_prompt_tokens` between 1,900 and 2,048; `training.peak_gpu_mib` below 19,000. If `runtime_verified` is false, record the error and stop; that is the Ada gate failing.
- [ ] **Step 5: Snapshot**, record `snapshot.json` (snapshot id, name, size).
- [ ] **Step 6: Destroy**, record `destroy.json` with the final billed minutes from `doctl compute droplet get` before deletion.
- [ ] **Step 7: Record** the outcome (numbers, image hash, snapshot id) in `skills/.private/project-context.md` and the PR description.

---

## Self-review notes

- Spec coverage: node image (Tasks 2–4), roles for `miner` and `fleet-validator` (Tasks 1, 3), boot-time injection (Task 3 units read `/etc/zils-node/*`; the provisioner never writes secrets, Task 4 test), rebuild via script and snapshot (Tasks 4, 5, 6), costs (Task 5 runbook, Task 6). Serving roles, release delivery, H2O fleet support and Tailscale automation are later phases by design.
- Names used consistently: `node_role.check/load_role/units_for/required_files/ROLES`; `qualify_node.qualify`; unit names `zils-queue-miner.service`, `zils-fleet-validator.service`; paths `/opt/zils`, `/var/lib/zils-node`, `/etc/zils-node`.
- Review Focus items are pinned by: 1 and 2 (Task 1 tests 3 and 4), 3 (Task 4 dry-run order test plus the driver exit codes), 4 (Task 2 test 2), 5 (Task 4 stability test).
