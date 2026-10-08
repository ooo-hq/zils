"""Assign validated jobs and activate accepted adapters using explicit operator policy."""

import argparse
import grp
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bittensor_wallet import Keypair

from . import models, version_selection
from .adapter_releases import publish, register, registry_entry
from .cloud import APIError, Supabase, trusted_url
from .coordinator import ASSIGNMENTS, JOBS
from .model_names import model_name, valid_customer_aliases
from .runtime import locked


class Workflow:
    def __init__(self, store, hotkey, releases, capacity, activate):
        self.store, self.hotkey, self.releases = store, hotkey, Path(releases)
        self.capacity, self.activate = capacity, activate
        self.accepted_offset = 0
        self.pending_offset = 0

    def status(self, job, state, message, **extra):
        result = job.get("result") or {}
        value = {"state": state, "message": message, **extra}
        if result.get("workflow") == value:
            return
        values = {"result": {**result, "workflow": value}}
        # Completion time is immutable release provenance, not an activation heartbeat.
        if job["status"] != "completed":
            values["updated_at"] = datetime.now(timezone.utc).isoformat()
        changed = self.store.patch(
            JOBS,
            f"id=eq.{job['id']}&status=eq.{job['status']}",
            values,
        )
        if changed:
            job.update(changed[0])

    def tick(self):
        accepted = self.store.rows(
            JOBS,
            "status=eq.completed&result->delivery->>status=eq.accepted&"
            "or=(result->workflow->>state.is.null,result->workflow->>state.not.in.(ready,needs_review))&"
            f"order=created_at.asc,id.asc&limit=1&offset={self.accepted_offset}",
        )
        self.accepted_offset = self.accepted_offset + 1 if accepted else 0
        for job in accepted:
            result = job.get("result") or {}
            if result.get("delivery", {}).get("status") != "accepted" or result.get(
                "workflow", {}
            ).get("state") in ("ready", "needs_review"):
                continue
            try:
                self.status(job, "activating", "Training passed. Preparing your API model.")
                release = publish(self.store, job["id"], self.releases)
                if release is None:
                    continue
                model_id = self.activate(release)
                if model_id != release["release_id"]:
                    raise ValueError("Activation identity differs from accepted release")
                self.status(
                    job,
                    "ready",
                    "Your model is ready to use with your existing API key.",
                    model_id=model_id,
                    model_name=model_name(job["name"], job["id"]),
                    fingerprint=release["fingerprint"],
                    **(
                        {"model_alias": "zils-task-" + release["selection"]["root_job_id"]}
                        if "selection" in release
                        else {}
                    ),
                )
            except version_selection.StaleVersion:
                self.status(
                    job,
                    "needs_review",
                    "A newer model is already active. Start a new run comparing against that version.",
                )
            except (
                APIError,
                OSError,
                ValueError,
                KeyError,
                TypeError,
                requests.RequestException,
                subprocess.SubprocessError,
            ):
                self.status(
                    job,
                    "activation_failed",
                    "Training passed. API activation will retry automatically.",
                )
        pending = self.store.rows(
            JOBS,
            f"status=eq.awaiting_approval&order=created_at.asc,id.asc&limit=100&offset={self.pending_offset}",
        )
        self.pending_offset = self.pending_offset + 100 if len(pending) == 100 else 0
        if not pending:
            return
        workers = self.store.rows(
            "fez_training_workers", f"hotkey=eq.{self.hotkey}&enabled=eq.true"
        )
        enabled = any(w["hotkey"] == self.hotkey and w["enabled"] for w in workers)
        busy = self.store.rows(
            ASSIGNMENTS,
            f"select=job_id,{JOBS}!inner(status)&hotkey=eq.{self.hotkey}&"
            f"state=in.(ready,leased)&{JOBS}.status=in.(queued,running)&limit=1",
        )
        try:
            capacity = self.capacity() if enabled and not busy else {"ready": False}
        except (OSError, ValueError, subprocess.SubprocessError):
            capacity = {"ready": False}
        for job in pending:
            manifest = job.get("manifest") or {}
            if (
                manifest.get("data_access") != "approved-workers-training-export"
                or models.job_model(job) != models.JEVK5
            ):
                self.status(
                    job, "needs_review", "This run needs an operator review before training."
                )
                continue
            if not enabled:
                self.status(job, "waiting_worker", "Waiting for an approved training worker.")
            elif busy:
                self.status(job, "waiting_worker", "Another run is using the training worker.")
            elif not capacity.get("ready"):
                self.status(
                    job,
                    "waiting_capacity",
                    "Waiting for GPU capacity. Your examples are saved; training will start automatically.",
                )
            else:
                # The RPC checks status and enabled workers again under a row lock.
                self.store.rpc(
                    "fez_approve_training_job", {"p_job": job["id"], "p_hotkeys": [self.hotkey]}
                )
                busy = True


def gpu_capacity(min_free_mib, command, services=()):
    if services:
        result = subprocess.run(
            ["systemctl", "is-active", *services], capture_output=True, timeout=10, check=False
        )
        if result.returncode:
            return {"ready": False}
    result = subprocess.run(
        [command, "--query-gpu=memory.free", "--format=csv,noheader,nounits", "--id=0"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    available = int(result.stdout.strip())
    return {"ready": available >= min_free_mib, "free_mib": available}


def activate(release, config):
    root = Path(config["releases"])
    path = root / release["release_id"]
    if config.get("release_group"):
        gid = grp.getgrnam(config["release_group"]).gr_gid
        for file in [path, *path.iterdir()]:
            if file.is_symlink():
                raise ValueError("Release cannot contain symlinks")
            os.chown(file, -1, gid)
            file.chmod(0o750 if file.is_dir() else 0o440)
    runtime_url = trusted_url(config["runtime_url"])
    with requests.get(
        runtime_url + "/health",
        headers={"Authorization": "Bearer " + os.environ[config["token_env"]]},
        timeout=(5, 30),
        allow_redirects=False,
    ) as response:
        response.raise_for_status()
        identities = response.json().get("models", [])
    identity = {k: release[k] for k in ("release_id", "fingerprint")}
    if identity not in identities:
        raise ValueError("Serving runtime has not verified the accepted adapter")
    entry = registry_entry(
        release,
        config.get("gateway_runtime_url", runtime_url),
        config["token_env"],
        name=json.loads((path / "source.json").read_text())["name"],
    )
    if config.get("registry"):
        register(config["registry"], entry, selection=release.get("selection"))
    else:
        # Operator-owned argv only. The child receives no Supabase credential.
        command = config["register_command"]
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(x, str) for x in command)
        ):
            raise ValueError("Supply an explicit registration command argv")
        environment = {
            k: v
            for k, v in os.environ.items()
            if k not in ("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_DB_URL")
        }
        payload = (
            {"entry": entry, "selection": release["selection"]} if "selection" in release else entry
        )
        try:
            result = subprocess.run(
                command,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=45,
                check=True,
                env=environment,
            )
        except subprocess.CalledProcessError as error:
            if error.returncode == 3:
                raise version_selection.StaleVersion("A newer model is already active") from None
            raise
        if json.loads(result.stdout).get("model_id") != entry["id"]:
            raise ValueError("Registry did not acknowledge this release")
    return entry["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--once", action="store_true")
    registration = sub.add_parser("register")
    registration.add_argument("--registry", type=Path, required=True)
    registration.add_argument("--customer-only", action="store_true")
    registration.add_argument("--runtime-url")
    registration.add_argument("--token-env")
    args = parser.parse_args()
    if args.action == "register":
        raw = sys.stdin.read(16385)
        if len(raw) > 16384:
            raise ValueError("Registry entry exceeds the size limit")
        payload = json.loads(raw)
        selection = None
        if isinstance(payload, dict) and set(payload) == {"entry", "selection"}:
            entry, selection = payload["entry"], payload["selection"]
            if not isinstance(selection, dict):
                raise ValueError("Supply frozen version selection")
        else:
            entry = payload
        if args.customer_only and (
            entry.get("url") != args.runtime_url
            or entry.get("token_env") != args.token_env
            or not valid_customer_aliases(entry, selection)
            or not isinstance(entry.get("owners"), list)
            or len(entry["owners"]) != 1
            or not re.fullmatch(r"zils-adapter-[a-f0-9-]{36}-[a-f0-9]{64}", entry.get("id", ""))
        ):
            raise ValueError("Only private adapters at the configured runtime may be registered")
        try:
            register(args.registry, entry, selection=selection)
        except version_selection.StaleVersion:
            parser.exit(3, "A newer version is active; evaluate against that version.\n")
        print(json.dumps({"model_id": entry["id"]}))
        return
    config = json.loads(args.config.read_text())
    Keypair(ss58_address=config["hotkey"])
    minimum = config["min_free_mib"]
    if type(minimum) is not int or minimum < 1:
        raise ValueError("Set a measured positive GPU memory requirement")
    root = Path(config["releases"])
    root.mkdir(mode=0o750, parents=True, exist_ok=True)
    flow = Workflow(
        Supabase(),
        config["hotkey"],
        root,
        lambda: gpu_capacity(
            minimum, config.get("nvidia_smi", "nvidia-smi"), config.get("training_services", [])
        ),
        lambda release: activate(release, config),
    )
    with locked(root / ".workflow.lock"):
        while True:
            try:
                flow.tick()
            except (
                APIError,
                OSError,
                ValueError,
                requests.RequestException,
                subprocess.SubprocessError,
            ) as error:
                print(f"Workflow interrupted ({type(error).__name__}); retrying.", flush=True)
                if args.once:
                    raise
            if args.once:
                return
            time.sleep(15)


if __name__ == "__main__":
    main()
