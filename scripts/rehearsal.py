"""Two local miners and one validator: signed artifacts and dry-run rewards."""

import argparse
import json
import math
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.request import Request

from bittensor_wallet import Keypair

import zils
from zils.protocol import (
    BASE_REVISION,
    MAX_ANNOUNCEMENT,
    Handler,
    canonical,
    endpoint_ok,
    fetch_checkpoint,
    local_server,
    opener,
    register,
    write_json,
)


def miner(config):
    work = Path(config["work"])
    key = Keypair.create_from_seed(config["seed"])
    entry = zils.submission(config["checkpoint"], config["uid"])
    checkpoint = work / "artifacts"
    zils.stage(entry, checkpoint)

    class Artifacts(Handler):
        def do_GET(self):
            files = {"/artifacts/" + name: checkpoint / name for name in zils.ARTIFACT_FILES}
            if self.path not in files:
                self.reply(404, {"error": "unknown artifact"})
                return
            with files[self.path].open("rb") as source:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(os.fstat(source.fileno()).st_size))
                self.end_headers()
                shutil.copyfileobj(source, self.wfile)

    endpoint_ok(config["validator"])
    with local_server(Artifacts) as endpoint:
        claim = {
            "round_id": config["round_id"],
            "uid": config["uid"],
            "hotkey": key.ss58_address,
            "sha256": entry["sha256"],
            "endpoint": endpoint,
        }
        message = {"claim": claim, "signature": key.sign(canonical(claim)).hex()}
        write_json(work / "announcement.json", message)
        request = Request(
            config["validator"] + "/submit",
            json.dumps(message).encode(),
            {"Content-Type": "application/json"},
            method="POST",
        )
        with opener().open(request, timeout=10) as response:
            if json.loads(response.read(MAX_ANNOUNCEMENT))["status"] != "accepted":
                raise ValueError("validator did not accept the submission")
        print(f"miner {config['uid']}: signed checkpoint announced at {endpoint}", flush=True)
        # The launcher terminates this process after evaluation, or on any failure.
        threading.Event().wait()


def validator(config):
    work = Path(config["work"])
    cases = [
        json.loads(line) for line in Path(config["cases"]).read_text().splitlines() if line.strip()
    ]
    zils.validate_cases(cases)
    members = {int(uid): key for uid, key in config["members"].items()}
    if (
        len(members) != 2
        or len(set(members.values())) != 2
        or any(not 0 <= uid <= 65535 for uid in members)
    ):
        raise ValueError("this rehearsal requires two unique local identities")
    registry, condition = {}, threading.Condition()

    class Announcements(Handler):
        def do_POST(self):
            if self.path != "/submit":
                self.reply(404, {"error": "unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "-1"))
                if not 0 < size <= MAX_ANNOUNCEMENT or self.headers.get("Transfer-Encoding"):
                    raise ValueError("invalid announcement size")
                message = json.loads(self.rfile.read(size))
                with condition:
                    claim = register(message, config["round_id"], members, registry)
                    condition.notify_all()
                self.reply(200, {"status": "accepted", "uid": claim["uid"]})
                print(f"validator: authenticated miner {claim['uid']}", flush=True)
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.reply(400, {"error": str(error)[:300]})

    with local_server(Announcements) as endpoint:
        write_json(work / "ready.json", {"endpoint": endpoint, "round_id": config["round_id"]})
        deadline = time.monotonic() + 60
        with condition:
            while len(registry) < len(members):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("timed out waiting for both miners to announce")
                condition.wait(min(remaining, 1))
    write_json(work / "announcements.json", [registry[uid] for uid in sorted(registry)])
    downloads = work / "downloads"
    downloads.mkdir()
    entries, participants, failed = [], [], {}
    for uid in sorted(registry):
        c = registry[uid]["claim"]
        path = downloads / str(uid)
        participant = {**c, "signature_verified": True}
        try:
            participant.update(fetch_checkpoint(c, path))
            print(f"validator: verified checkpoint bytes for miner {uid}", flush=True)
        except (ValueError, OSError) as error:
            failed[uid] = f"artifact download rejected: {error}"
            # A missing path makes the existing evaluator reject this candidate.
            path = downloads / f"rejected-{uid}"
            participant["download_error"] = failed[uid]
        entries.append({"uid": uid, "checkpoint": str(path), "sha256": c["sha256"]})
        participants.append(participant)
    write_json(work / "submissions.json", entries)
    args = argparse.Namespace(
        cases=config["cases"],
        submissions=str(work / "submissions.json"),
        base_revision=config["base_revision"],
        device=config["device"],
        runner_python=config["runner_python"],
        timeout=config["timeout"],
        report=str(work / "evaluation.json"),
    )
    zils.evaluate(args)
    report = json.loads((work / "evaluation.json").read_text())
    for row in report["miners"]:
        if row["uid"] in failed:
            row["error"] = failed[row["uid"]]
    report.update(
        mode="local-network-rehearsal",
        round_id=config["round_id"],
        identity_source="local-allowlist",
        chain_write=False,
        participants=participants,
    )
    write_json(work / "report.json", report)
    print("validator: completed local reward allocation", flush=True)


def run(args):
    root = Path(args.out).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    round_id = uuid.uuid4().hex
    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    zils.validate_cases(cases)
    entries = [zils.submission(path, uid) for uid, path in enumerate(args.checkpoints, 1)]
    zils.validate_submissions(entries)
    work = root / "validator"
    work.mkdir(mode=0o700)
    private_cases = work / "cases.jsonl"
    private_cases.write_text("\n".join(json.dumps(case) for case in cases) + "\n")
    seeds = ["0x" + secrets.token_hex(32) for _ in entries]
    members = {
        entry["uid"]: Keypair.create_from_seed(seed).ss58_address
        for entry, seed in zip(entries, seeds)
    }
    config = {
        "work": str(work),
        "cases": str(private_cases),
        "members": members,
        "round_id": round_id,
        "base_revision": args.base_revision,
        "device": args.device,
        "timeout": args.timeout,
        # Preserve a venv's executable symlink: resolving it discards its installed packages.
        "runner_python": str(Path(args.runner_python).absolute()),
    }
    write_json(work / "config.json", config)
    children, logs, processes = [], [], []

    def launch(role, directory):
        log = (directory / "process.log").open("x")
        logs.append(log)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.rehearsal",
                role,
                "--config",
                str(directory / "config.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=zils.ROOT,
        )
        children.append(process)
        processes.append(
            {"role": role if role == "validator" else directory.name, "pid": process.pid}
        )
        return process

    try:
        validating = launch("validator", work)
        deadline = time.monotonic() + 30
        while True:
            if validating.poll() is not None:
                raise RuntimeError(
                    f"validator exited during startup; inspect {work / 'process.log'}"
                )
            try:
                ready = json.loads((work / "ready.json").read_text())
                break
            except (FileNotFoundError, json.JSONDecodeError):
                if time.monotonic() > deadline:
                    raise RuntimeError("validator startup timed out")
                time.sleep(0.1)
        for entry, seed in zip(entries, seeds):
            directory = root / f"miner-{entry['uid']}"
            directory.mkdir(mode=0o700)
            write_json(
                directory / "config.json",
                {
                    "work": str(directory),
                    "uid": entry["uid"],
                    "seed": seed,
                    "checkpoint": entry["checkpoint"],
                    "round_id": round_id,
                    "validator": ready["endpoint"],
                },
            )
            launch("miner", directory)
        write_json(root / "processes.json", processes)
        print(
            "Two miner processes started; validator is discovering signed submissions.", flush=True
        )
        deadline = time.monotonic() + 120 + len(entries) * args.timeout
        while validating.poll() is None:
            if any(p.poll() is not None for p in children[1:]):
                raise RuntimeError(
                    f"a miner exited early; inspect miner process.log files in {root}"
                )
            if time.monotonic() > deadline:
                raise RuntimeError("rehearsal exceeded its deadline")
            time.sleep(0.2)
        if validating.returncode:
            raise RuntimeError(f"validator failed; inspect {work / 'process.log'}")
        report = json.loads((work / "report.json").read_text())
        print(
            json.dumps(
                {
                    "report": str(work / "report.json"),
                    "weights": report["weights"],
                    "evaluated": sum(m["status"] == "evaluated" for m in report["miners"]),
                }
            )
        )
    finally:
        for process in children:
            # Each actor owns a process group, including the validator's model workers.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in children:
            try:
                process.wait(timeout=6)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=6)
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run")
    launch.add_argument(
        "--checkpoints", nargs=2, default=["models/reference", "models/fez-local-probe"]
    )
    launch.add_argument("--cases", default="examples/diagnostics.jsonl")
    launch.add_argument("--out", required=True)
    launch.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    launch.add_argument("--runner-python", default=sys.executable)
    launch.add_argument("--base-revision", default=BASE_REVISION)
    launch.add_argument("--timeout", type=float, default=600)
    for role in ("miner", "validator"):
        commands.add_parser(role).add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            if os.name != "posix":
                raise ValueError("the rehearsal launcher requires macOS, Linux, or WSL")
            if not math.isfinite(args.timeout) or args.timeout <= 0:
                raise ValueError("timeout must be positive and finite")
            run(args)
        else:
            config = json.loads(Path(args.config).read_text())
            (miner if args.command == "miner" else validator)(config)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"zils rehearsal: {error}\n")


if __name__ == "__main__":
    main()
