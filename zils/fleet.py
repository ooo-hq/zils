"""Create a private miner fleet and launch its services."""

import argparse
import json
import math
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

from bittensor_wallet import Keypair

import zils
from miner.worker import miner

from . import ROOT, protocol as wire
from .runtime import digest, locked, prepare_base, signing_key
from .validator import validator


def initialize(out, benchmark_path, checkpoint, host, port, miner_ports, identities=None):
    from . import benchmark

    endpoint = f"http://{host}:{port}"
    wire.endpoint_ok(endpoint, allowed=[endpoint])
    if not 1 <= len(miner_ports) <= 16 or len(set([port, *miner_ports])) != len(miner_ports) + 1:
        raise ValueError("require 1..16 miners and distinct validator/miner ports")
    for p in miner_ports:
        wire.endpoint_ok(f"http://{host}:{p}", allowed=[f"http://{host}:{p}"])
    if identities is not None:
        from .testnet import check_config

        check_config(identities)
        if len(identities["miners"]) != len(miner_ports):
            raise ValueError("one registered identity is required per miner port")
        zils.weight_vector([{**m, "skill": 0.0} for m in identities["miners"]])
        hotkeys = [identities["validator"]["hotkey"], *[m["hotkey"] for m in identities["miners"]]]
        if len(set(hotkeys)) != len(hotkeys):
            raise ValueError("validator and miners must have distinct hotkeys")
        for item in [identities["validator"], *identities["miners"]]:
            Keypair(ss58_address=item["hotkey"])
            if (
                set(item["wallet"]) - {"name", "hotkey", "path"}
                or not {"name", "hotkey"} <= item["wallet"].keys()
            ):
                raise ValueError(
                    "wallet requires name and hotkey, with optional path; never put keys in the roster"
                )
    benchmark_path = Path(benchmark_path)
    benchmark.audit(benchmark_path)
    manifest = json.loads((benchmark_path / "manifest.json").read_text())
    entry = zils.submission(checkpoint, 0)
    out = Path(out)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    validator_dir = out / "validator"
    validator_dir.mkdir(mode=0o700)
    data = validator_dir / "benchmark"
    data.mkdir(mode=0o700)
    for name in (*benchmark.FILES, "manifest.json"):
        shutil.copyfile(benchmark_path / name, data / name)
        (data / name).chmod(0o600)
    validator_seed = "0x" + secrets.token_hex(32)
    validator_identity = {"seed": validator_seed}
    validator_hotkey = Keypair.create_from_seed(validator_seed).ss58_address
    if identities is not None:
        validator_identity = {"wallet": identities["validator"]["wallet"]}
        validator_hotkey = identities["validator"]["hotkey"]
    shared = {
        "validator": endpoint,
        "validator_hotkey": validator_hotkey,
        "base_revision": wire.BASE_REVISION,
        "initial_sha256": entry["sha256"],
        "training_sha256": digest(data / "miner-training.jsonl"),
    }
    if "job_id" in manifest:
        shared.update(job_id=manifest["job_id"], job_sha256=digest(data / "manifest.json"))
        zils.stage(entry, validator_dir / "reference")
    if identities is not None:
        shared["chain"] = dict(identities["chain"])
    members = {}
    for index, miner_port in enumerate(miner_ports, 1):
        identity = identities["miners"][index - 1] if identities is not None else None
        uid = identity["uid"] if identity else index
        directory = out / f"miner-{uid}"
        directory.mkdir(mode=0o700)
        seed = "0x" + secrets.token_hex(32)
        key = Keypair.create_from_seed(seed)
        secret = {"wallet": identity["wallet"]} if identity else {"seed": seed}
        hotkey = identity["hotkey"] if identity else key.ss58_address
        members[uid] = {"hotkey": hotkey, "port": miner_port}
        wire.write_json(
            directory / "config.json",
            {**shared, "uid": uid, **secret, "hotkey": hotkey, "port": miner_port},
        )
        shutil.copyfile(data / "miner-training.jsonl", directory / "miner-training.jsonl")
        zils.stage(entry, directory / "reference")
        for name in ("zils", "fez", "miner", "requirements"):
            shutil.copytree(
                ROOT / name, directory / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
        (directory / "docs").mkdir()
        for name in ("mining.md", "testnet.md"):
            shutil.copyfile(ROOT / "docs" / name, directory / "docs" / name)
        (directory / "README.md").write_text(
            "# Zils miner\n\nSee [setup and operation](docs/mining.md). "
            "After setup, run `./start-miner`.\n"
        )
        launcher = directory / "start-miner"
        launcher.write_text(
            '#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nexec "${ZILS_PYTHON:-${FEZ_PYTHON:-.venv-kev/bin/python}}" -m miner --config config.json "$@"\n'
        )
        launcher.chmod(0o700)
        archive = out / f"miner-{uid}.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(directory, arcname=directory.name)
        archive.chmod(0o600)
    wire.write_json(
        validator_dir / "config.json",
        {
            **shared,
            **validator_identity,
            "members": members,
            "benchmark_sha256": digest(data / "manifest.json"),
            "round_timeout": 3600,
            "round_pause": 60,
            "result_grace": 30,
        },
    )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="role", required=True)
    create = commands.add_parser("init")
    create.add_argument("--out", required=True)
    create.add_argument("--benchmark", default=".private/benchmarks/fez-v1-002")
    create.add_argument("--checkpoint", default="models/reference")
    create.add_argument("--host", default="127.0.0.1")
    create.add_argument("--port", type=int, default=8900)
    create.add_argument("--miner-ports", type=int, nargs="+", default=[8901, 8902, 8903])
    create.add_argument(
        "--testnet-identities",
        help="JSON roster of registered UIDs, public hotkeys and wallet paths",
    )
    for role in ("miner", "validator"):
        command = commands.add_parser(role)
        command.add_argument("--config", required=True)
        command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        command.add_argument("--runtime-python", default=sys.executable)
        command.add_argument(
            "--rounds",
            type=int,
            default=0,
            help="0 keeps running; a positive number stops after that many rounds",
        )
        command.add_argument("--poll", type=float, default=5)
        command.add_argument(
            "--no-download", action="store_true", help="use an already-provisioned model cache"
        )
        if role == "validator":
            command.add_argument(
                "--publish-weights",
                action="store_true",
                help="publish configured testnet weights after evaluation",
            )
    args = parser.parse_args(argv)

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        if args.role == "init":
            identities = (
                json.loads(Path(args.testnet_identities).read_text())
                if args.testnet_identities
                else None
            )
            out = initialize(
                args.out,
                args.benchmark,
                args.checkpoint,
                args.host,
                args.port,
                args.miner_ports,
                identities,
            )
            print(
                f"Created validator config and {len(args.miner_ports)} separate miner bundles in {out}"
            )
            return
        if args.rounds < 0 or not math.isfinite(args.poll) or args.poll <= 0:
            raise ValueError("rounds must be nonnegative and poll must be positive")
        path = Path(args.config).resolve()
        directory = path.parent
        config = json.loads(path.read_text())
        if getattr(args, "publish_weights", False) and "chain" not in config:
            raise ValueError("--publish-weights requires a testnet fleet config")
        signing_key(config)
        if "chain" in config:
            from . import testnet

            with testnet.connect(config) as sub:
                testnet.preflight(config, sub)
        wire.endpoint_ok(config["validator"], allowed=[config["validator"]])
        args.runtime_python = str(Path(args.runtime_python).absolute())
        state = directory / "state"
        state.mkdir(mode=0o700, exist_ok=True)
        with locked(state / "service.lock"):
            if not args.no_download:
                prepare_base()
            if args.device == "auto":
                import torch

                args.device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )
            (miner if args.role == "miner" else validator)(config, directory, args)
    except KeyboardInterrupt:
        print("Zils service stopped.", flush=True)
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"zils fleet: {error}\n")


if __name__ == "__main__":
    main()
