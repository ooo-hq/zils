"""Train one candidate per round and submit its signed checkpoint."""

import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

import zils
from zils import models, protocol as wire
from zils.runtime import digest, request, run_child, signed, signing_key


def train_candidate(config, directory, job, runtime, device, *, check_lease=None):
    directory = Path(directory)
    for name in ("base_revision", "initial_sha256", "training_sha256"):
        if job.get(name) != config[name]:
            raise ValueError("validator round differs from the installed training configuration")
    for name in ("job_id", "job_sha256"):
        if job.get(name) != config.get(name):
            raise ValueError("validator round differs from the installed customer job")
    if not re.fullmatch("[a-f0-9]{32}", job.get("round_id", "")):
        raise ValueError("invalid round id")
    if digest(directory / "miner-training.jsonl") != config["training_sha256"]:
        raise ValueError("training data changed")
    if zils.checkpoint_hash(directory / "reference") != config["initial_sha256"]:
        raise ValueError("initial checkpoint changed")
    work = directory / "state/jobs" / job["round_id"]
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    candidate = work / "candidate.json"
    if candidate.exists():
        entry = json.loads(candidate.read_text())
        if zils.checkpoint_hash(entry["checkpoint"]) != entry["sha256"]:
            raise ValueError("saved submission was modified")
        return entry
    if shutil.disk_usage(work).free < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB free; archive completed rounds before training again")
    if not (work / "job.json").exists():
        wire.write_json(work / "job.json", job)
    identity = config.get("seed") or config["hotkey"]
    authority = config.get("training_authority") or config["validator_hotkey"]
    seed = (
        int(
            hashlib.sha256((authority + identity + job["round_id"]).encode()).hexdigest()[:8],
            16,
        )
        % 2**31
    )
    raw = work / ("training-" + uuid.uuid4().hex)
    command = [
        runtime,
        "-u",
        "-m",
        "kev.train",
        "--data",
        str(directory / "miner-training.jsonl"),
        "--base",
        zils.BASE,
        "--base_revision",
        config["base_revision"],
        "--init_from",
        str(directory / "reference"),
        "--epochs",
        "1",
        "--lr",
        "2e-5",
        "--batch",
        "1",
        "--accum",
        "4",
        "--dtype",
        "fp32",
        "--device",
        device,
        "--p_none",
        "0",
        "--p_none_distract",
        "0",
        "--p_distract",
        "0",
        "--seed",
        str(seed),
        "--out",
        str(raw),
    ]
    model = models.checkpoint_model(directory / "reference")
    if model == models.JEVK5:
        if config["base_revision"] != models.spec(models.JEVK5)["base_revision"]:
            raise ValueError("training model revision differs from its reference")
        command = [
            runtime,
            "-u",
            "-m",
            "zils.jevk5",
            "train",
            "--data",
            str(directory / "miner-training.jsonl"),
            "--reference",
            str(directory / "reference"),
            "--device",
            device,
            "--seed",
            str(seed),
            "--out",
            str(raw),
        ]
    if model == models.IMAJEV:
        command = [
            runtime,
            "-u",
            "-m",
            "zils.imajev_runner",
            "--train",
            "--cases",
            str(directory / "miner-training.jsonl"),
            "--checkpoint",
            str(directory / "reference"),
            "--images",
            str(directory / "images"),
            "--device",
            device,
            "--seed",
            str(seed),
            "--out",
            str(raw),
        ]
    print(
        f"miner {config['uid']}: training round {job['round_id']} on {device}; log {work / 'training.log'}",
        flush=True,
    )
    if model == models.IMAJEV:
        run_child(
            command,
            work / "training.log",
            device,
            model=model,
            minimum_mib=job["min_free_mib"],
            timeout=job["max_seconds"],
            check_lease=check_lease,
        )
    else:
        run_child(
            command,
            work / "training.log",
            device,
            model=model,
            minimum_mib=job.get("min_free_mib", 0),
            check_lease=check_lease,
        )
    frozen = work / ("artifacts-" + uuid.uuid4().hex)
    zils.stage(zils.submission(raw, config["uid"]), frozen)
    entry = zils.submission(frozen, config["uid"])
    wire.write_json(candidate, entry)
    return entry


def miner(config, directory, args):
    key = signing_key(config)
    address = urlsplit(config["validator"])
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((address.hostname, address.port))
        host = route.getsockname()[0]
    endpoint = f"http://{host}:{config['port']}"
    wire.endpoint_ok(endpoint, allowed=[endpoint])

    class Artifacts(wire.Handler):
        def do_GET(self):
            match = re.fullmatch(r"/artifacts/([a-f0-9]{32})/([^/]+)", self.path)
            if not match or match[2] not in zils.ARTIFACT_FILES:
                self.reply(404, {"error": "unknown artifact"})
                return
            try:
                entry = json.loads(
                    (directory / "state/jobs" / match[1] / "candidate.json").read_text()
                )
                path = Path(entry["checkpoint"]) / match[2]
                with path.open("rb") as source:
                    self.send_response(200)
                    self.send_header("Content-Length", str(os.fstat(source.fileno()).st_size))
                    self.end_headers()
                    shutil.copyfileobj(source, self.wfile)
            except FileNotFoundError:
                self.reply(404, {"error": "checkpoint not ready"})

    finished, last_error = set(), None
    with wire.local_server(Artifacts, host, config["port"]):
        print(f"miner {config['uid']}: serving {endpoint}; waiting for validator", flush=True)
        while not args.rounds or len(finished) < args.rounds:
            try:
                job = request(config, "/round")
                if job.get("kind") != "round" or job.get("round_id") in finished:
                    time.sleep(args.poll)
                    continue
                rid = job["round_id"]
                if not re.fullmatch("[a-f0-9]{32}", rid):
                    raise ValueError("invalid validator round")
                work = directory / "state/jobs" / rid
                if job["status"] == "collecting":
                    if "chain" in config:
                        from zils import testnet

                        if job.get("chain") != config["chain"]:
                            raise ValueError("validator round is not for this testnet")
                        if not (work / "candidate.json").exists():
                            with testnet.connect(config) as sub:
                                testnet.preflight(config, sub)
                    entry = train_candidate(
                        config, directory, job, args.runtime_python, args.device
                    )
                    claim = {
                        "round_id": rid,
                        "uid": config["uid"],
                        "hotkey": key.ss58_address,
                        "sha256": entry["sha256"],
                        "endpoint": endpoint,
                    }
                    if "job_sha256" in config:
                        claim["job_sha256"] = config["job_sha256"]
                    reply = request(
                        config,
                        "/submit",
                        {"claim": claim, "signature": key.sign(wire.canonical(claim)).hex()},
                    )
                    if reply.get("status") != "accepted" or reply.get("round_id") != rid:
                        raise ValueError("validator did not acknowledge this round's submission")
                result = request(config, "/results/" + rid)
                if result.get("kind") == "result" and result.get("round_id") == rid:
                    work.mkdir(mode=0o700, parents=True, exist_ok=True)
                    if not (work / "result.json").exists():
                        wire.write_json(work / "result.json", result)
                    ack = {"kind": "ack", "uid": config["uid"], "round_id": rid}
                    request(config, "/ack", signed(ack, key))
                    finished.add(rid)
                    print(
                        f"miner {config['uid']}: round complete; proposed weight {result['weights'].get(str(config['uid']), 0):.4f}",
                        flush=True,
                    )
                last_error = None
            except HTTPError as error:
                if error.code not in (404, 409, 503):
                    raise RuntimeError(
                        f"validator rejected request (HTTP {error.code}); check config and validator log"
                    ) from error
            except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
                message = str(error)
                if message != last_error:
                    print(
                        f"miner {config['uid']}: validator unavailable; retrying: {message}",
                        flush=True,
                    )
                    last_error = message
            time.sleep(args.poll)
