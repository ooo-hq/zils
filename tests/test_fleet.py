"""Three real miner services and a validator; only expensive ML work uses a fixture executable."""

import importlib.util
import json
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import zils
from zils import benchmark


@unittest.skipUnless(
    importlib.util.find_spec("bittensor_wallet") and importlib.util.find_spec("kev"),
    "use .venv-kev",
)
class FleetTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("zils.fleet"), "persistent miner service is missing"
        )
        from zils import fleet

        return fleet

    def test_pinned_network_and_authenticated_validator(self):
        from bittensor_wallet import Keypair

        from zils import protocol as r, runtime

        key = Keypair.create_from_seed("0x" + "01" * 32)
        claim = {
            "round_id": "a" * 32,
            "uid": 1,
            "hotkey": key.ss58_address,
            "sha256": "b" * 64,
            "endpoint": "http://192.168.1.20:8901",
        }
        signed = {"claim": claim, "signature": key.sign(r.canonical(claim)).hex()}
        with self.assertRaises(ValueError):
            r.register(signed, "a" * 32, {1: key.ss58_address}, {})
        self.assertEqual(
            r.register(
                signed, "a" * 32, {1: key.ss58_address}, {}, endpoints={1: claim["endpoint"]}
            ),
            claim,
        )
        with self.assertRaises(ValueError):
            r.register(
                signed,
                "a" * 32,
                {1: key.ss58_address},
                {},
                endpoints={1: "http://192.168.1.21:8901"},
            )
        for url in (
            "http://169.254.169.254:80",
            "http://8.8.8.8:80",
            "http://0.0.0.0:80",
            "http://example.com:80",
        ):
            with self.assertRaises(ValueError):
                r.endpoint_ok(url, allowed=[url])
        payload = {"kind": "round", "round_id": "a" * 32, "status": "collecting"}
        message = runtime.signed(payload, key)
        self.assertEqual(runtime.verified(message, key.ss58_address), payload)
        message["payload"]["round_id"] = "b" * 32
        with self.assertRaises(ValueError):
            runtime.verified(message, key.ss58_address)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "state.json"
            with (
                patch("zils.protocol.os.fsync", side_effect=OSError("interrupted write")),
                self.assertRaises(OSError),
            ):
                r.write_json(destination, {"complete": True})
            self.assertFalse(destination.exists(), "incomplete state must never become visible")
            r.write_json(destination, {"complete": True})
            with self.assertRaises(FileExistsError):
                r.write_json(destination, {"complete": False})
            self.assertEqual(json.loads(destination.read_text()), {"complete": True})

    def test_three_miners_complete_two_rounds_and_keep_private_data_local(self):
        from miner.worker import train_candidate

        f = self.module()
        from zils import runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "benchmark"
            benchmark.build(data, seed=553)
            source = root / "source"
            source.mkdir()
            for name in zils.ARTIFACT_FILES:
                (source / name).write_bytes(b"initial fixture")
            sockets = []
            for _ in range(4):
                sock = socket.socket()
                sock.bind(("127.0.0.1", 0))
                sockets.append(sock)
            ports = [s.getsockname()[1] for s in sockets]
            for sock in sockets:
                sock.close()
            package = root / "fleet"
            f.initialize(package, data, source, "127.0.0.1", ports[0], ports[1:])
            for uid in (1, 2, 3):
                with tarfile.open(package / f"miner-{uid}.tar.gz") as archive:
                    names = archive.getnames()
                self.assertFalse(
                    any(
                        Path(n).name in {"calibration.jsonl", "test.jsonl"}
                        or "validator" in Path(n).parts
                        for n in names
                    )
                )
                self.assertTrue(any(n.endswith("miner-training.jsonl") for n in names))
                self.assertEqual(
                    (package / f"miner-{uid}/config.json").stat().st_mode & 0o777, 0o600
                )
            validator_path = package / "validator/config.json"
            config = json.loads(validator_path.read_text())
            config.update(round_timeout=45, round_pause=0.1, result_grace=8)
            validator_path.write_text(json.dumps(config))
            worker = root / "fixture-python"
            worker.write_text(
                f"#!{sys.executable}\n"
                + """import json, sys
from pathlib import Path
import torch
from kev.checkpoint import Meta, read_meta, write_meta
if '-m' in sys.argv:
    assert sys.argv[sys.argv.index('-m')+1]=='kev.train'
    out=Path(sys.argv[sys.argv.index('--out')+1]); out.mkdir()
    seed=int(sys.argv[sys.argv.index('--seed')+1])
    data=Path(sys.argv[sys.argv.index('--data')+1])
    assert all(set(json.loads(line))=={'state','questions'} for line in data.read_text().splitlines())
    (out/'adapter_config.json').write_text('{}')
    (out/'adapter_model.safetensors').write_text(str(seed))
    write_meta(out, Meta(base='Qwen/Qwen3.5-0.8B-Base', head={'fixture':torch.tensor([seed])}, temperature=1.0))
    with (data.parent/'training-calls.jsonl').open('a') as log: log.write(json.dumps({'seed':seed})+'\\n')
else:
    meta=read_meta(sys.argv[sys.argv.index('--checkpoint')+1])
    requests=json.load(sys.stdin)
    assert all(set(row)=={'id','state','question'} for row in requests)
    import zils
    predictions=[{'id':row['id'],'elapsed_ms':1,'probabilities':dict.fromkeys(zils.options(row['question']),1/len(zils.options(row['question'])))} for row in requests]
    print(json.dumps({'predictions':predictions,'runtime':{'fixture':True,'temperature':meta.temperature}}))
"""
            )
            worker.chmod(0o700)
            processes, logs = [], []
            try:
                for role, directory in [
                    ("validator", package / "validator"),
                    *[("miner", package / f"miner-{i}") for i in (1, 2, 3)],
                ]:
                    log = (directory / "test.log").open("w")
                    logs.append(log)
                    command = (
                        [str(directory / "start-miner")]
                        if role == "miner"
                        else [
                            sys.executable,
                            "-m",
                            "zils.fleet",
                            role,
                            "--config",
                            str(directory / "config.json"),
                        ]
                    )
                    processes.append(
                        subprocess.Popen(
                            [
                                *command,
                                "--runtime-python",
                                str(worker),
                                "--device",
                                "cpu",
                                "--no-download",
                                "--rounds",
                                "2",
                                "--poll",
                                ".1",
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env={**os.environ, "ZILS_PYTHON": sys.executable},
                        )
                    )
                    if role == "validator":
                        deadline = time.monotonic() + 10
                        while True:
                            try:
                                if runtime.request(config, "/round").get("kind") == "round":
                                    break
                            except URLError:
                                pass
                            if time.monotonic() > deadline:
                                self.fail("validator did not start")
                            time.sleep(0.05)
                        for message in ([], {"claim": []}):
                            with self.assertRaises(HTTPError) as error:
                                runtime.request(config, "/submit", message)
                            self.assertEqual(error.exception.code, 400)
                        with self.assertRaises(HTTPError) as error:
                            runtime.request(
                                config, "/submit", {"claim": {"uid": 1, "round_id": "0" * 32}}
                            )
                        self.assertEqual(
                            error.exception.code,
                            409,
                            "a late miner must be able to retry the next round",
                        )
                    elif directory.name == "miner-1":
                        # Restart the actual service after training, while the other miners are still offline.
                        deadline = time.monotonic() + 30
                        while not list((directory / "state/jobs").glob("*/candidate.json")):
                            if time.monotonic() > deadline or processes[-1].poll() is not None:
                                self.fail("miner did not train: " + Path(log.name).read_text())
                            time.sleep(0.1)
                        previous = processes[-1]
                        previous.terminate()
                        previous.wait(timeout=8)
                        self.assertEqual(previous.returncode, 0)
                        processes[-1] = subprocess.Popen(
                            previous.args,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env={**os.environ, "FEZ_PYTHON": sys.executable},
                        )
                # Two rounds repeatedly start CPU PyTorch subprocesses. Allow shared
                # CI runners time to finish, with one deadline for the whole fleet.
                deadline = time.monotonic() + 300
                for process in processes:
                    try:
                        process.wait(timeout=max(0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        details = "\n".join(
                            f"{Path(log.name).parent.name}:\n{Path(log.name).read_text()}"
                            for log in logs
                        )
                        self.fail("fleet did not finish within five minutes:\n" + details)
                for process, log in zip(processes, logs):
                    log.flush()
                    self.assertEqual(process.returncode, 0, Path(log.name).read_text())
                reports = list((package / "validator/state/rounds").glob("*/report.json"))
                self.assertEqual(len(reports), 2)
                for path in reports:
                    report = json.loads(path.read_text())
                    self.assertFalse(report["chain_write"])
                    self.assertEqual(len(report["miners"]), 3)
                    self.assertTrue(all(m["status"] == "evaluated" for m in report["miners"]))
                for uid in (1, 2, 3):
                    miner = package / f"miner-{uid}"
                    calls = benchmark.read_jsonl(miner / "training-calls.jsonl")
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(len({c["seed"] for c in calls}), 2)
                    self.assertEqual(len(list((miner / "state/jobs").glob("*/result.json"))), 2)
                    # Restarting an already trained round must return the frozen submission without retraining.
                    config = json.loads((miner / "config.json").read_text())
                    job_path = next((miner / "state/jobs").glob("*/job.json"))
                    job = json.loads(job_path.read_text())
                    first = train_candidate(config, miner, job, str(worker), "cpu")
                    second = train_candidate(config, miner, job, str(worker), "cpu")
                    self.assertEqual(first, second)
                    self.assertEqual(len(benchmark.read_jsonl(miner / "training-calls.jsonl")), 2)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                for log in logs:
                    log.close()


if __name__ == "__main__":
    unittest.main()
