"""Fleet profile selection and private image exports; no GPU weights are downloaded."""

import contextlib
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import zils
from tests.fixture_images import (
    POLICY as IMAGE_POLICY,
    SCRIPT as IMAGE_SCRIPT,
    fixture as image_fixture,
)
from tests.test_jobs import POLICY, examples
from zils import benchmark, core, fleet, image_jobs, jobs, models, protocol, validator
from zils.runtime import CapacityUnavailable, run_child


def checkpoint(path, model):
    path.mkdir()
    for name in models.candidate_files(model):
        (path / name).write_text("fixture")
    models.write_metadata(path, model=model)
    return path


def image_dataset(root):
    job, splits, assets = image_fixture()
    images = root / "photos"
    images.mkdir()
    for index, asset in enumerate(assets.values()):
        output = io.BytesIO()
        Image.new("RGB", (8, 8), (index % 2, index, 0)).save(output, format="PNG")
        data = output.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        asset.update(canonical_sha256=sha, pixel_sha256=sha, canonical_bytes=len(data))
        (images / (sha + ".png")).write_bytes(data)
    data = root / "data"
    manifest = image_jobs.build(data, job, splits, assets, IMAGE_POLICY)
    return data, images, manifest["assets"]


class FleetProfilesTest(unittest.TestCase):
    def test_each_vertical_completes_a_signed_training_and_evaluation_round(self):
        for model in (models.H2O, models.IMAJEV, models.D3):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                options = {}
                if model == models.IMAJEV:
                    data, images, _ = image_dataset(root)
                    options = {"images": images, "min_free_mib": 12288, "max_seconds": 120}
                else:
                    data = root / "data"
                    jobs.build(
                        data,
                        "text",
                        examples(),
                        POLICY,
                        model=model,
                        allow_training_data_export=True,
                    )
                reference = checkpoint(root / "reference", model)
                with socket.socket() as a, socket.socket() as b:
                    a.bind(("127.0.0.1", 0))
                    b.bind(("127.0.0.1", 0))
                    ports = [a.getsockname()[1], b.getsockname()[1]]
                with patch("zils.imajev.verify_starting_checkpoint"):
                    out = fleet.initialize(
                        root / "fleet", data, reference, "127.0.0.1", ports[0], ports[1:], **options
                    )
                config = out / "validator/config.json"
                cfg = json.loads(config.read_text())
                cfg.update(round_timeout=90, result_grace=1)
                config.write_text(json.dumps(cfg))
                runtime = root / "fixture-runtime"
                text_script = """import json, sys
from pathlib import Path
from zils import models
if '-m' in sys.argv:
    assert sys.argv[sys.argv.index('-m')+1]==TRAINER
    rows=[json.loads(s) for s in Path(sys.argv[sys.argv.index('--data')+1]).read_text().splitlines()]
    assert all(set(r)=={'state','questions'} for r in rows)
    out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir()
    (out/'adapter_config.json').write_text('{}')
    (out/'adapter_model.safetensors').write_text('trained')
    models.write_metadata(out, model=MODEL)
else:
    path=Path(sys.argv[sys.argv.index('--checkpoint')+1])
    p=.9 if (path/'adapter_model.safetensors').read_text()=='trained' else .5
    temp=models.temperature(path);a,b=p**(1/temp),(1-p)**(1/temp)
    rows=json.load(sys.stdin)
    assert all(set(r)=={'id','state','question'} for r in rows)
    print(json.dumps({'runtime':{'temperature':temp},'predictions':[
      {'id':r['id'],'probabilities':{'true':a/(a+b),'false':b/(a+b)},'elapsed_ms':1} for r in rows]}))
"""
                trainer = "zils" + models.TEXT_RUNTIME_MODULES.get(model, ".h2o")
                runtime.write_text(
                    f"#!{sys.executable}\n"
                    + (
                        IMAGE_SCRIPT
                        if model == models.IMAJEV
                        else f"TRAINER={trainer!r}\nMODEL={model!r}\n" + text_script
                    )
                )
                runtime.chmod(0o700)
                capacity = root / "nvidia-smi"
                capacity.write_text("#!/bin/sh\necho 65536\n")
                capacity.chmod(0o700)
                env = {
                    **os.environ,
                    "ZILS_IMAGE_RUNTIME_PYTHON": str(runtime),
                    "ZILS_H2O_RUNTIME_PYTHON": str(runtime),
                    "ZILS_D3_RUNTIME_PYTHON": str(runtime),
                    "ZILS_NVIDIA_SMI": str(capacity),
                    "ZILS_COMPUTE_LOCK": str(root / "gpu.lock"),
                    "ZILS_PYTHON": sys.executable,
                }
                processes, logs = [], []
                try:
                    for role, path in (
                        ("validator", config),
                        ("miner", out / "miner-1/config.json"),
                    ):
                        log = (root / (role + ".log")).open("w")
                        logs.append(log)
                        command = (
                            [str(out / "miner-1/start-miner")]
                            if role == "miner"
                            else [sys.executable, "-m", "zils.fleet", role, "--config", str(path)]
                        )
                        processes.append(
                            subprocess.Popen(
                                [
                                    *command,
                                    "--device",
                                    "cuda",
                                    "--runtime-python",
                                    str(runtime),
                                    "--no-download",
                                    "--rounds",
                                    "1",
                                    "--poll",
                                    ".05",
                                ],
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                env=env,
                            )
                        )
                    for process in processes:
                        process.wait(timeout=120)
                    for role, process in zip(("validator", "miner"), processes, strict=True):
                        self.assertEqual(
                            process.returncode, 0, (root / (role + ".log")).read_text()
                        )
                    reports = list((out / "validator/state/rounds").glob("*/report.json"))
                    self.assertEqual(len(reports), 1)
                    report = json.loads(reports[0].read_text())
                    self.assertEqual(report["model"]["id"], model)
                    self.assertEqual(report["delivery"]["status"], "accepted", report)
                    self.assertEqual(report["weights"], {"1": 1.0})
                    result = next((out / "miner-1/state/jobs").glob("*/result.json"))
                    self.assertEqual(json.loads(result.read_text())["model"]["id"], model)
                    if model == models.IMAJEV:
                        entry = json.loads(
                            next((out / "miner-1/state/jobs").glob("*/candidate.json")).read_text()
                        )
                        registry = {
                            1: {
                                "claim": {
                                    "uid": 1,
                                    "sha256": entry["sha256"],
                                    "endpoint": "fixture",
                                }
                            }
                        }

                        def fetch(claim, destination, **kwargs):
                            zils.stage(zils.submission(entry["checkpoint"], 1), destination)

                        waiting = True

                        def initially_busy(*args, **kwargs):
                            nonlocal waiting
                            if waiting:
                                waiting = False
                                raise CapacityUnavailable()
                            return run_child(*args, **kwargs)

                        work = root / "capacity-retry"
                        work.mkdir()
                        with (
                            patch.dict(os.environ, env),
                            patch("zils.validator.run_child", side_effect=initially_busy),
                        ):
                            retry = validator.evaluate_round(
                                cfg,
                                out / "validator",
                                work,
                                registry,
                                SimpleNamespace(
                                    device="cuda", runtime_python=str(runtime), poll=0.05
                                ),
                                fetch_checkpoint=fetch,
                                wait_for_capacity=True,
                            )
                        self.assertEqual(retry["delivery"]["status"], "accepted")
                        self.assertEqual(len(list(work.glob("evaluation-*"))), 1)
                        waiting = True
                        deferred = root / "queue-defers"
                        deferred.mkdir()
                        with (
                            patch.dict(os.environ, env),
                            patch("zils.validator.run_child", side_effect=initially_busy),
                        ):
                            with self.assertRaises(CapacityUnavailable):
                                validator.evaluate_round(
                                    cfg,
                                    out / "validator",
                                    deferred,
                                    registry,
                                    SimpleNamespace(device="cuda", runtime_python=str(runtime)),
                                    fetch_checkpoint=fetch,
                                )
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=10)
                    for log in logs:
                        log.close()

    def test_d3_runtime_receives_image_paths_with_each_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, images, _ = image_dataset(root)
            frozen = benchmark.audit(data)
            # image_jobs freezes Imajev datasets today; the d3 run binds the same split to its profile.
            manifest = json.loads((data / "manifest.json").read_text())
            (data / "manifest.json").write_text(
                json.dumps({**manifest, "model": models.spec(models.D3)})
            )
            runtime = root / "fixture-runtime"
            runtime.write_text(
                f"#!{sys.executable}\n"
                + """import json, sys
from pathlib import Path
assert sys.argv[sys.argv.index('--base-revision')+1]
rows=json.load(sys.stdin)
assert all(set(r)=={'id','state','question','images'} for r in rows)
for r in rows:
    assert all(Path(p).is_absolute() and Path(p).read_bytes()[:8]==b'\\x89PNG\\r\\n\\x1a\\n' for p in r['images'])
print(json.dumps({'runtime':{},'predictions':[
  {'id':r['id'],'probabilities':{k:.5 for k in r['question']['criteria']},
   'elapsed_ms':len(r['images'])} for r in rows]}))
"""
            )
            runtime.chmod(0o700)
            (root / "submissions.json").write_text(
                json.dumps([core.submission(checkpoint(root / "candidate", models.D3), 1)])
            )
            args = SimpleNamespace(
                cases=data / "test.jsonl",
                submissions=root / "submissions.json",
                base_revision=models.spec(models.D3)["base_revision"],
                timeout=30,
                report=root / "report.json",
                device="cuda",
                runner_python=sys.executable,
                images=images,
            )
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
                patch.dict(os.environ, {"ZILS_D3_RUNTIME_PYTHON": str(runtime)}),
                patch("zils.benchmark.audit", return_value=frozen),
            ):
                core.evaluate(args)
            miner = json.loads(args.report.read_text())["miners"][0]
            self.assertEqual(miner["status"], "evaluated", miner.get("error"))
            self.assertEqual(
                [p["elapsed_ms"] for p in miner["predictions"]], [1] * len(frozen["test"])
            )
            asset = manifest["assets"][frozen["test"][0]["image"]["asset_id"]]
            cached = images / (asset["canonical_sha256"] + ".png")
            cached.write_bytes(cached.read_bytes() + b"tampered")
            with (
                patch("zils.benchmark.audit", return_value=frozen),
                self.assertRaisesRegex(ValueError, "Frozen image bytes changed"),
            ):
                core.evaluate(SimpleNamespace(**{**vars(args), "report": root / "tampered"}))
            text = {
                "base_revision": models.spec(models.H2O)["base_revision"],
                "report": root / "h2o",
            }
            with self.assertRaisesRegex(ValueError, "Image data cannot use a text runtime"):
                core.evaluate(SimpleNamespace(**{**vars(args), **text}))

    def test_text_fleet_uses_frozen_profile_and_rejects_a_different_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            jobs.build(
                data,
                "text",
                examples(),
                POLICY,
                model=models.H2O,
                allow_training_data_export=True,
            )
            reference = checkpoint(root / "reference", models.H2O)
            out = fleet.initialize(root / "fleet", data, reference, "127.0.0.1", 8900, [8901])
            cfg = json.loads((out / "miner-1/config.json").read_text())
            self.assertEqual(cfg["base_revision"], "acaf0d4ea251e54de928c75ef4352670d33192d3")
            self.assertEqual(cfg["model"]["id"], "h2o-lightning-4b-v1.2.3")
            wrong = checkpoint(root / "wrong", models.IMAJEV)
            with self.assertRaisesRegex(ValueError, "profile"):
                fleet.initialize(root / "bad", data, wrong, "127.0.0.1", 8900, [8901])
            self.assertFalse((root / "bad").exists())

    def test_image_bundles_export_only_training_photos_and_pin_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, images, assets = image_dataset(root)
            reference = checkpoint(root / "reference", models.IMAJEV)
            with patch("zils.imajev.verify_starting_checkpoint"):
                out = fleet.initialize(
                    root / "fleet",
                    data,
                    reference,
                    "127.0.0.1",
                    8900,
                    [8901],
                    images=images,
                    min_free_mib=14000,
                    max_seconds=900,
                )
            cfg = json.loads((out / "miner-1/config.json").read_text())
            self.assertEqual(cfg["model"]["id"], "imajev-4b-v1")
            self.assertEqual(cfg["min_free_mib"], 14000)
            self.assertEqual(cfg["max_seconds"], 900)
            from miner.worker import train_candidate

            for field, value in (("min_free_mib", 12288), ("max_seconds", 3600)):
                job = {**cfg, "round_id": "a" * 32, field: value}
                with (
                    patch.dict(os.environ, {"ZILS_IMAGE_RUNTIME_PYTHON": sys.executable}),
                    patch(
                        "miner.worker.run_child",
                        side_effect=AssertionError("must not start training"),
                    ),
                    self.assertRaisesRegex(ValueError, "configuration"),
                ):
                    train_candidate(cfg, out / "miner-1", job, sys.executable, "cuda")
            with tarfile.open(out / "miner-1.tar.gz") as archive:
                names = set(archive.getnames())
            for asset in assets.values():
                filename = asset["canonical_sha256"] + ".png"
                self.assertEqual("miner-1/images/" + filename in names, asset["split"] == "train")
                self.assertEqual(
                    (out / "validator/images" / filename).exists(), asset["split"] != "train"
                )
            self.assertFalse(
                any(
                    Path(n).name in {"test.jsonl", "calibration.jsonl", "manifest.json"}
                    for n in names
                    if "/zils/" not in n
                )
            )

    def test_image_export_rejects_tampering_before_creating_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, images, assets = image_dataset(root)
            reference = checkpoint(root / "reference", models.IMAJEV)
            asset = next(iter(assets.values()))
            (images / (asset["canonical_sha256"] + ".png")).write_bytes(b"changed")
            with (
                patch("zils.imajev.verify_starting_checkpoint"),
                self.assertRaisesRegex(ValueError, "image"),
            ):
                fleet.initialize(
                    root / "fleet",
                    data,
                    reference,
                    "127.0.0.1",
                    8900,
                    [8901],
                    images=images,
                    min_free_mib=14000,
                )
            self.assertFalse((root / "fleet").exists())

    def test_transfer_uses_expected_profile_and_rejects_cross_profile_metadata(self):
        for model in (models.H2O, models.D3, models.IMAJEV):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = checkpoint(root / "source", model)
                requested = []

                class Files(protocol.Handler):
                    def do_GET(self):
                        name = self.path.rsplit("/", 1)[-1]
                        requested.append(name)
                        path = source / name
                        if not path.exists():
                            self.reply(404, {})
                            return
                        data = path.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)

                with protocol.local_server(Files) as endpoint:
                    claim = {"endpoint": endpoint, "sha256": zils.checkpoint_hash(source)}
                    protocol.fetch_checkpoint(claim, root / "download", model=model)
                    self.assertEqual(set(requested), set(models.candidate_files(model)))
                    self.assertEqual(zils.checkpoint_hash(root / "download"), claim["sha256"])
                    # A signed base-only hash must not bypass the adapter-file contract.
                    if model == models.H2O:
                        models.write_metadata(source, model=model, kind="base")
                        claim["sha256"] = zils.checkpoint_hash(source)
                        with self.assertRaisesRegex(ValueError, "profile|adapter"):
                            protocol.fetch_checkpoint(claim, root / "bad", model=model)

    def test_queue_job_carries_capacity_limits_for_pinned_profiles(self):
        import uuid

        from miner import queue

        class Stop(Exception):
            pass

        for model in (models.H2O, models.D3):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                reference = checkpoint(root / "reference", model)
                training = b"{}\n"
                job_id = str(uuid.uuid4())
                assignment = {
                    "job_id": job_id,
                    "lease_token": str(uuid.uuid4()),
                    "uid": 1,
                    "model": models.spec(model),
                    "base_revision": models.spec(model)["base_revision"],
                    "initial_sha256": zils.checkpoint_hash(reference),
                    "training_sha256": hashlib.sha256(training).hexdigest(),
                    "job_sha256": "c" * 64,
                    "training": {"url": "https://example.test/training"},
                    "min_free_mib": 14000,
                    "max_seconds": 900,
                }

                class Client:
                    url = "https://example.test"
                    key = SimpleNamespace(ss58_address="hotkey")

                    def call(self, name, payload=None):
                        return {"assignment": assignment} if name == "claim" else {}

                jobs_seen = []

                def train(config, directory, job, runtime, device, **options):
                    jobs_seen.append(job)
                    raise Stop()

                with (
                    patch("miner.queue.gpu_ready", return_value=True),
                    patch(
                        "miner.queue.download", lambda url, path, limit: path.write_bytes(training)
                    ),
                    patch("miner.queue.train_candidate", train),
                    patch("miner.queue.lease_heartbeat") as heartbeat,
                    self.assertRaises(Stop),
                ):
                    heartbeat.return_value.__enter__.return_value = lambda: None
                    queue.run_once(Client(), root / "state", reference, sys.executable, "cuda")
                self.assertEqual(jobs_seen[0]["min_free_mib"], 14000)
                self.assertEqual(jobs_seen[0]["max_seconds"], 900)

    def test_queue_miner_refuses_to_start_with_a_reference_it_cannot_run(self):
        from miner import queue

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            seed = "0x" + "11" * 32
            config.write_text(json.dumps({"coordinator": "https://example.test", "seed": seed}))
            reference = checkpoint(root / "d3-reference", models.D3)
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("ZILS_D3_RUNTIME_PYTHON", "FEZ_D3_RUNTIME_PYTHON")
            }
            cases = (
                (reference, "ZILS_D3_RUNTIME_PYTHON"),
                (root / "missing-reference", "missing-reference"),
            )
            for path, message in cases:
                argv = ["queue", "--config", str(config), "--state", str(root / "state")]
                argv += ["--reference", str(path), "--no-download", "--once"]
                with (
                    self.subTest(reference=path.name),
                    patch.dict(os.environ, env, clear=True),
                    patch.object(sys, "argv", argv),
                    patch("miner.queue.run_once") as run_once,
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                    self.assertRaises(SystemExit) as stop,
                ):
                    queue.main()
                self.assertEqual(stop.exception.code, 1)
                self.assertIn(message, stderr.getvalue())
                run_once.assert_not_called()


if __name__ == "__main__":
    unittest.main()
