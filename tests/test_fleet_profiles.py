"""Fleet profile selection and private image exports; no GPU weights are downloaded."""

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
from zils import fleet, image_jobs, jobs, models, protocol, validator
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
        for model in (models.JEVK5, models.IMAJEV):
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
    assert sys.argv[sys.argv.index('-m')+1]=='zils.jevk5'
    rows=[json.loads(s) for s in Path(sys.argv[sys.argv.index('--data')+1]).read_text().splitlines()]
    assert all(set(r)=={'state','questions'} for r in rows)
    out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir()
    (out/'adapter_config.json').write_text('{}')
    (out/'adapter_model.safetensors').write_text('trained')
    models.write_metadata(out)
else:
    path=Path(sys.argv[sys.argv.index('--checkpoint')+1])
    p=.9 if (path/'adapter_model.safetensors').read_text()=='trained' else .5
    temp=models.temperature(path);a,b=p**(1/temp),(1-p)**(1/temp)
    rows=json.load(sys.stdin)
    assert all(set(r)=={'id','state','question'} for r in rows)
    print(json.dumps({'runtime':{'temperature':temp},'predictions':[
      {'id':r['id'],'probabilities':{'true':a/(a+b),'false':b/(a+b)},'elapsed_ms':1} for r in rows]}))
"""
                runtime.write_text(
                    f"#!{sys.executable}\n"
                    + (IMAGE_SCRIPT if model == models.IMAJEV else text_script)
                )
                runtime.chmod(0o700)
                capacity = root / "nvidia-smi"
                capacity.write_text("#!/bin/sh\necho 65536\n")
                capacity.chmod(0o700)
                env = {
                    **os.environ,
                    "ZILS_IMAGE_RUNTIME_PYTHON": str(runtime),
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
                                    "cuda" if model == models.IMAJEV else "cpu",
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

    def test_text_fleet_uses_frozen_profile_and_rejects_a_different_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            jobs.build(
                data,
                "text",
                examples(),
                POLICY,
                model=models.JEVK5,
                allow_training_data_export=True,
            )
            reference = checkpoint(root / "reference", models.JEVK5)
            out = fleet.initialize(root / "fleet", data, reference, "127.0.0.1", 8900, [8901])
            cfg = json.loads((out / "miner-1/config.json").read_text())
            self.assertEqual(cfg["base_revision"], "c4f7fdb3aeab5582336406e78d3bef11bf98833d")
            self.assertEqual(cfg["model"]["id"], "jevk5-4b-v0.3")
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
        for model in (models.JEVK5, models.IMAJEV):
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
                    if model == models.JEVK5:
                        models.write_metadata(source, model=model, kind="base")
                        claim["sha256"] = zils.checkpoint_hash(source)
                        with self.assertRaisesRegex(ValueError, "profile|adapter"):
                            protocol.fetch_checkpoint(claim, root / "bad", model=model)


if __name__ == "__main__":
    unittest.main()
