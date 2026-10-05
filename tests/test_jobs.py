"""Customer data boundaries and delivery gating; fixture inference is not model quality."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import zils
from zils import benchmark, jobs


def examples():
    return {
        split: [
            {
                "id": f"{split}-{i}",
                "group_id": f"{split}-{i}",
                "family": "routing",
                "state": {"ticket": f"{split}-{i}"},
                "question": {"type": "noul"},
                "label": "true",
            }
            for i in range(2)
        ]
        for split in ("train", "calibration", "test")
    }


POLICY = {"min_accuracy": 0.8, "min_brier_improvement": 0.01}


class JobTest(unittest.TestCase):
    def test_dataset_boundaries_and_export_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "job"
            with self.assertRaisesRegex(ValueError, "permission"):
                jobs.build(root, "support-v1", examples(), POLICY)
            self.assertFalse(root.exists())
            jobs.build(root, "support-v1", examples(), POLICY, allow_training_data_export=True)
            self.assertEqual(benchmark.audit(root), examples())
            exported = benchmark.read_jsonl(root / "miner-training.jsonl")
            self.assertTrue(all(set(c) == {"state", "questions"} for c in exported))
            (root / "test.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                benchmark.audit(root)
        for field in ("id", "group_id", "state"):
            splits = examples()
            splits["test"][0][field] = splits["train"][0][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "overlaps"):
                jobs.validate_splits(splits)

    def test_delivery_requires_improvement_and_customer_threshold(self):
        baseline = {"status": "evaluated", "brier": 0.3}
        candidate = {
            "uid": 1,
            "sha256": "a" * 64,
            "status": "evaluated",
            "brier": 0.2,
            "accuracy": 0.9,
            "skill": 0.6,
        }
        self.assertEqual(jobs.select(baseline, [candidate], POLICY)["status"], "accepted")
        for changes in ({"brier": 0.3}, {"accuracy": 0.7}, {"skill": 0}, {"status": "rejected"}):
            with self.subTest(changes=changes):
                self.assertEqual(
                    jobs.select(baseline, [{**candidate, **changes}], POLICY)["status"],
                    "no_qualifying_model",
                )
        with self.assertRaises(ValueError):
            jobs.select({"status": "rejected"}, [candidate], POLICY)
        with self.assertRaises(ValueError):
            jobs.validate_policy({**POLICY, "min_accuracy": float("nan")})

    @unittest.skipUnless(
        importlib.util.find_spec("kev") and importlib.util.find_spec("bittensor_wallet"),
        "install model and signing dependencies",
    )
    def test_customer_job_evaluation_and_signed_binding(self):
        import torch
        from bittensor_wallet import Keypair
        from kev.checkpoint import Meta, write_meta

        from miner.worker import train_candidate
        from zils import fleet, protocol, validator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "job"
            jobs.build(data, "support-v1", examples(), POLICY, allow_training_data_export=True)
            source = root / "reference"
            source.mkdir()
            (source / "adapter_config.json").write_text("{}")
            (source / "adapter_model.safetensors").write_text("baseline")
            write_meta(
                source, Meta(base=zils.BASE, head={"fixture": torch.zeros(1)}, temperature=2.0)
            )
            package = fleet.initialize(root / "fleet", data, source, "127.0.0.1", 8900, [8901])
            directory = package / "validator"
            config = json.loads((directory / "config.json").read_text())
            miner_config = json.loads((package / "miner-1/config.json").read_text())
            self.assertEqual(config["job_sha256"], benchmark.file_hash(data / "manifest.json"))
            self.assertFalse((package / "miner-1/manifest.json").exists())
            self.assertFalse((package / "miner-1/test.jsonl").exists())
            job = {
                k: miner_config[k]
                for k in (
                    "base_revision",
                    "initial_sha256",
                    "training_sha256",
                    "job_id",
                    "job_sha256",
                )
            }
            with self.assertRaisesRegex(ValueError, "customer job"):
                train_candidate(
                    miner_config,
                    package / "miner-1",
                    {**job, "job_sha256": "f" * 64},
                    sys.executable,
                    "cpu",
                )
            candidate = root / "candidate"
            zils.stage(zils.submission(source, 1), candidate)
            (candidate / "adapter_model.safetensors").chmod(0o600)
            (candidate / "adapter_model.safetensors").write_text("good")
            (candidate / "head.pt").chmod(0o600)
            write_meta(
                candidate, Meta(base=zils.BASE, head={"fixture": torch.ones(1)}, temperature=1.0)
            )
            worker = root / "fixture-python"
            worker.write_text(
                f"#!{sys.executable}\n"
                + """import json, sys
from pathlib import Path
from kev.checkpoint import read_meta
path = Path(sys.argv[sys.argv.index('--checkpoint') + 1])
meta = read_meta(path)
p = 0.9 if (path / 'adapter_model.safetensors').read_text() == 'good' else 0.5
a, b = p ** (1 / meta.temperature), (1 - p) ** (1 / meta.temperature)
requests = json.load(sys.stdin)
assert all(set(r) == {'id', 'state', 'question'} for r in requests)
print(json.dumps({'runtime': {'temperature': meta.temperature}, 'predictions': [
    {'id': r['id'], 'elapsed_ms': 1, 'probabilities': {'true': a/(a+b), 'false': b/(a+b)}} for r in requests]}))
"""
            )
            worker.chmod(0o700)
            rid = "a" * 32
            key = Keypair.create_from_seed(miner_config["seed"])
            claim = {
                "round_id": rid,
                "uid": 1,
                "hotkey": key.ss58_address,
                "sha256": zils.checkpoint_hash(candidate),
                "endpoint": "http://127.0.0.1:8901",
                "job_sha256": config["job_sha256"],
            }
            message = {"claim": claim, "signature": key.sign(protocol.canonical(claim)).hex()}
            registry = {}
            protocol.register(
                message, rid, {1: key.ss58_address}, registry, job_sha256=config["job_sha256"]
            )
            with self.assertRaisesRegex(ValueError, "customer job"):
                protocol.register(message, rid, {1: key.ss58_address}, {}, job_sha256="f" * 64)
            changed = copy.deepcopy(message)
            changed["claim"]["job_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "signature"):
                protocol.register(changed, rid, {1: key.ss58_address}, {}, job_sha256="f" * 64)

            def download(claim, destination, **kwargs):
                zils.stage(zils.submission(candidate, claim["uid"]), destination)

            args = SimpleNamespace(runtime_python=str(worker), device="cpu")
            work = root / rid
            work.mkdir()
            with patch("zils.protocol.fetch_checkpoint", side_effect=download):
                report = validator.evaluate_round(config, directory, work, registry, args)
            self.assertEqual(report["delivery"]["status"], "accepted")
            summary = benchmark.summarize(data, report, "test")
            self.assertEqual(summary["job_id"], "support-v1")
            self.assertEqual(summary["miners"][0]["overall"]["accuracy"], 1)
            self.assertEqual(report["baseline"]["accuracy"], 0)
            self.assertEqual(report["weights"], {1: 1})
            release = work / report["delivery"]["checkpoint"]
            self.assertEqual(zils.checkpoint_hash(release), report["delivery"]["sha256"])
            self.assertEqual(
                json.loads((release / "release.json").read_text())["job_sha256"],
                config["job_sha256"],
            )
            empty = root / ("b" * 32)
            empty.mkdir()
            report = validator.evaluate_round(config, directory, empty, {}, args)
            self.assertEqual(report["delivery"]["status"], "no_qualifying_model")
            self.assertEqual(report["weights"], {})
