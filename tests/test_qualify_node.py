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
                return SimpleNamespace(
                    returncode=0, stdout="NVIDIA RTX 4000 Ada, 580.95\n", stderr=""
                )
            if "verify_runtime" in text:
                return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
            if " train " in text or "zils.h2o train" in text:
                out = Path(command[command.index("--out") + 1])
                out.mkdir(parents=True)
                (out / "training_metrics.json").write_text(
                    json.dumps(
                        {
                            "examples": 6,
                            "optimizer_steps": 2,
                            "training_seconds": 27.6,
                            "peak_gpu_allocated_bytes": 10233864192,
                            "max_tokens": 2048,
                        }
                    )
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if "reload_predict" in text:
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"ok": True, "max_prompt_tokens": 2031}),
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        return run

    def test_report_binds_identities_and_measurements(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(__file__).resolve().parents[1]
            report = qualify_node.qualify(
                root, "python3.12", Path(tmp) / "q", run=self.fake_run(calls), free_mib=19500
            )
        self.assertEqual(report["model"], models.H2O)
        self.assertEqual(
            report["profile_sha256"], models.profile_identity(models.H2O)["profile_sha256"]
        )
        self.assertTrue(
            report["capacity_ok"] and report["runtime_verified"] and report["reload_predict_ok"]
        )
        self.assertEqual(report["training"]["peak_gpu_mib"], 9759)
        self.assertEqual(report["training"]["max_prompt_tokens"], 2031)
        self.assertTrue(any("verify_runtime" in " ".join(map(str, c)) for c in calls))

    def test_below_floor_stops_before_training(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(__file__).resolve().parents[1]
            with self.assertRaises(SystemExit):
                qualify_node.qualify(
                    root, "python3.12", Path(tmp) / "q", run=self.fake_run(calls), free_mib=12000
                )
            report = json.loads((Path(tmp) / "q" / "qualification.json").read_text())
        self.assertFalse(report["capacity_ok"])
        self.assertEqual(report["gpu"]["free_mib_at_start"], 12000)
        self.assertFalse(any("train" in " ".join(map(str, c)) for c in calls))
