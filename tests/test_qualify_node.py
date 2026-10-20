"""Qualification records pinned identities, capacity and a 2,048-token training step."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts import qualify_node
from zils import models


class QualifyNodeTest(unittest.TestCase):
    SIZING = {"overhead": 300, "per_repeat": 2.5}

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
            if "sizing probe" in text:
                return SimpleNamespace(returncode=0, stdout=json.dumps(self.SIZING), stderr="")
            if "zils.d3 parity" in text:
                cases = Path(command[command.index("--cases") + 1]).read_text().splitlines()
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"cases": len(cases), "max_difference": 2.5e-8}),
                    stderr="",
                )
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
        self.assertTrue(all("--id=0" in c for c in calls if c[0] == "nvidia-smi"))

    def test_report_carries_the_accepted_terms_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(__file__).resolve().parents[1]
            report = qualify_node.qualify(
                root,
                "python3.12",
                Path(tmp) / "q",
                run=self.fake_run([]),
                free_mib=19500,
                terms_sha256="f" * 64,
            )
            self.assertEqual(report["terms_sha256"], "f" * 64)
            saved = json.loads((Path(tmp) / "q" / "qualification.json").read_text())
            self.assertEqual(saved["terms_sha256"], "f" * 64)
            with self.assertRaises(ValueError):
                qualify_node.qualify(
                    root, "python3.12", Path(tmp) / "r", run=self.fake_run([]), terms_sha256="x"
                )
        self.assertNotIn("terms_sha256", self.fake_report())

    def fake_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(__file__).resolve().parents[1]
            return qualify_node.qualify(
                root, "python3.12", Path(tmp) / "q", run=self.fake_run([]), free_mib=19500
            )

    def test_d3_uses_its_module_and_measured_sizing_and_checks_parity(self):
        calls = []
        rows = []
        cases = []
        run = self.fake_run(calls)

        def recording_run(command, **kwargs):
            if "--data" in command:
                data = Path(command[command.index("--data") + 1])
                rows.extend(json.loads(line) for line in data.read_text().splitlines())
            if "--cases" in command:
                path = Path(command[command.index("--cases") + 1])
                cases.extend(json.loads(line) for line in path.read_text().splitlines())
            return run(command, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(__file__).resolve().parents[1]
            report = qualify_node.qualify(
                root,
                "python3.12",
                Path(tmp) / "q",
                run=recording_run,
                free_mib=19500,
                model=models.D3,
            )
        text = [" ".join(map(str, c)) for c in calls]
        self.assertEqual(report["model"], models.D3)
        self.assertEqual(
            report["profile_sha256"], models.profile_identity(models.D3)["profile_sha256"]
        )
        self.assertEqual(report["parity"], {"cases": 9, "max_difference": 2.5e-8})
        self.assertEqual(report["sizing"], self.SIZING)
        self.assertTrue(report["reload_predict_ok"])
        self.assertTrue(any("zils.d3 reference" in c for c in text))
        self.assertTrue(any("zils.d3 train" in c for c in text))
        self.assertTrue(any("from zils.d3 import verify_runtime" in c for c in text))
        self.assertTrue(any("from zils.d3 import DecisionModel" in c for c in text))
        self.assertFalse(any("zils.h2o" in c for c in text))
        self.assertLess(
            next(i for i, c in enumerate(text) if "sizing probe" in c),
            next(i for i, c in enumerate(text) if "zils.d3 train" in c),
        )

        def repeats(state):
            return state["history"].count("payment issue ")

        def expected(target):
            return qualify_node.repeats_for(target, **self.SIZING)

        cap = models.spec(models.D3)["max_input_tokens"]
        self.assertEqual(len(rows), 6)
        self.assertEqual({repeats(r["state"]) for r in rows}, {expected(cap - 200)})
        self.assertEqual(expected(cap - 200), 1438)
        self.assertEqual(len(cases), 9)
        self.assertEqual({repeats(c["state"]) for c in cases[:6]}, {expected(cap - 200)})
        self.assertEqual(
            [repeats(c["state"]) for c in cases[6:]],
            [expected(1000), expected(2000), expected(cap - 200)],
        )
        self.assertEqual({c["question"]["type"] for c in cases[6:]}, {"choice"})

    def test_repeats_for_uses_measured_costs(self):
        self.assertEqual(qualify_node.repeats_for(1000, 131, 2), 434)
        self.assertEqual(qualify_node.repeats_for(100, 300, 2), 0)
        with self.assertRaises(ValueError):
            qualify_node.repeats_for(1000, 100, 0)

    def test_h2o_report_has_no_parity_and_keeps_2048_rows(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(__file__).resolve().parents[1]
            report = qualify_node.qualify(
                root, "python3.12", Path(tmp) / "q", run=self.fake_run(calls), free_mib=19500
            )
        self.assertNotIn("parity", report)
        self.assertEqual(qualify_node.STATE["history"], "payment issue " * 950)
        self.assertFalse(any("parity" in " ".join(map(str, c)) for c in calls))

    def test_below_floor_stops_before_training(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(__file__).resolve().parents[1]
            with self.assertRaises(SystemExit) as raised:
                qualify_node.qualify(
                    root, "python3.12", Path(tmp) / "q", run=self.fake_run(calls), free_mib=12000
                )
            report = json.loads((Path(tmp) / "q" / "qualification.json").read_text())
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(report["capacity_ok"])
        self.assertEqual(report["gpu"]["free_mib_at_start"], 12000)
        self.assertFalse(any("train" in " ".join(map(str, c)) for c in calls))

    def qualify_failing(self, calls, intercept, prepare=None):
        """Run qualify() through a fake whose intercept(command) may override a result."""
        base = self.fake_run(calls)

        def run(command, **kwargs):
            try:
                outcome = intercept(command)
            except Exception:
                calls.append(command)
                raise
            if outcome is not None:
                calls.append(command)
                return outcome
            return base(command, **kwargs)

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "q"
            if prepare:
                prepare(out)
            with self.assertRaises(SystemExit) as raised:
                qualify_node.qualify(root, "python3.12", out, run=run, free_mib=19500)
            return raised.exception.code, json.loads((out / "qualification.json").read_text())

    def test_train_failure_records_error_and_exits_5(self):
        calls = []

        def intercept(command):
            if "train" in command:
                raise subprocess.CalledProcessError(1, command, stderr="boom")

        code, report = self.qualify_failing(calls, intercept)
        self.assertEqual(code, 5)
        self.assertIn("boom", report["error"])
        self.assertIn("train", report["error"])
        self.assertNotIn("training", report)

    def test_failure_replaces_stale_report(self):
        def prepare(out):
            out.mkdir()
            (out / "qualification.json").write_text(json.dumps({"reload_predict_ok": True}))

        def intercept(command):
            if "verify_runtime" in " ".join(map(str, command)):
                raise subprocess.TimeoutExpired(command, 300)

        code, report = self.qualify_failing([], intercept, prepare)
        self.assertEqual(code, 5)
        self.assertNotIn("reload_predict_ok", report)
        self.assertIn("runtime check", report["error"])

    def test_existing_adapter_is_rejected_before_any_subprocess(self):
        calls = []

        def prepare(out):
            (out / "adapter").mkdir(parents=True)

        code, report = self.qualify_failing(calls, lambda command: None, prepare)
        self.assertEqual(code, 5)
        self.assertIn("adapter directory exists", report["error"])
        self.assertEqual(calls, [])

    def test_reload_probe_failure_records_stderr_and_exits_4(self):
        def intercept(command):
            if "reload_predict" in " ".join(map(str, command)):
                return SimpleNamespace(returncode=1, stdout="", stderr="probe failed")

        code, report = self.qualify_failing([], intercept)
        self.assertEqual(code, 4)
        self.assertIn("probe failed", report["error"])
        self.assertFalse(report["reload_predict_ok"])

    def test_half_written_reference_is_recreated(self):
        calls = []
        stale = []

        def prepare(out):
            (out / "reference").mkdir(parents=True)
            (out / "reference" / "partial.bin").write_text("x")
            stale.append(out / "reference" / "partial.bin")

        def intercept(command):
            if "reference" in command:
                self.assertFalse(stale[0].exists())
                raise subprocess.CalledProcessError(1, command, stderr="stop here")

        code, report = self.qualify_failing(calls, intercept, prepare)
        self.assertEqual(code, 5)
        self.assertTrue(any("reference" in c for c in calls))
        self.assertIn("reference", report["error"])
