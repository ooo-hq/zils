"""Compatibility at the package, subprocess, settings, and signature boundaries."""

import importlib
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import fez
import zils
from zils import queue_protocol, runtime, settings


class InternalNamesTest(unittest.TestCase):
    def test_legacy_exports_and_modules_are_the_canonical_implementation(self):
        self.assertIs(fez.score, zils.score)
        for name in ("cloud", "models", "coordinator", "runtime", "api", "queue_protocol"):
            with self.subTest(name=name):
                old = importlib.import_module("fez." + name)
                new = importlib.import_module("zils." + name)
                self.assertIs(old, new)
        with patch("fez.runtime.gpu_ready", return_value=False):
            self.assertFalse(runtime.gpu_ready("cuda"))

    def test_settings_precedence_empty_and_legacy(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.get("ZILS_WEB_ORIGIN", "default"), "default")
            os.environ["FEZ_WEB_ORIGIN"] = "https://legacy.example"
            self.assertEqual(settings.required("ZILS_WEB_ORIGIN"), "https://legacy.example")
            os.environ["ZILS_WEB_ORIGIN"] = "https://canonical.example"
            self.assertEqual(settings.required("ZILS_WEB_ORIGIN"), "https://canonical.example")
            os.environ["ZILS_WEB_ORIGIN"] = ""
            with self.assertRaisesRegex(ValueError, "ZILS_WEB_ORIGIN"):
                settings.required("ZILS_WEB_ORIGIN")

    def test_canonical_capacity_gate_overrides_legacy_setting(self):
        with patch.dict(
            os.environ,
            {
                "ZILS_GPU_MIN_FREE_MIB": "123",
                "FEZ_GPU_MIN_FREE_MIB": "456",
                "ZILS_NVIDIA_SMI": "/canonical/probe",
            },
            clear=True,
        ):
            with patch(
                "zils.runtime.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, stdout="124"),
            ) as probe:
                self.assertTrue(runtime.gpu_ready("cuda"))
                self.assertEqual(probe.call_args.args[0][0], "/canonical/probe")

    def test_persisted_signature_domains_do_not_change(self):
        self.assertEqual(queue_protocol.canonical({}), b"fez-training-queue/v1\0{}")
        self.assertEqual(runtime.canonical({}), b"fez-fleet/v1\0{}")

    def test_new_and_old_cli_and_direct_runners(self):
        for namespace in ("zils", "fez"):
            for module in ("", ".coordinator", ".workflow"):
                result = subprocess.run(
                    [sys.executable, "-m", namespace + module, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
            for runner in ("kev_runner.py", "jevk5_runner.py"):
                result = subprocess.run(
                    [sys.executable, str(zils.ROOT / namespace / runner), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
