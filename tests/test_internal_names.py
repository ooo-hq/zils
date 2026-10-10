"""Canonical entry points and compatibility for deployed settings and signatures."""

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import zils
from zils import queue_protocol, runtime, settings


class InternalNamesTest(unittest.TestCase):
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

    def test_cli_and_direct_runners(self):
        for module in ("zils", "miner.queue", "zils.fleet"):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
        for runner in ("kev_runner.py", "jevk5_runner.py"):
            with self.subTest(runner=runner):
                result = subprocess.run(
                    [sys.executable, str(zils.ROOT / "zils" / runner), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
