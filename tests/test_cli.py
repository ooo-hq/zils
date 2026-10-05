"""Public commands must work from the repository without installing a package."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


class CLITest(unittest.TestCase):
    def test_module_entry_points(self):
        root = Path(__file__).resolve().parents[1]
        modules = [
            "zils",
            "zils.benchmark",
            "zils.jobs",
            "zils.calibrate",
            "zils.testnet",
            "scripts.download_models",
        ]
        if importlib.util.find_spec("bittensor_wallet"):
            modules += [
                "miner",
                "zils.fleet",
                "scripts.rehearsal",
                "zils.coordinator",
                "miner.queue",
            ]
        for module in modules:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
