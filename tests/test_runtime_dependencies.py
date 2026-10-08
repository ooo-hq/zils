"""Inference runtimes do not need the payment provider SDK installed."""

import subprocess
import sys
import unittest


class RuntimeDependenciesTest(unittest.TestCase):
    def test_runtime_registry_imports_without_stripe(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules['stripe'] = None; from zils.api import Registry; Registry([])",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
