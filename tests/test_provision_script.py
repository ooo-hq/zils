"""The provisioner is idempotent, checks the driver first and never touches secrets."""

import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/provision_node.sh"
PHASES = [
    "driver",
    "accounts",
    "source",
    "environments",
    "references",
    "units",
    "qualify",
    "image",
]


class ProvisionScriptTest(unittest.TestCase):
    def dry_run(self):
        return subprocess.run(
            ["bash", str(SCRIPT)],
            env={**os.environ, "ZILS_NODE_DRY_RUN": "1"},
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(SCRIPT)]).returncode, 0)

    def test_dry_run_lists_phases_in_order_and_is_stable(self):
        first, second = self.dry_run(), self.dry_run()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        phases = [
            line.split(":")[1].strip()
            for line in first.stdout.splitlines()
            if line.startswith("phase:")
        ]
        self.assertEqual(phases, PHASES)

    def test_script_never_reads_secret_locations(self):
        text = SCRIPT.read_text()
        forbidden = (
            "coldkey",
            "SUPABASE_SERVICE_ROLE_KEY",
            "ZILS_RUNTIME_TOKEN",
            "/etc/zils-node/hotkey",
            "/etc/zils-node/wallets",
        )
        for item in forbidden:
            self.assertNotIn(item, text, item)
