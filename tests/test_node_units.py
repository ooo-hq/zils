"""Every unit a role can enable ships in deploy/units with the pinned command shape."""

import unittest
from pathlib import Path

from zils import node_role

ROOT = Path(__file__).resolve().parents[1]


class NodeUnitsTest(unittest.TestCase):
    def test_every_role_unit_exists_and_runs_from_pinned_environment(self):
        for role, spec in node_role.ROLES.items():
            for unit in spec["units"]:
                text = (ROOT / "deploy/units" / unit).read_text()
                self.assertIn("EnvironmentFile=/etc/zils-node/node.env", text, unit)
                self.assertRegex(
                    text,
                    r"ExecStart=/opt/zils/\.venv-kev/bin/python -m (miner\.queue|zils\.fleet)",
                    unit,
                )
                self.assertIn("ProtectSystem=strict", text, unit)
                self.assertNotIn("SUPABASE", text, unit)

    def test_apply_script_uses_role_check_and_never_hardcodes_units(self):
        text = (ROOT / "scripts/apply_role.sh").read_text()
        self.assertIn("zils.node_role check", text)
        self.assertNotIn("systemctl enable zils-queue-miner", text)
