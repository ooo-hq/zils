"""Every unit a role can enable ships in deploy/units with the pinned command shape."""

import re
import unittest
from pathlib import Path

from zils import node_role

ROOT = Path(__file__).resolve().parents[1]
EXEC_START = {
    "zils-queue-miner.service": "ExecStart=/opt/zils/.venv-kev/bin/python -m miner.queue ",
    "zils-fleet-validator.service": (
        "ExecStart=/opt/zils/.venv-kev/bin/python -m zils.fleet validator "
    ),
}
LOCK_DIR = "/var/lock/zils-training"


class NodeUnitsTest(unittest.TestCase):
    def test_every_role_unit_runs_its_module_from_pinned_environment(self):
        for role, spec in node_role.ROLES.items():
            for unit in spec["units"]:
                with self.subTest(role=role, unit=unit):
                    text = (ROOT / "deploy/units" / unit).read_text()
                    self.assertIn(EXEC_START[unit], text)
                    self.assertIn("EnvironmentFile=/etc/zils-node/node.env", text)
                    self.assertIn("ProtectSystem=strict", text)
                    self.assertIn("User=zilsminer", text)
                    self.assertIn("SupplementaryGroups=zils-compute", text)
                    self.assertIn("--runtime-python /opt/zils/.venv-h2o/bin/python", text)
                    self.assertIn("--device cuda", text)
                    self.assertNotIn("SUPABASE", text)
                    paths = re.search(r"^ReadWritePaths=(.*)$", text, re.MULTILINE)
                    self.assertIsNotNone(paths)
                    self.assertIn("/var/lib/zils-node", paths.group(1).split())
                    self.assertIn(LOCK_DIR, paths.group(1).split())

    def test_every_role_unit_has_an_expected_module(self):
        for spec in node_role.ROLES.values():
            for unit in spec["units"]:
                self.assertIn(unit, EXEC_START)

    def test_env_example_has_no_supabase_settings(self):
        text = (ROOT / "deploy/units/zils-node.env.example").read_text()
        self.assertNotIn("SUPABASE", text)

    def test_lock_directory_is_recreated_at_boot(self):
        lines = (ROOT / "deploy/tmpfiles.d/zils-node.conf").read_text().splitlines()
        self.assertEqual(len(lines), 1)
        fields = lines[0].split()
        self.assertEqual(fields[0], "d")
        self.assertEqual(fields[1], LOCK_DIR)
        self.assertEqual(fields[2], "0770")
        self.assertEqual(fields[4], "zils-compute")

    def test_apply_script_uses_role_check_and_never_hardcodes_units(self):
        text = (ROOT / "scripts/apply_role.sh").read_text()
        self.assertIn("zils.node_role check", text)
        self.assertNotIn("systemctl enable zils-queue-miner", text)
