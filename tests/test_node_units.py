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
                    self.assertIn("--runtime-python /opt/zils/.venv-d3/bin/python", text)
                    self.assertNotIn("h2o", text)
                    self.assertIn("--device cuda", text)
                    self.assertNotIn("SUPABASE", text)
                    paths = re.search(r"^ReadWritePaths=(.*)$", text, re.MULTILINE)
                    self.assertIsNotNone(paths)
                    self.assertIn("/var/lib/zils-node", paths.group(1).split())
                    self.assertIn(LOCK_DIR, paths.group(1).split())

    def test_queue_miner_passes_the_d3_reference_only(self):
        text = (ROOT / "deploy/units/zils-queue-miner.service").read_text()
        self.assertEqual(text.count("--reference "), 1)
        self.assertIn("--reference /opt/zils/models/d3-reference", text)

    def test_env_example_names_the_d3_runtime(self):
        text = (ROOT / "deploy/units/zils-node.env.example").read_text()
        self.assertNotIn("H2O", text)
        self.assertIn("ZILS_D3_RUNTIME_PYTHON=/opt/zils/.venv-d3/bin/python\n", text)

    def test_every_role_unit_has_an_expected_module(self):
        for spec in node_role.ROLES.values():
            for unit in spec["units"]:
                self.assertIn(unit, EXEC_START)

    def test_env_example_has_no_supabase_settings(self):
        text = (ROOT / "deploy/units/zils-node.env.example").read_text()
        self.assertNotIn("SUPABASE", text)

    def test_env_example_paths_are_ones_the_provisioner_creates(self):
        script = (ROOT / "scripts/provision_node.sh").read_text()
        text = (ROOT / "deploy/units/zils-node.env.example").read_text()
        found = re.findall(r"^[A-Z_0-9]+=(/\S*)$", text, re.MULTILINE)
        self.assertTrue(found)
        for path in found:
            with self.subTest(path=path):
                made = path.startswith(("/var/lib/zils-node", "/var/lock/zils-training")) or (
                    path.startswith("/opt/zils/") and path[len("/opt/zils/") :] in script
                )
                self.assertTrue(made, path)
        self.assertNotIn("ZILS_H2O_BASE_DIR", text)

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

    def test_apply_script_enables_graded_presence_for_miners(self):
        text = (ROOT / "scripts/apply_role.sh").read_text()
        self.assertIn('"graded_scheduling": True', text)
        self.assertIn('"graded_min_free_mib": 12288', text)
