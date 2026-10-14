"""Node role configuration selects units and refuses incomplete identities."""

import json
import tempfile
import unittest
from pathlib import Path

from zils import node_role


def write(path, value):
    path.write_text(json.dumps(value))
    return path


class NodeRoleTest(unittest.TestCase):
    def miner(self):
        return {
            "role": "miner",
            "coordinator": "https://training.zils.ai",
            "hotkey": "5GnLjWkMA6UhM4K7k94j8tWrxPGvECzwbRMdbEgPAUeMbJuj",
            "wallet": {
                "name": "fez-testnet",
                "hotkey": "gpu4090",
                "path": "/etc/zils-node/wallets",
            },
            "device": "cuda",
        }

    def test_miner_role_enables_queue_miner_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wallets/fez-testnet/hotkeys").mkdir(parents=True)
            (root / "wallets/fez-testnet/hotkeys/gpu4090").write_text("{}")
            role = self.miner()
            role["wallet"]["path"] = str(root / "wallets")
            path = write(root / "role.json", role)
            self.assertEqual(node_role.check(path, root=root), ["zils-queue-miner.service"])

    def test_fleet_validator_role_enables_validator_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wallets/fez-testnet/hotkeys").mkdir(parents=True)
            (root / "wallets/fez-testnet/hotkeys/validator").write_text("{}")
            (root / "fleet/validator").mkdir(parents=True)
            (root / "fleet/validator/config.json").write_text("{}")
            role = {
                "role": "fleet-validator",
                "hotkey": "5HTAVUuHxV81VbYoX7DYFzebdyA1SRk3zazdZiiBN7zCcpxy",
                "wallet": {
                    "name": "fez-testnet",
                    "hotkey": "validator",
                    "path": str(root / "wallets"),
                },
                "fleet_config": str(root / "fleet/validator/config.json"),
                "device": "cuda",
            }
            path = write(root / "role.json", role)
            self.assertEqual(node_role.check(path, root=root), ["zils-fleet-validator.service"])
            (root / "fleet-config.json").write_text("{}")
            role["fleet_config"] = str(root / "fleet-config.json")
            path = write(root / "role.json", role)
            with self.assertRaisesRegex(
                ValueError, "fleet_config must be .*fleet/validator/config.json"
            ):
                node_role.check(path, root=root)

    def test_unknown_role_and_missing_field_are_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "role must be one of"):
                node_role.check(write(root / "a.json", {"role": "serve-h2o"}), root=root)
            role = self.miner()
            del role["coordinator"]
            with self.assertRaisesRegex(ValueError, "coordinator"):
                node_role.check(write(root / "b.json", role), root=root)

    def test_missing_hotkey_file_refuses_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            role = self.miner()
            role["wallet"]["path"] = str(root / "wallets")
            with self.assertRaisesRegex(ValueError, "hotkeys/gpu4090"):
                node_role.check(write(root / "role.json", role), root=root)

    def test_identity_files_outside_root_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            wallets = Path(outside) / "wallets"
            (wallets / "fez-testnet/hotkeys").mkdir(parents=True)
            (wallets / "fez-testnet/hotkeys/gpu4090").write_text("{}")
            role = self.miner()
            role["wallet"]["path"] = str(wallets)
            with self.assertRaisesRegex(ValueError, "root"):
                node_role.check(write(root / "role.json", role), root=root)

    def test_examples_validate_against_schema_fields(self):
        root = Path(__file__).resolve().parents[1] / "deploy/role-examples"
        for name in ("miner.json", "fleet-validator.json"):
            role = node_role.load_role(root / name)
            self.assertIn(role["role"], ("miner", "fleet-validator"))
            self.assertTrue(node_role.units_for(role))
