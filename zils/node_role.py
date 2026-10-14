"""Select the systemd units a node runs from /etc/zils-node/role.json.

A node has exactly one role. Missing identity files refuse the role; there is
no fallback to a weaker role. Serving and hosted-validator roles are installed
by the platform repository and are not accepted here.
"""

import argparse
import json
import sys
from pathlib import Path

ROLES = {
    "miner": {
        "fields": ("coordinator", "hotkey", "wallet", "device"),
        "units": ["zils-queue-miner.service"],
    },
    "fleet-validator": {
        "fields": ("hotkey", "wallet", "fleet_config", "device"),
        "units": ["zils-fleet-validator.service"],
    },
}
WALLET_FIELDS = ("name", "hotkey", "path")


def load_role(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value.get("role") not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    for field in ROLES[value["role"]]["fields"]:
        if field not in value:
            raise ValueError(f"role.json is missing required field: {field}")
    wallet = value["wallet"]
    if not isinstance(wallet, dict) or any(k not in wallet for k in WALLET_FIELDS):
        raise ValueError(f"wallet must contain {WALLET_FIELDS}")
    if value["device"] != "cuda":
        raise ValueError("fleet nodes run on cuda")
    return value


def required_files(role):
    wallet = role["wallet"]
    files = [Path(wallet["path"]) / wallet["name"] / "hotkeys" / wallet["hotkey"]]
    if role["role"] == "fleet-validator":
        files.append(Path(role["fleet_config"]))
    return files


def units_for(role):
    return list(ROLES[role["role"]]["units"])


def check(path, root=Path("/etc/zils-node")):
    role = load_role(path)
    missing = [str(p) for p in required_files(role) if not p.is_file()]
    if missing:
        raise ValueError("refusing role; missing identity files: " + ", ".join(missing))
    return units_for(role)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("check")
    cmd.add_argument("--path", default="/etc/zils-node/role.json")
    cmd.add_argument("--root", default="/etc/zils-node")
    args = parser.parse_args(argv)
    try:
        for unit in check(Path(args.path), root=Path(args.root)):
            print(unit)
    except (OSError, ValueError) as error:
        print(f"role check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
