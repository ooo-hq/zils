#!/usr/bin/env bash
# Enable exactly the units /etc/zils-node/role.json selects; disable the rest.
set -euo pipefail
ROOT=/opt/zils
cd "$ROOT"
mapfile -t WANTED < <("$ROOT/.venv-kev/bin/python" -m zils.node_role check)
if [ "${#WANTED[@]}" -eq 0 ]; then echo "role check selected no units" >&2; exit 1; fi
"$ROOT/.venv-kev/bin/python" - <<'PY'
import json, os, pathlib, shutil
role = json.loads(pathlib.Path("/etc/zils-node/role.json").read_text())
if role["role"] == "miner":
    cfg = {"coordinator": role["coordinator"], "hotkey": role["hotkey"], "wallet": role["wallet"]}
    p = "/etc/zils-node/queue-miner.json"
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
    os.fchmod(fd, 0o640)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(cfg, indent=2) + "\n")
    shutil.chown(p, group="zils-compute")
# Operator-supplied wallet files must be readable by the service group, never world-readable.
etc, wallets = pathlib.Path("/etc/zils-node"), pathlib.Path("/etc/zils-node/wallets")
for p in [etc] + ([wallets] + list(wallets.rglob("*")) if wallets.exists() else []):
    if p.is_symlink():
        continue
    shutil.chown(p, group="zils-compute")
    if p != etc:
        p.chmod(0o750 if p.is_dir() else 0o640)
PY
for f in "$ROOT"/deploy/units/*.service; do
  install -m 0644 "$f" /etc/systemd/system/
done
systemctl daemon-reload
for f in "$ROOT"/deploy/units/*.service; do
  unit=$(basename "$f")
  if [[ " ${WANTED[*]} " == *" $unit "* ]]; then
    systemctl enable --now "$unit"
  else
    systemctl disable --now "$unit" 2>/dev/null || true
  fi
done
systemctl --no-pager --no-legend list-units 'zils-*' || true
