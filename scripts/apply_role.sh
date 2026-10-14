#!/usr/bin/env bash
# Enable exactly the units /etc/zils-node/role.json selects; disable the rest.
set -euo pipefail
ROOT=/opt/zils
cd "$ROOT"
mapfile -t WANTED < <("$ROOT/.venv-kev/bin/python" -m zils.node_role check)
if [ "${#WANTED[@]}" -eq 0 ]; then echo "role check selected no units" >&2; exit 1; fi
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
