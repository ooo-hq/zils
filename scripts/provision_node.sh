#!/usr/bin/env bash
# Provision a Zils fleet node from a bare GPU droplet. Idempotent: every phase
# writes a marker under /var/lib/zils-node/markers and skips itself when the
# marker exists. ZILS_NODE_DRY_RUN=1 prints the plan and changes nothing.
set -euo pipefail
# Service units and hardened shells may inherit a restrictive umask; the service
# user must be able to read and execute everything this script installs.
umask 022

ZILS_SUBNET_REF="${ZILS_SUBNET_REF:-main}"
ROOT=/opt/zils
STATE=/var/lib/zils-node
MARKERS=$STATE/markers
DRY="${ZILS_NODE_DRY_RUN:-0}"
case "$DRY" in 0|"") DRY=0;; *) DRY=1;; esac

phase() { echo "phase: $1"; }
# Check out the fetched remote branch, else a tag or exact commit, detached. A stale local branch of the
# same name (an old snapshot's `main`) must never win over what was just fetched.
checkout_ref() {
  local target="$2"
  if git -C "$1" rev-parse -q --verify "origin/$2^{commit}" >/dev/null; then target="origin/$2"; fi
  git -C "$1" checkout -q --detach "$target"
}
done_marker() { [ "$DRY" = 1 ] || touch "$MARKERS/$1"; }
skip() { [ "$DRY" = 0 ] && [ -f "$MARKERS/$1" ]; }

main() {
  phase driver
  if [ "$DRY" = 0 ]; then
    command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing; use the NVIDIA AI/ML Ready image" >&2; exit 10; }
    if ! DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) || [ -z "$DRIVER" ]; then
      echo "nvidia-smi failed; is the NVIDIA driver installed?" >&2
      exit 10
    fi
    DRIVER_MAJOR=${DRIVER%%.*}
    # CUDA 13.0 needs driver >= 580 (NVIDIA release notes); refuse before downloading anything.
    [ "$DRIVER_MAJOR" -ge 580 ] || { echo "driver $DRIVER is older than the CUDA 13.0 minimum (580)" >&2; exit 11; }
  fi

  phase accounts
  if ! skip accounts; then
    if [ "$DRY" = 0 ]; then
      getent group zils-compute >/dev/null || groupadd zils-compute
      id zilsminer >/dev/null 2>&1 || useradd --system --home-dir /var/lib/zils-node --shell /usr/sbin/nologin -G zils-compute zilsminer
      install -d -m 0750 -o zilsminer -g zilsminer $STATE $STATE/queue-miner $STATE/cache $STATE/qualification-d3 $MARKERS
      install -d -m 0750 -o zilsminer -g zilsminer "$STATE/cache/triton" "$STATE/cache/inductor" "$STATE/cache/hf"
      install -d -m 0770 -o root -g zils-compute /var/lock/zils-training
      install -d -m 0750 -o root -g zils-compute /etc/zils-node
      install -d -m 0770 -o root -g zils-compute /etc/zils-node/fleet
    fi
    done_marker accounts
  fi

  phase source
  if ! skip source; then
    if [ "$DRY" = 0 ]; then
      apt-get update -q && apt-get install -y -q git curl >/dev/null
      export UV_NO_MODIFY_PATH=1 UV_PYTHON_INSTALL_DIR=/opt/uv-python
      command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="${HOME:-/root}/.local/bin:$PATH"
      if [ ! -d $ROOT/.git ]; then git clone -q https://github.com/ooo-hq/zils.git $ROOT; fi
      git -C $ROOT fetch -q --tags origin && checkout_ref $ROOT "$ZILS_SUBNET_REF"
      chown -R root:zils-compute $ROOT && chmod -R g+rX $ROOT
    fi
    done_marker source
    # Later phases must come from the checked-out commit, not the copy of this script that started.
    if [ "$DRY" = 0 ] && [ -z "${ZILS_NODE_REEXEC:-}" ] && [ -f "$0" ] \
      && ! cmp -s "$0" $ROOT/scripts/provision_node.sh; then
      ZILS_NODE_REEXEC=1 exec bash $ROOT/scripts/provision_node.sh "$@"
    fi
  fi

  phase environments
  if ! skip environments; then
    if [ "$DRY" = 0 ]; then
      export PATH="${HOME:-/root}/.local/bin:$PATH"
      export UV_PYTHON_INSTALL_DIR=/opt/uv-python
      cd $ROOT
      [ -x .venv-kev/bin/python ] || uv venv --python 3.13 .venv-kev
      uv pip install -q --python .venv-kev/bin/python --torch-backend=cu128 -r requirements/model.txt -r requirements/rehearsal.txt -r requirements/testnet.txt
      .venv-kev/bin/python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()'
      [ -x .venv-d3/bin/python ] || uv venv --python 3.12 .venv-d3
      uv pip install -q --python .venv-d3/bin/python --torch-backend=cu130 -r requirements/d3.txt
      # The venvs symlink into the managed interpreters; the service user must be able to exec them.
      [ -d /opt/uv-python ] && chmod -R a+rX /opt/uv-python || true
      # uv hardlinks site-packages from its cache, so the files keep the mode they were cached with.
      chmod -R a+rX .venv-kev .venv-d3
    fi
    done_marker environments
  fi

  phase references
  if ! skip references; then
    if [ "$DRY" = 0 ]; then
      cd $ROOT
      [ -f models/d3-reference/model.json ] || HF_HOME=$STATE/cache/hf .venv-d3/bin/python -m zils.d3 reference --out models/d3-reference
      chown -R root:zils-compute models && chmod -R g+rX models
    fi
    done_marker references
  fi

  phase units
  if ! skip units; then
    if [ "$DRY" = 0 ]; then
      install -m 0644 $ROOT/deploy/units/*.service /etc/systemd/system/
      install -m 0644 $ROOT/deploy/tmpfiles.d/zils-node.conf /etc/tmpfiles.d/zils-node.conf
      systemd-tmpfiles --create /etc/tmpfiles.d/zils-node.conf
      [ -f /etc/zils-node/node.env ] || install -m 0640 -g zils-compute $ROOT/deploy/units/zils-node.env.example /etc/zils-node/node.env
      sed -i "s#^HF_HOME=.*#HF_HOME=$STATE/cache/hf#" /etc/zils-node/node.env
      systemctl daemon-reload
      for unit in $ROOT/deploy/units/*.service; do systemctl disable "$(basename "$unit")" 2>/dev/null || true; done
    fi
    done_marker units
  fi

  phase qualify
  if ! skip qualify; then
    if [ "$DRY" = 0 ]; then
      cd $ROOT
      chown -R zilsminer:zilsminer "$STATE/cache" "$STATE/qualification-d3"
      # A failed run leaves its adapter and training data behind; clear them so the phase is retryable.
      rm -rf "$STATE/qualification-d3/adapter" "$STATE/qualification-d3/train.jsonl"
      # Run as the service user inside the unit sandbox; EnvironmentFile supplies node.env.
      qualify_runtime() {
        systemd-run --wait --pipe --collect -p User=zilsminer -p Group=zilsminer \
          -p SupplementaryGroups=zils-compute -p WorkingDirectory=$ROOT \
          -p EnvironmentFile=/etc/zils-node/node.env -p ProtectSystem=strict \
          -p "ReadWritePaths=$STATE /var/lock/zils-training" -p PrivateTmp=yes -p NoNewPrivileges=yes \
          $ROOT/.venv-kev/bin/python -m scripts.qualify_node --root $ROOT "$@"
      }
      qualify_runtime --model d3-mini-4.5b-v1 --runtime-python $ROOT/.venv-d3/bin/python --out $STATE/qualification-d3
    fi
    done_marker qualify
  fi

  phase image
  if [ "$DRY" = 0 ]; then
    cd $ROOT
    PINS=$(sha256sum zils/d3-pins.json requirements/d3.txt | sha256sum | cut -c1-16)
    Q=$(sha256sum $STATE/qualification-d3/qualification.json | cut -c1-16)
    printf '{"subnet_ref":"%s","subnet_commit":"%s","pins_sha256_16":"%s","qualification_sha256_16":"%s","built_at":"%s"}\n' \
      "$ZILS_SUBNET_REF" "$(git rev-parse HEAD)" "$PINS" "$Q" "$(date -u +%FT%TZ)" > /etc/zils-node/image.json
    echo "image: $(cat /etc/zils-node/image.json)"
  fi
  echo "provisioned; apply a role with: scripts/apply_role.sh after writing /etc/zils-node/role.json"
}

main "$@"
