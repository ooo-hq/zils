#!/bin/sh
# Zils miner installer: clones zils, builds both runtimes, puts `zils` on PATH, runs `zils setup`.
#   curl -fsSL https://raw.githubusercontent.com/ooo-hq/zils/main/scripts/install.sh | sh
# Rerun it to update. ZILS_HOME (default ~/.zils), ZILS_REF (default main) and ZILS_REPO override.
set -eu

home="${ZILS_HOME:-$HOME/.zils}"
ref="${ZILS_REF:-main}"
repo="${ZILS_REPO:-https://github.com/ooo-hq/zils.git}"
dir="$home/zils"
bin="$HOME/.local/bin"

command -v git >/dev/null || { echo "zils: install git first" >&2; exit 1; }
if ! command -v uv >/dev/null; then
  echo "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$bin:$PATH"
fi

if [ -d "$dir/.git" ]; then
  echo "updating $dir to $ref"
  git -C "$dir" fetch --quiet "$repo" "$ref"
  git -C "$dir" checkout --quiet --detach FETCH_HEAD
else
  echo "cloning zils into $dir"
  mkdir -p "$home"
  git clone --quiet --branch "$ref" "$repo" "$dir"
fi
cd "$dir"

echo "building the miner runtime (.venv-kev, Python 3.13)"
uv venv --quiet --allow-existing --python 3.13 .venv-kev
uv pip install --quiet --python .venv-kev/bin/python -r requirements/model.txt \
  -r requirements/rehearsal.txt -r requirements/testnet.txt -e .

if [ "$(uname -s)-$(uname -m)" = Linux-x86_64 ] && command -v nvidia-smi >/dev/null; then
  echo "building the d3 training runtime (.venv-d3, Python 3.12, CUDA 13.0)"
  uv venv --quiet --allow-existing --python 3.12 .venv-d3
  uv pip install --quiet --python .venv-d3/bin/python --torch-backend=cu130 -r requirements/d3.txt
else
  echo "skipped the d3 training runtime: it needs Linux x86_64 with an NVIDIA GPU"
fi

mkdir -p "$bin"
ln -sf "$dir/.venv-kev/bin/zils" "$bin/zils"
case ":$PATH:" in
  *":$bin:"*) ;;
  *) echo "add $bin to PATH: echo 'export PATH=\"$bin:\$PATH\"' >> ~/.profile" ;;
esac

echo "installed: $bin/zils"
if [ -r /dev/tty ]; then
  "$bin/zils" setup </dev/tty || true
else
  echo "next: zils setup"
fi
