#!/bin/sh
# Zils miner installer: clones zils, builds both runtimes, puts `zils-miner` on PATH, runs `zils-miner setup`.
# `zils` itself is the customer CLI (npm @zils/cli); it forwards miner commands to `zils-miner`.
#   curl -fsSL https://zils.ai/install.sh | sh
# `zils-miner update` reruns it. It installs the commit named in release/stable on main; releasing is a
# PR that changes that sha. ZILS_HOME (default ~/.zils), ZILS_REF (any branch or sha, e.g. main)
# and ZILS_REPO override; ZILS_SKIP_SETUP skips `zils-miner setup`; NO_COLOR turns colors off.
set -eu

home="${ZILS_HOME:-$HOME/.zils}"
repo="${ZILS_REPO:-https://github.com/ooo-hq/zils.git}"
stable_url="${ZILS_STABLE_URL:-https://raw.githubusercontent.com/ooo-hq/zils/main/release/stable}"
dir="$home/zils"
bin="$HOME/.local/bin"
logs="$home/logs"
user_path="$PATH"
PATH="$bin:$PATH"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  esc=$(printf '\033')
  red="$esc[0;31m" green="$esc[0;32m" yellow="$esc[0;33m" cyan="$esc[0;36m"
  brand="$esc[1;32m" bold="$esc[1m" dim="$esc[2m" reset="$esc[0m"
  live=1
else
  red="" green="" yellow="" cyan="" brand="" bold="" dim="" reset="" live=""
fi

log() { printf '%s→%s %s\n' "$cyan" "$reset" "$*"; }
ok() { printf '%s✓%s %s\n' "$green" "$reset" "$*"; }
warn() { printf '%s⚠%s %s\n' "$yellow" "$reset" "$*"; }
fail() { printf '%s✗ %s%s\n' "$red" "$*" "$reset" >&2; exit 1; }

banner() {
  printf '\n%s' "$brand"
  printf '  ┌─────────────────────────────────────────────────────┐\n'
  printf '  │  ◆ Zils Miner Installer                             │\n'
  printf '  ├─────────────────────────────────────────────────────┤\n'
  printf '  │  Decision models, trained by Bittensor miners.      │\n'
  printf '  └─────────────────────────────────────────────────────┘\n'
  printf '%s\n' "$reset"
}

# run LABEL DONE COMMAND...: log the command to a file; on a terminal, show its newest output line
# in place. On failure print the last 20 lines and the log path.
run() {
  label=$1 done=$2
  shift 2
  file="$logs/$(printf '%s' "$label" | tr -c 'a-zA-Z0-9\n' '-').log"
  log "$label"
  if [ -n "$live" ]; then
    cols=$(($(tput cols 2>/dev/null || echo 80) - 4))
    "$@" >"$file" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      line=$(tail -c 2000 "$file" 2>/dev/null | tr '\r' '\n' | grep -v '^[[:space:]]*$' | tail -n 1 | cut -c "1-$cols")
      printf '\r\033[K  %s%s%s' "$dim" "$line" "$reset"
      sleep 0.2
    done
    printf '\r\033[K'
    wait "$pid" && status=0 || status=$?
  else
    "$@" >"$file" 2>&1 && status=0 || status=$?
  fi
  if [ "$status" -ne 0 ]; then
    printf '%s✗ %s failed (exit %s). Last output:%s\n' "$red" "$label" "$status" "$reset" >&2
    tail -n 20 "$file" | sed 's/^/    /' >&2
    fail "full log: $file"
  fi
  ok "$done"
}

# Add ~/.local/bin to the login shell's rc file once, under a marker comment.
wire_path() {
  case ":$user_path:" in *":$bin:"*) return ;; esac
  case "${SHELL:-}" in
    */zsh) rc="$HOME/.zshrc" line="export PATH=\"$bin:\$PATH\"" reload="source ~/.zshrc" ;;
    */bash) rc="$HOME/.bashrc" line="export PATH=\"$bin:\$PATH\"" reload="source ~/.bashrc" ;;
    */fish) rc="$HOME/.config/fish/config.fish" line="fish_add_path $bin" reload="source ~/.config/fish/config.fish" ;;
    *) rc="$HOME/.profile" line="export PATH=\"$bin:\$PATH\"" reload=". ~/.profile" ;;
  esac
  if ! grep -qs '# Zils miner command' "$rc"; then
    mkdir -p "$(dirname "$rc")"
    printf '\n# Zils miner command\n%s\n' "$line" >>"$rc"
    ok "added ~/.local/bin to PATH in ~/${rc#"$HOME"/}"
  fi
  path_hint="$reload"
}

banner
mkdir -p "$logs"

command -v git >/dev/null || fail "git is required; install it and run this again"
command -v curl >/dev/null || fail "curl is required; install it and run this again"
if command -v uv >/dev/null; then
  ok "uv $(uv --version | cut -d' ' -f2)"
else
  run "installing uv" "uv installed" sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
fi

if [ -n "${ZILS_REF:-}" ]; then
  ref=$ZILS_REF
else
  ref=$(curl -fsSL "$stable_url") || fail "could not read the stable release from $stable_url"
  case "$ref" in *[!0-9a-f]* | "") fail "$stable_url is not a commit sha" ;; esac
  [ "${#ref}" -eq 40 ] || fail "$stable_url is not a full 40-hex commit sha"
fi

if [ ! -d "$dir/.git" ]; then
  mkdir -p "$home"
  run "cloning zils" "zils cloned into ~/${dir#"$HOME"/}" git clone --progress --no-checkout "$repo" "$dir"
fi
run "checking out $ref" "zils at $ref" \
  sh -c "git -C '$dir' fetch --progress '$repo' '$ref' && git -C '$dir' checkout --quiet --detach FETCH_HEAD"
cd "$dir"

run "building the miner runtime (Python 3.13)" "miner runtime ready" sh -c \
  'uv venv --allow-existing --python 3.13 .venv-kev &&
   uv pip install --python .venv-kev/bin/python -r requirements/model.txt \
     -r requirements/rehearsal.txt -r requirements/testnet.txt -e .'

# WSL keeps nvidia-smi in /usr/lib/wsl/lib, which is often not on PATH.
if [ "$(uname -s)-$(uname -m)" = Linux-x86_64 ] &&
  { command -v nvidia-smi >/dev/null || [ -x /usr/lib/wsl/lib/nvidia-smi ]; }; then
  run "building the d3 training runtime (Python 3.12, CUDA 13.0)" "d3 training runtime ready" sh -c \
    'uv venv --allow-existing --python 3.12 .venv-d3 &&
     uv pip install --python .venv-d3/bin/python --torch-backend=cu130 -r requirements/d3.txt'
else
  warn "skipped the d3 training runtime: it needs Linux x86_64 with an NVIDIA GPU"
fi

mkdir -p "$bin"
# btcli (from bittensor) creates and registers the wallet that doctor asks for.
# This script always comes from main, but the release it checks out may predate zils-miner;
# such a release still installs the miner as `zils`.
miner=zils-miner
[ -e "$dir/.venv-kev/bin/zils-miner" ] || miner=zils
for tool in "$miner" btcli; do ln -sf "$dir/.venv-kev/bin/$tool" "$bin/$tool"; done
ok "$miner and btcli commands ready in ~/.local/bin"
# Installs before zils-miner linked the miner as `zils`, which hid the customer CLI; drop only that link.
if [ "$miner" = zils-miner ] && [ -L "$bin/zils" ] &&
  [ "$(readlink "$bin/zils")" = "$dir/.venv-kev/bin/zils" ]; then
  rm "$bin/zils"
  ok "removed the old zils link; the miner command is now zils-miner"
fi
path_hint=""
wire_path

next="$miner setup"
[ -z "${ZILS_SKIP_SETUP:-}" ] || next="$miner doctor"
printf '\n%sZils miner installed.%s Next: %s%s%s\n' "$green$bold" "$reset" "$bold" "$next" "$reset"
[ -z "$path_hint" ] || printf '%sReload your shell first: %s%s\n' "$cyan" "$path_hint" "$reset"
if [ -z "${ZILS_SKIP_SETUP:-}" ] && [ -t 1 ] && [ -r /dev/tty ]; then
  printf '\n'
  "$bin/$miner" setup </dev/tty || true
fi
