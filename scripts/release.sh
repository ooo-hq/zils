#!/bin/sh
# Stable releases; see scripts/release.py or run with --help.
cd "$(dirname "$0")/.." && exec python3 -m scripts.release "$@"
