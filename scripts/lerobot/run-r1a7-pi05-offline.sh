#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly PYTHON_BIN="${R1A7_PI05_PYTHON:-$ROOT_DIR/runtime/envs/pi05-262d2fd/bin/python}"

[[ -x "$PYTHON_BIN" ]] || {
  printf 'Missing PI0.5 Python environment: %s\n' "$PYTHON_BIN" >&2
  exit 1
}

exec env \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" "$ROOT_DIR/scripts/lerobot/r1a7_pi05_offline_test.py" "$@"
