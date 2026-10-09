#!/usr/bin/env bash
# Product-independent DAgger entry; the selected dataset supplies the 16D contract.
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export R1A7_PI05_PYTHON="${R1A7_PI05_PYTHON:-$ROOT_DIR/runtime/envs/pi05-262d2fd/bin/python}"
export R1A7_PI05_SOURCE="${R1A7_PI05_SOURCE:-$ROOT_DIR/runtime/lerobot-a100-262d2fd/src}"
exec "$ROOT_DIR/scripts/lerobot/run-r1a7-pi05-real.sh" \
  --dex1-profile matched-teleop \
  --max-commands 0 \
  --dagger-record-dir "$ROOT_DIR/runtime/datasets/r1a7_dagger_raw" \
  "$@"
