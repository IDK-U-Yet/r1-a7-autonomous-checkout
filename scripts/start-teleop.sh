#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly TELEOP_DIR="$ROOT_DIR/robot/xr-teleoperate/teleop"
export R1A7_ROOT="$ROOT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/runtime/cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ -z "${TV_PYTHON:-}" ]]; then
  TV_PYTHON="$(command -v python || command -v python3)"
fi

cd "$TELEOP_DIR"
exec "$TV_PYTHON" teleop_hand_and_arm.py "$@"
