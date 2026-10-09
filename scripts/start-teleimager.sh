#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly TELEIMAGER_DIR="$ROOT_DIR/robot/xr-teleoperate/teleop/teleimager"
export R1A7_ROOT="$ROOT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/runtime/cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ -z "${TV_PYTHON:-}" ]]; then
  TV_PYTHON="$(command -v python || command -v python3)"
fi

cd "$TELEIMAGER_DIR"
export R1A7_CAMERA_CONTROLS_ON_READY="$ROOT_DIR/scripts/cameras/apply-r1a7-camera-controls.sh"
exec "$TV_PYTHON" -m teleimager.image_server "$@"
