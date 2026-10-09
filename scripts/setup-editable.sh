#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly MODE="${1:-}"
readonly PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"

usage() {
  printf 'Usage: %s {teleop|lerobot}\n' "$0" >&2
}

[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || {
  printf 'Activate the target Python environment first.\n' >&2
  exit 1
}

case "$MODE" in
  teleop)
    packages=(
      "$ROOT_DIR/vendor/unitree-sdk2-python"
      "$ROOT_DIR/robot/xr-teleoperate/teleop/teleimager"
      "$ROOT_DIR/robot/xr-teleoperate/teleop/televuer"
      "$ROOT_DIR/robot/xr-teleoperate/teleop/robot_control/dex-retargeting"
    )
    ;;
  lerobot)
    packages=(
      "$ROOT_DIR/data/unitree-lerobot/unitree_lerobot/lerobot"
      "$ROOT_DIR/data/unitree-lerobot"
    )
    ;;
  supermarket)
    printf 'The checkout application is private and is not included in this release.\n' >&2
    exit 2
    ;;
  *)
    usage
    exit 2
    ;;
esac

for package in "${packages[@]}"; do
  [[ -d "$package" ]] || {
    printf 'Missing package directory: %s\n' "$package" >&2
    exit 1
  }
  "$PYTHON_BIN" -m pip install \
    --disable-pip-version-check \
    --no-build-isolation \
    --no-deps \
    -e "$package"
done

printf 'Editable paths repaired for %s using %s\n' "$MODE" "$PYTHON_BIN"
