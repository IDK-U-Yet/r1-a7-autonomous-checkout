#!/usr/bin/env bash
# Analysis only: no model, camera or hardware initialization.
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export PYTHONPATH="$ROOT_DIR/data/unitree-lerobot${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -m unitree_lerobot.r1a7_rtc_compare "$@"
