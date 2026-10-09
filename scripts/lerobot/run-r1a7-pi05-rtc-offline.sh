#!/usr/bin/env bash
set -euo pipefail
R1A7_ROOT="${R1A7_ROOT:-/home/user/r1-a7-autonomous-checkout}"
export PYTHONPATH="$R1A7_ROOT/data/unitree-lerobot:$R1A7_ROOT/runtime/lerobot-a100-262d2fd/src:$R1A7_ROOT/robot/xr-teleoperate:$R1A7_ROOT/robot/xr-teleoperate/teleop/teleimager/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "${LEROBOT_PYTHON:-$R1A7_ROOT/runtime/envs/pi05-262d2fd/bin/python}" -m unitree_lerobot.r1a7_rtc_offline "$@"
