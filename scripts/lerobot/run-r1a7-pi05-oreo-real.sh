#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly PYTHON_BIN="$ROOT_DIR/runtime/envs/pi05-262d2fd/bin/python"
readonly LEROBOT_SOURCE="$ROOT_DIR/runtime/lerobot-a100-262d2fd/src"
readonly REAL_LAUNCHER="$ROOT_DIR/scripts/lerobot/run-r1a7-pi05-real.sh"
readonly TASK_BEFORE='Pick up a product from the blue tray on the left, present it to the scanner in the scanning area, and wait until the scan is accepted.'
readonly TASK_AFTER='Place the scanned product in the black tray on the right.'

checkpoint_root="$ROOT_DIR/runtime/checkpoints/pi05_oreo_a100_b24_6ep_02/checkpoints"
checkpoint_step=4k
checkpoint_override=""
checkpoint_step_set=0
dataset_root="/home/user/.cache/huggingface/lerobot/local/r1a7_oreo_4cam_16d_v1_80_pi05_stats"
barcode=6901668005939
preflight_only=0
forwarded=()

while (($#)); do
  case "$1" in
    --preflight)
      preflight_only=1
      shift
      ;;
    --checkpoint-step)
      (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      case "$2" in
        4k|6k|8k) checkpoint_step="$2" ;;
        *) printf 'Invalid checkpoint step: %s (choose 4k, 6k or 8k)\n' "$2" >&2; exit 2 ;;
      esac
      checkpoint_step_set=1
      shift 2
      ;;
    --checkpoint|--dataset-root|--expected-barcode)
      (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --checkpoint) checkpoint_override="$2" ;;
        --dataset-root) dataset_root="$2" ;;
        --expected-barcode) barcode="$2" ;;
      esac
      shift 2
      ;;
    --tokenizer-root|--task-before|--task-after|--task-event-bind)
      printf '%s cannot be overridden here: tokenizer, task text and local scanner endpoint are fixed for this test.\n' "$1" >&2
      exit 2
      ;;
    --help|-h)
      printf 'Oreo PI0.5 real-robot launcher. --preflight checks inputs without connecting to the robot.\n'
      printf 'Defaults: 4K checkpoint, Oreo dataset/barcode, 30 policy commands.\n'
      printf 'Choose: --checkpoint-step 4k|6k|8k. Advanced override: --checkpoint PATH.\n'
      printf 'Other overrides: --dataset-root PATH --expected-barcode CODE; real-launcher options are forwarded.\n\n'
      exec "$REAL_LAUNCHER" --help
      ;;
    *)
      forwarded+=("$1")
      shift
      ;;
  esac
done

if ((checkpoint_step_set)) && [[ -n "$checkpoint_override" ]]; then
  printf 'Use either --checkpoint-step or --checkpoint, not both.\n' >&2
  exit 2
fi
if [[ -n "$checkpoint_override" ]]; then
  checkpoint="$checkpoint_override"
  checkpoint_step=custom
else
  case "$checkpoint_step" in
    4k) checkpoint_dir=004000 ;;
    6k) checkpoint_dir=006000 ;;
    8k) checkpoint_dir=008000 ;;
  esac
  checkpoint="$checkpoint_root/$checkpoint_dir/pretrained_model"
fi

[[ -x "$PYTHON_BIN" && -d "$LEROBOT_SOURCE/lerobot" ]] || {
  printf 'Missing pinned PI0.5 Python environment or LeRobot source.\n' >&2
  exit 1
}
[[ -f "$checkpoint/model.safetensors" && -f "$checkpoint/tokenizer/tokenizer.json" ]] || {
  printf 'Checkpoint weights or local tokenizer missing: %s\n' "$checkpoint" >&2
  exit 1
}
[[ -f "$dataset_root/meta/info.json" ]] || {
  printf 'LeRobot dataset missing: %s\n' "$dataset_root" >&2
  exit 1
}

"$PYTHON_BIN" - "$checkpoint" "$dataset_root" "$TASK_BEFORE" "$TASK_AFTER" <<'PY'
import json
import socket
import sys
from pathlib import Path

import logging_mp  # noqa: F401 - needed by the camera and scanner clients
import zmq  # noqa: F401 - needed by the live control transport
from lerobot.policies import make_policy  # noqa: F401 - registers PI0.5 config
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

checkpoint, dataset_root = map(Path, sys.argv[1:3])
task_before, task_after = sys.argv[3:]
config = PreTrainedConfig.from_pretrained(checkpoint)
if config.type != "pi05":
    raise SystemExit(f"Expected PI0.5, got {config.type}")
if tuple(config.input_features["observation.state"].shape) != (16,):
    raise SystemExit("Checkpoint state is not 16D")
if tuple(config.output_features["action"].shape) != (16,):
    raise SystemExit("Checkpoint action is not 16D")
dataset = LeRobotDataset("local/" + dataset_root.name, root=dataset_root, video_backend="pyav")
tasks = set(dataset.meta.tasks.index)
if task_before not in tasks or task_after not in tasks:
    raise SystemExit("Scan/place task text differs from the dataset")
camera_keys = set(dataset.meta.camera_keys)
checkpoint_cameras = {key for key in config.input_features if key.startswith("observation.images.")}
if len(camera_keys) != 4 or camera_keys != checkpoint_cameras:
    raise SystemExit("Four-camera checkpoint/dataset contract mismatch")
for port in (55580, 55582):
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SystemExit(f"Required local port {port} is unavailable: {exc}") from exc
print(json.dumps({"event": "preflight_ok", "checkpoint": str(checkpoint),
                  "episodes": len(dataset.meta.episodes), "state_dim": 16,
                  "action_dim": 16, "hardware_commands_sent": False}))
PY

if ((preflight_only)); then
  exit 0
fi

export R1A7_PI05_PYTHON="$PYTHON_BIN"
export R1A7_PI05_SOURCE="$LEROBOT_SOURCE"
exec "$REAL_LAUNCHER" \
  --checkpoint "$checkpoint" \
  --dataset-root "$dataset_root" \
  --tokenizer-root "$checkpoint/tokenizer" \
  --expected-barcode "$barcode" \
  --task-event-bind tcp://127.0.0.1:55580 \
  --max-commands 30 \
  --output "$ROOT_DIR/runtime/logs/pi05_oreo_real_${checkpoint_step}_$(date +%Y%m%d_%H%M%S).jsonl" \
  "${forwarded[@]}"
