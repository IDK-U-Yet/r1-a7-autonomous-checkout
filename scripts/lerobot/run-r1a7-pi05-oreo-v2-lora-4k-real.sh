#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON_BIN="$ROOT_DIR/runtime/envs/pi05-262d2fd/bin/python"
LEROBOT_SOURCE="$ROOT_DIR/runtime/lerobot-a100-262d2fd/src"
REAL_LAUNCHER="$ROOT_DIR/scripts/lerobot/run-r1a7-pi05-real.sh"
CHECKPOINT_ROOT="$ROOT_DIR/runtime/checkpoints/pi05_oreo_v2_lora_localized"
BASE_MODEL="$ROOT_DIR/runtime/checkpoints/pi05_base"
DATASET_ROOT="$ROOT_DIR/runtime/datasets/lerobot/r1a7_oreo_4cam_16d_v2_80_standard_train"
TASK_BEFORE='Bring the product from its current location to the scanner in the scanning area, present its barcode, and wait until the scan is accepted.'
TASK_AFTER='Place the scanned product in the black tray on the right.'
CONFIRMATION=I_UNDERSTAND_R1A7_COMMANDS_MOVE_REAL_HARDWARE

usage() {
  cat <<'EOF'
Usage: run-r1a7-pi05-oreo-v2-lora-4k-real.sh --checkpoint-step 4k|6k|8k [options]

Options:
  --preflight             Check the selected LoRA, local base model and dataset without moving the robot.
  --checkpoint-step STEP  Select 4k, 6k or 8k (default: 4k).
  --max-commands N        Stop after N policy commands; 0 runs until Ctrl-C (default: 30).
  --network-interface IF  Robot DDS interface (default: enx00e04c187010).
  --image-host HOST       Four-camera image server (default: 127.0.0.1).
  --confirm-real-robot I_UNDERSTAND_R1A7_COMMANDS_MOVE_REAL_HARDWARE

Real motion requires R1A7_REAL_INFERENCE=YES and the exact confirmation.
Dex1 matches collection: kp=3.0, kd=0.05, command_step=0.24.
With --max-commands 0, P + Enter holds; R + Enter discards queued actions,
returns to the scan start pose and begins a fresh cycle. After confirming
placement with D + Enter, the robot waits 20 seconds before returning.
Keep both grippers empty and the return path clear before R or D.
EOF
}

max_commands=30
checkpoint_step=4k
network_interface=enx00e04c187010
image_host=127.0.0.1
user_confirmation=""
preflight_only=0

while (($#)); do
  case "$1" in
    --preflight) preflight_only=1; shift ;;
    --checkpoint-step|--max-commands|--network-interface|--image-host|--confirm-real-robot)
      (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --checkpoint-step) checkpoint_step="$2" ;;
        --max-commands) max_commands="$2" ;;
        --network-interface) network_interface="$2" ;;
        --image-host) image_host="$2" ;;
        --confirm-real-robot) user_confirmation="$2" ;;
      esac
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

case "$checkpoint_step" in
  4k) checkpoint_dir=004000 ;;
  6k) checkpoint_dir=006000 ;;
  8k) checkpoint_dir=008000 ;;
  *) printf 'Choose --checkpoint-step 4k, 6k or 8k.\n' >&2; exit 2 ;;
esac
CHECKPOINT="$CHECKPOINT_ROOT/$checkpoint_dir/pretrained_model"

[[ "$max_commands" =~ ^[0-9]+$ ]] || {
  printf 'max-commands must be a non-negative integer.\n' >&2
  exit 2
}
[[ -x "$PYTHON_BIN" && -d "$LEROBOT_SOURCE/lerobot" ]] || {
  printf 'Missing pinned PI0.5 environment or LeRobot source.\n' >&2
  exit 1
}
for file in \
  "$CHECKPOINT/config.json" \
  "$CHECKPOINT/adapter_config.json" \
  "$CHECKPOINT/adapter_model.safetensors" \
  "$CHECKPOINT/tokenizer/tokenizer.json" \
  "$BASE_MODEL/model.safetensors" \
  "$DATASET_ROOT/meta/info.json" \
  "$DATASET_ROOT/meta/tasks.parquet"; do
  [[ -f "$file" ]] || { printf 'Missing required file: %s\n' "$file" >&2; exit 1; }
done

if ((preflight_only)); then
  PYTHONPATH="$LEROBOT_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" - "$CHECKPOINT" "$BASE_MODEL" "$DATASET_ROOT" "$TASK_BEFORE" "$TASK_AFTER" <<'PY'
import json
import socket
import sys
from pathlib import Path

from lerobot.policies import make_policy  # noqa: F401 - register PI0.5 config
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

checkpoint, base_model, dataset_root = map(Path, sys.argv[1:4])
task_before, task_after = sys.argv[4:]
adapter = json.loads((checkpoint / "adapter_config.json").read_text())
if Path(adapter["base_model_name_or_path"]).resolve() != base_model.resolve():
    raise SystemExit("LoRA adapter does not point to the pinned local base model")
config = PreTrainedConfig.from_pretrained(checkpoint)
if config.type != "pi05" or not config.use_peft:
    raise SystemExit("Checkpoint is not a PI0.5 PEFT adapter")
if tuple(config.input_features["observation.state"].shape) != (16,):
    raise SystemExit("Checkpoint state is not 16D")
if tuple(config.output_features["action"].shape) != (16,):
    raise SystemExit("Checkpoint action is not 16D")
dataset = LeRobotDataset("local/" + dataset_root.name, root=dataset_root, video_backend="pyav")
tasks = set(dataset.meta.tasks.index)
if task_before not in tasks or task_after not in tasks:
    raise SystemExit("Scan/place task text differs from the dataset")
camera_keys = set(dataset.meta.camera_keys)
model_cameras = {key for key in config.input_features if key.startswith("observation.images.")}
if len(camera_keys) != 4 or camera_keys != model_cameras:
    raise SystemExit("Four-camera checkpoint/dataset contract mismatch")
for port in (55580, 55582):
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SystemExit(f"Required local port {port} is unavailable: {exc}") from exc
print(json.dumps({"event": "preflight_ok", "checkpoint": str(checkpoint),
                  "base_model": str(base_model), "episodes": len(dataset.meta.episodes),
                  "state_dim": 16, "action_dim": 16,
                  "dex1": {"kp": 3.0, "kd": 0.05, "command_step": 0.24},
                  "hardware_commands_sent": False}))
PY
  exit 0
fi

[[ "${R1A7_REAL_INFERENCE:-}" == YES ]] || {
  printf 'Set R1A7_REAL_INFERENCE=YES for real motion.\n' >&2
  exit 2
}
[[ "$user_confirmation" == "$CONFIRMATION" ]] || {
  printf 'The exact real-robot confirmation is required.\n' >&2
  exit 2
}

export R1A7_PI05_PYTHON="$PYTHON_BIN"
export R1A7_PI05_SOURCE="$LEROBOT_SOURCE"
printf 'Selected LoRA checkpoint: %s (%s)\n' "$checkpoint_step" "$CHECKPOINT"

manual_cycle_args=()
if [[ "$max_commands" == 0 ]]; then
  [[ -t 0 ]] || {
    printf 'Interactive P/R control requires a terminal when max-commands is 0.\n' >&2
    exit 2
  }
  manual_cycle_args+=(--manual-cycles --completion-wait-seconds 20)
fi

exec "$REAL_LAUNCHER" \
  --checkpoint "$CHECKPOINT" \
  --tokenizer-root "$CHECKPOINT/tokenizer" \
  --dataset-root "$DATASET_ROOT" \
  --network-interface "$network_interface" \
  --image-host "$image_host" \
  --task-event-bind tcp://127.0.0.1:55580 \
  --task-before "$TASK_BEFORE" \
  --task-after "$TASK_AFTER" \
  --expected-barcode 6901668005939 \
  --dex1-profile matched-teleop \
  --max-commands "$max_commands" \
  "${manual_cycle_args[@]}" \
  --output "$ROOT_DIR/runtime/logs/pi05_oreo_v2_lora_${checkpoint_step}_real_$(date +%Y%m%d_%H%M%S).jsonl" \
  --confirm-real-robot "$user_confirmation"
