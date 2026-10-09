#!/usr/bin/env bash
set -Eeuo pipefail

report_error() {
  local status=$?
  printf 'PI0.5 live shadow launcher failed at line %s (exit %s).\n' "${BASH_LINENO[0]}" "$status" >&2
  return "$status"
}
trap report_error ERR

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly CONDA_ROOT="${CONDA_ROOT:-/home/user/miniconda3}"
readonly TV_PYTHON="$CONDA_ROOT/envs/tv_official/bin/python"
readonly LEROBOT_PYTHON="$CONDA_ROOT/envs/lerobot/bin/python"
readonly PACKAGE_ROOT="$ROOT_DIR/data/unitree-lerobot"
readonly LEROBOT_SOURCE="$PACKAGE_ROOT/unitree_lerobot/lerobot/src"
readonly XR_ROOT="$ROOT_DIR/robot/xr-teleoperate"
readonly TELEOP_ROOT="$XR_ROOT/teleop"
readonly TELEIMAGER_SOURCE="$TELEOP_ROOT/teleimager/src"
readonly SDK_ROOT="$ROOT_DIR/vendor/unitree-sdk2-python"

usage() {
  cat <<'EOF'
Usage: run-r1a7-pi05-live-shadow.sh [options]

Options:
  --network-interface NAME  DDS interface (default: enx00e04c187010).
  --image-host HOST         Four-camera image server (default: 127.0.0.1).
  --dataset-root PATH       LeRobot v3 dataset used for metadata/stats.
  --checkpoint PATH         PI0.5 legacy pretrained_model directory.
  --tokenizer-root PATH     Local PaliGemma tokenizer snapshot.
  --expected-barcode CODE   Optional single-product guard for this run.
  --task-before TEXT        Scan-stage language instruction.
  --task-after TEXT         Place-stage language instruction.
  --output PATH             Prediction JSONL path.
  --max-steps N             Stop after N predictions; 0 runs until Ctrl-C.
  --device DEVICE           Torch device (default: cuda).
  -h, --help                Show this help.

Prerequisites: four-camera image server and Dex1 service must already be
running. Start the barcode bridge only after this command prints its JSON READY
line. This is shadow inference: it never publishes robot commands.
Dataset and checkpoint must use the 16D dual-arm plus dual-gripper contract.
EOF
}

DATASET_ROOT="${R1A7_PI05_SOURCE_DATASET:-/home/user/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30}"
CHECKPOINT="${R1A7_PI05_LEGACY_CHECKPOINT:-$ROOT_DIR/data/unitree-lerobot/unitree_lerobot/lerobot/outputs/train/r1a7_pi05_single_gpu_bs32_20260826-010706/checkpoints/008000/pretrained_model_legacy}"
TOKENIZER_ROOT="${R1A7_PI05_TOKENIZER:-}"
NETWORK_INTERFACE=enx00e04c187010
IMAGE_HOST=127.0.0.1
EXPECTED_BARCODE="${R1A7_EXPECTED_BARCODE:-}"
TASK_BEFORE="Pick up a product from the blue tray on the left, present it to the scanner in the scanning area, and wait until the scan is accepted."
TASK_AFTER="Place the scanned product in the black tray on the right."
OUTPUT="$ROOT_DIR/runtime/logs/pi05_live_shadow_$(date +%Y%m%d_%H%M%S).jsonl"
MAX_STEPS=0
DEVICE=cuda

while (($#)); do
  case "$1" in
    --network-interface|--image-host|--dataset-root|--checkpoint|--tokenizer-root|--expected-barcode|--task-before|--task-after|--output|--max-steps|--device)
      (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --network-interface) NETWORK_INTERFACE="$2" ;;
        --image-host) IMAGE_HOST="$2" ;;
        --dataset-root) DATASET_ROOT="$2" ;;
        --checkpoint) CHECKPOINT="$2" ;;
        --tokenizer-root) TOKENIZER_ROOT="$2" ;;
        --expected-barcode) EXPECTED_BARCODE="$2" ;;
        --task-before) TASK_BEFORE="$2" ;;
        --task-after) TASK_AFTER="$2" ;;
        --output) OUTPUT="$2" ;;
        --max-steps) MAX_STEPS="$2" ;;
        --device) DEVICE="$2" ;;
      esac
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -x "$TV_PYTHON" && -x "$LEROBOT_PYTHON" ]] || {
  printf 'Missing tv_official or lerobot Python environment.\n' >&2
  exit 1
}
[[ -d "$DATASET_ROOT" && -d "$CHECKPOINT" ]] || {
  printf 'Dataset or checkpoint directory is missing.\n' >&2
  exit 1
}
[[ "$MAX_STEPS" =~ ^[0-9]+$ ]] || {
  printf 'max-steps must be a non-negative integer.\n' >&2
  exit 2
}
if [[ -z "$TOKENIZER_ROOT" ]]; then
  readonly HF_ROOT="/home/user/.cache/huggingface/hub/models--google--paligemma-3b-pt-224"
  [[ -f "$HF_ROOT/refs/main" ]] || { printf 'Missing tokenizer cache.\n' >&2; exit 1; }
  IFS= read -r HF_REVISION < "$HF_ROOT/refs/main" || [[ -n "$HF_REVISION" ]]
  [[ "$HF_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
    printf 'Invalid Hugging Face revision in %s: %s\n' "$HF_ROOT/refs/main" "$HF_REVISION" >&2
    exit 1
  }
  TOKENIZER_ROOT="$HF_ROOT/snapshots/$HF_REVISION"
fi
[[ -d "$TOKENIZER_ROOT" ]] || { printf 'Missing tokenizer: %s\n' "$TOKENIZER_ROOT" >&2; exit 1; }

STATE_BRIDGE_PID=""
cleanup() {
  if [[ -n "$STATE_BRIDGE_PID" ]] && kill -0 "$STATE_BRIDGE_PID" 2>/dev/null; then
    kill -TERM "$STATE_BRIDGE_PID" 2>/dev/null || true
    wait "$STATE_BRIDGE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SDK_ROOT:$TELEOP_ROOT" \
  "$TV_PYTHON" "$TELEOP_ROOT/tools/r1a7_readonly_state_bridge.py" \
  --network-interface "$NETWORK_INTERFACE" \
  --bind tcp://127.0.0.1:55582 &
STATE_BRIDGE_PID=$!

sleep 1
kill -0 "$STATE_BRIDGE_PID" 2>/dev/null || {
  printf 'Read-only state bridge failed during startup.\n' >&2
  wait "$STATE_BRIDGE_PID"
}

RUNNER_ARGS=(
  --dataset-root "$DATASET_ROOT"
  --policy-path "$CHECKPOINT"
  --tokenizer-path "$TOKENIZER_ROOT"
  --image-host "$IMAGE_HOST"
  --state-endpoint tcp://127.0.0.1:55582
  --task-event-bind tcp://0.0.0.0:55580
  --task-before "$TASK_BEFORE"
  --task-after "$TASK_AFTER"
  --device "$DEVICE"
  --max-steps "$MAX_STEPS"
  --output "$OUTPUT"
)
if [[ -n "$EXPECTED_BARCODE" ]]; then
  RUNNER_ARGS+=(--expected-barcode "$EXPECTED_BARCODE")
fi

env \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PACKAGE_ROOT:$LEROBOT_SOURCE:$XR_ROOT:$TELEIMAGER_SOURCE" \
  "$LEROBOT_PYTHON" -m unitree_lerobot.r1a7_live_shadow "${RUNNER_ARGS[@]}"
