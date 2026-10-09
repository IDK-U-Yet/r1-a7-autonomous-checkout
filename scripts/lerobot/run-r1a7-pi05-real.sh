#!/usr/bin/env bash
set -Eeuo pipefail

report_error() {
  local status=$?
  printf 'PI0.5 real-robot launcher failed at line %s (exit %s).\n' "${BASH_LINENO[0]}" "$status" >&2
  return "$status"
}
trap report_error ERR

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly CONDA_ROOT="${CONDA_ROOT:-/home/user/miniconda3}"
readonly TV_PYTHON="$CONDA_ROOT/envs/tv_official/bin/python"
readonly LEROBOT_PYTHON="${R1A7_PI05_PYTHON:-$CONDA_ROOT/envs/lerobot/bin/python}"
readonly PACKAGE_ROOT="$ROOT_DIR/data/unitree-lerobot"
readonly LEROBOT_SOURCE="${R1A7_PI05_SOURCE:-$PACKAGE_ROOT/unitree_lerobot/lerobot/src}"
readonly XR_ROOT="$ROOT_DIR/robot/xr-teleoperate"
readonly TELEOP_ROOT="$XR_ROOT/teleop"
readonly TELEIMAGER_SOURCE="$TELEOP_ROOT/teleimager/src"
readonly SDK_ROOT="$ROOT_DIR/vendor/unitree-sdk2-python"
readonly CONFIRMATION=I_UNDERSTAND_R1A7_COMMANDS_MOVE_REAL_HARDWARE

usage() {
  cat <<'EOF'
Usage: run-r1a7-pi05-real.sh [options]

Required:
  export R1A7_REAL_INFERENCE=YES
  --confirm-real-robot I_UNDERSTAND_R1A7_COMMANDS_MOVE_REAL_HARDWARE

Options:
  --network-interface NAME  DDS interface (default: enx00e04c187010).
  --image-host HOST         Four-camera image server (default: 127.0.0.1).
  --task-event-bind ENDPOINT Scanner task-event listener (default: tcp://0.0.0.0:55580).
  --dataset-root PATH       LeRobot v3 dataset used for metadata/stats.
  --checkpoint PATH         PI0.5 legacy pretrained_model directory.
  --tokenizer-root PATH     Local PaliGemma tokenizer snapshot.
  --expected-barcode CODE   Optional single-product guard for this run.
  --task-before TEXT        Scan-stage language instruction.
  --task-after TEXT         Place-stage language instruction.
  --output PATH             Real-command audit JSONL path.
  --diagnostic-frames-dir PATH
                            Save first model input and first closing-command images.
  --dagger-record-dir PATH   Enable multi-trial PICO takeover and four-camera audit recording.
  --collection-name NAME    Optional DAgger batch label; defaults to dataset-root folder name.
  --takeover-blend-seconds SEC  DAgger offset fade minimum duration (default 3; 0 = fixed-relative).
  --max-commands N          Stop after N commands; 0 runs until Ctrl-C (default: 1).
  --manual-cycles           Enable P/N/R/D operator controls; requires N=0 and a terminal.
  --completion-wait-seconds SEC
                            Wait after operator confirms placement with D (default: 0, disabled).
  --chunk-blend-steps N     Blend N commands at action-chunk boundaries (default: 3).
  --async-prefetch-actions N
                            Start the next inference with N actions left (default: 5).
  --initialization-duration SEC
                            Move to the dataset scan start pose over SEC (default: 3).
  --initialization-timeout SEC
                            Additional convergence wait (default: 12).
  --max-arm-limit-overshoot RAD
                            Largest arm overshoot clipped to a hard limit (default: 0.05).
  --command-timeout SEC     Latching policy-stream timeout (default: 0.5).
  --device DEVICE           Torch device (default: cuda).
  --dex1-profile PROFILE    legacy or matched-teleop (default: legacy).
  --confirm-real-robot TEXT Exact physical-motion confirmation.
  -h, --help                Show this help.

The arm profile uses official collection limits. Dex1 defaults to the previous
inference settings (kp=2.0 kd=0.05 step=0.18); matched-teleop uses the Oreo
collection settings (kp=3.0 kd=0.05 step=0.24). The policy contract is
16D (dual arms plus dual grippers). Waist yaw is not a model input or action;
deployment holds the activation waist pose in the hardware bridge. Hardware
hard position limits remain active. Chunk blending is not a velocity limit. A
command-stream timeout latches HOLD and requires restart. Old 17D checkpoints
are incompatible and are rejected by the model/dataset contract checks.
EOF
}

DATASET_ROOT="${R1A7_PI05_SOURCE_DATASET:-/home/user/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30}"
CHECKPOINT="${R1A7_PI05_LEGACY_CHECKPOINT:-$ROOT_DIR/data/unitree-lerobot/unitree_lerobot/lerobot/outputs/train/r1a7_pi05_single_gpu_bs32_20260826-010706/checkpoints/008000/pretrained_model_legacy}"
TOKENIZER_ROOT="${R1A7_PI05_TOKENIZER:-}"
NETWORK_INTERFACE=enx00e04c187010
IMAGE_HOST=127.0.0.1
TASK_EVENT_BIND=tcp://0.0.0.0:55580
EXPECTED_BARCODE="${R1A7_EXPECTED_BARCODE:-}"
TASK_BEFORE="Pick up a product from the blue tray on the left, present it to the scanner in the scanning area, and wait until the scan is accepted."
TASK_AFTER="Place the scanned product in the black tray on the right."
OUTPUT="$ROOT_DIR/runtime/logs/pi05_real_$(date +%Y%m%d_%H%M%S).jsonl"
MAX_COMMANDS=1
CHUNK_BLEND_STEPS=3
ASYNC_PREFETCH_ACTIONS=5
INITIALIZATION_DURATION=3.0
INITIALIZATION_TIMEOUT=12.0
MAX_ARM_LIMIT_OVERSHOOT=0.05
COMMAND_TIMEOUT=0.5
DEX1_PROFILE=legacy
MANUAL_CYCLES=0
COMPLETION_WAIT_SECONDS=0
DEVICE=cuda
DIAGNOSTIC_FRAMES_DIR=""
DAGGER_RECORD_DIR=""
COLLECTION_NAME=""
TAKEOVER_BLEND_SECONDS=3
USER_CONFIRMATION=""

while (($#)); do
  case "$1" in
    --manual-cycles)
      MANUAL_CYCLES=1
      shift
      ;;
    --network-interface|--image-host|--task-event-bind|--dataset-root|--checkpoint|--tokenizer-root|--expected-barcode|--task-before|--task-after|--output|--diagnostic-frames-dir|--dagger-record-dir|--collection-name|--takeover-blend-seconds|--max-commands|--completion-wait-seconds|--chunk-blend-steps|--async-prefetch-actions|--initialization-duration|--initialization-timeout|--max-arm-limit-overshoot|--command-timeout|--device|--dex1-profile|--confirm-real-robot)
      (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --network-interface) NETWORK_INTERFACE="$2" ;;
        --image-host) IMAGE_HOST="$2" ;;
        --task-event-bind) TASK_EVENT_BIND="$2" ;;
        --dataset-root) DATASET_ROOT="$2" ;;
        --checkpoint) CHECKPOINT="$2" ;;
        --tokenizer-root) TOKENIZER_ROOT="$2" ;;
        --expected-barcode) EXPECTED_BARCODE="$2" ;;
        --task-before) TASK_BEFORE="$2" ;;
        --task-after) TASK_AFTER="$2" ;;
        --output) OUTPUT="$2" ;;
        --diagnostic-frames-dir) DIAGNOSTIC_FRAMES_DIR="$2" ;;
        --dagger-record-dir) DAGGER_RECORD_DIR="$2" ;;
        --collection-name) COLLECTION_NAME="$2" ;;
        --takeover-blend-seconds) TAKEOVER_BLEND_SECONDS="$2" ;;
        --max-commands) MAX_COMMANDS="$2" ;;
        --completion-wait-seconds) COMPLETION_WAIT_SECONDS="$2" ;;
        --chunk-blend-steps) CHUNK_BLEND_STEPS="$2" ;;
        --async-prefetch-actions) ASYNC_PREFETCH_ACTIONS="$2" ;;
        --initialization-duration) INITIALIZATION_DURATION="$2" ;;
        --initialization-timeout) INITIALIZATION_TIMEOUT="$2" ;;
        --max-arm-limit-overshoot) MAX_ARM_LIMIT_OVERSHOOT="$2" ;;
        --command-timeout) COMMAND_TIMEOUT="$2" ;;
        --dex1-profile) DEX1_PROFILE="$2" ;;
        --device) DEVICE="$2" ;;
        --confirm-real-robot) USER_CONFIRMATION="$2" ;;
      esac
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ "${R1A7_REAL_INFERENCE:-}" == YES ]] || {
  printf 'Refusing real inference: export R1A7_REAL_INFERENCE=YES first.\n' >&2
  exit 2
}
[[ "$USER_CONFIRMATION" == "$CONFIRMATION" ]] || {
  printf 'Refusing real inference: the exact --confirm-real-robot value is required.\n' >&2
  exit 2
}
[[ "$MAX_COMMANDS" =~ ^[0-9]+$ ]] || {
  printf 'max-commands must be a non-negative integer.\n' >&2
  exit 2
}
[[ "$CHUNK_BLEND_STEPS" =~ ^[0-9]+$ ]] || {
  printf 'chunk-blend-steps must be a non-negative integer.\n' >&2
  exit 2
}
[[ "$ASYNC_PREFETCH_ACTIONS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'async-prefetch-actions must be a positive integer.\n' >&2
  exit 2
}
[[ "$DEX1_PROFILE" == legacy || "$DEX1_PROFILE" == matched-teleop ]] || {
  printf 'dex1-profile must be legacy or matched-teleop.\n' >&2
  exit 2
}
[[ "$MANUAL_CYCLES" == 0 || "$MAX_COMMANDS" == 0 ]] || {
  printf 'manual-cycles requires --max-commands 0.\n' >&2
  exit 2
}
[[ -x "$TV_PYTHON" && -x "$LEROBOT_PYTHON" ]] || {
  printf 'Missing tv_official or lerobot Python environment.\n' >&2
  exit 1
}
[[ -d "$DATASET_ROOT" && -d "$CHECKPOINT" ]] || {
  printf 'Dataset or checkpoint directory is missing.\n' >&2
  exit 1
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

COMMAND_BRIDGE_PID=""
cleanup() {
  if [[ -n "$COMMAND_BRIDGE_PID" ]] && kill -0 "$COMMAND_BRIDGE_PID" 2>/dev/null; then
    kill -TERM "$COMMAND_BRIDGE_PID" 2>/dev/null || true
    for ((attempt = 0; attempt < 30; attempt++)); do
      kill -0 "$COMMAND_BRIDGE_PID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$COMMAND_BRIDGE_PID" 2>/dev/null; then
      printf 'Real command bridge did not stop after SIGTERM; stopping its PID %s.\n' "$COMMAND_BRIDGE_PID" >&2
      kill -KILL "$COMMAND_BRIDGE_PID" 2>/dev/null || true
    fi
    wait "$COMMAND_BRIDGE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

printf 'REAL ROBOT: official collection limits; max commands=%s; timeout=%ss.\n' "$MAX_COMMANDS" "$COMMAND_TIMEOUT"
RUNNER_ARGS=(
  --dataset-root "$DATASET_ROOT"
  --policy-path "$CHECKPOINT"
  --tokenizer-path "$TOKENIZER_ROOT"
  --image-host "$IMAGE_HOST"
  --bridge-endpoint tcp://127.0.0.1:55582
  --task-event-bind "$TASK_EVENT_BIND"
  --task-before "$TASK_BEFORE"
  --task-after "$TASK_AFTER"
  --device "$DEVICE"
  --max-commands "$MAX_COMMANDS"
  --completion-wait-seconds "$COMPLETION_WAIT_SECONDS"
  --chunk-blend-steps "$CHUNK_BLEND_STEPS"
  --async-prefetch-actions "$ASYNC_PREFETCH_ACTIONS"
  --initialization-duration "$INITIALIZATION_DURATION"
  --initialization-timeout "$INITIALIZATION_TIMEOUT"
  --max-arm-limit-overshoot "$MAX_ARM_LIMIT_OVERSHOOT"
  --output "$OUTPUT"
  --confirm-real-robot "$USER_CONFIRMATION"
)
if [[ -n "$EXPECTED_BARCODE" ]]; then
  RUNNER_ARGS+=(--expected-barcode "$EXPECTED_BARCODE")
fi
if [[ -n "$DIAGNOSTIC_FRAMES_DIR" ]]; then
  RUNNER_ARGS+=(--diagnostic-frames-dir "$DIAGNOSTIC_FRAMES_DIR")
fi
if [[ "$MANUAL_CYCLES" == 1 ]]; then
  RUNNER_ARGS+=(--manual-cycles)
fi
BRIDGE_ARGS=()
RUNNER_MODULE=unitree_lerobot.r1a7_live_real
if [[ -n "$DAGGER_RECORD_DIR" ]]; then
  [[ "$MANUAL_CYCLES" == 0 && "$COMPLETION_WAIT_SECONDS" == 0 ]] || {
    printf 'DAgger uses S/Q completion, not manual or automatic reset.\n' >&2; exit 2;
  }
  if pgrep -f '[/]teleop_hand_and_arm.py|[/]r1a7_real_command_bridge.py|[/]r1a7_live_real.py|unitree_lerobot.r1a7_live_real|unitree_lerobot.r1a7_live_dagger' >/dev/null; then
    printf 'Another teleop/inference publisher is running; stop it before DAgger.\n' >&2
    exit 2
  fi
  BRIDGE_ARGS+=(--dagger --takeover-blend-seconds "$TAKEOVER_BLEND_SECONDS")
  RUNNER_ARGS+=(--dagger-record-dir "$DAGGER_RECORD_DIR" --takeover-blend-seconds "$TAKEOVER_BLEND_SECONDS")
  if [[ -n "$COLLECTION_NAME" ]]; then
    RUNNER_ARGS+=(--collection-name "$COLLECTION_NAME")
  fi
  RUNNER_MODULE=unitree_lerobot.r1a7_live_dagger
  env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$PACKAGE_ROOT:$LEROBOT_SOURCE:$XR_ROOT:$TELEIMAGER_SOURCE" \
    "$LEROBOT_PYTHON" -m unitree_lerobot.r1a7_dagger_preflight "${RUNNER_ARGS[@]}"
fi
env \
  R1A7_REAL_INFERENCE=YES \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$SDK_ROOT:$PACKAGE_ROOT:$XR_ROOT:$TELEOP_ROOT/televuer/src" \
  "$TV_PYTHON" "$TELEOP_ROOT/tools/r1a7_real_command_bridge.py" \
  --xr-teleoperate-root "$XR_ROOT" \
  --network-interface "$NETWORK_INTERFACE" \
  --bind tcp://127.0.0.1:55582 \
  --command-timeout "$COMMAND_TIMEOUT" \
  --dex1-profile "$DEX1_PROFILE" \
  "${BRIDGE_ARGS[@]}" \
  --confirm-real-robot "$USER_CONFIRMATION" &
COMMAND_BRIDGE_PID=$!

"$TV_PYTHON" - "$COMMAND_BRIDGE_PID" <<'PY'
import os
import socket
import sys
import time

bridge_pid = int(sys.argv[1])
deadline = time.monotonic() + 30.0
while time.monotonic() < deadline:
    try:
        os.kill(bridge_pid, 0)
    except ProcessLookupError:
        raise SystemExit("Real command bridge exited during startup")
    try:
        with socket.create_connection(("127.0.0.1", 55582), timeout=0.2):
            print("Real command bridge is listening on 127.0.0.1:55582", flush=True)
            break
    except OSError:
        time.sleep(0.2)
else:
    raise SystemExit("Real command bridge did not listen on 127.0.0.1:55582 within 30s")
PY

env \
  R1A7_REAL_INFERENCE=YES \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PACKAGE_ROOT:$LEROBOT_SOURCE:$XR_ROOT:$TELEIMAGER_SOURCE" \
  "$LEROBOT_PYTHON" -m "$RUNNER_MODULE" "${RUNNER_ARGS[@]}"
