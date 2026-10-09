#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly PACKAGE_ROOT="$ROOT_DIR/data/unitree-lerobot"
readonly LEROBOT_SOURCE="$ROOT_DIR/data/unitree-lerobot/unitree_lerobot/lerobot/src"

usage() {
  cat <<'EOF'
Usage: run-r1a7-pi05-local-shadow.sh [options]

Options:
  --dry-run                 Validate one dataset frame without loading PI0.5.
  --dataset-root PATH       Local LeRobot v3 dataset.
  --checkpoint PATH         Legacy pretrained_model directory.
  --tokenizer-root PATH     Local PaliGemma tokenizer snapshot.
  --episode N               Episode index (default: 0).
  --frame N                 Frame index inside the episode (default: 0).
  --task TEXT               Override the task stored in the dataset.
  --device DEVICE           Torch device for policy inference (default: cuda).
  --seed N                  Inference seed (default: 1000).
  -h, --help                Show this help.

This command only performs offline inference on a recorded dataset frame. It
does not import the R1-A7 hardware gateway and cannot publish robot commands.
The dataset and checkpoint must use the 16D dual-arm plus dual-gripper contract.
EOF
}

readonly CONDA_ROOT="${CONDA_ROOT:-/home/user/miniconda3}"
readonly PYTHON_BIN="$CONDA_ROOT/envs/lerobot/bin/python"

DATASET_ROOT="${R1A7_PI05_SOURCE_DATASET:-/home/user/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30}"
CHECKPOINT="${R1A7_PI05_LEGACY_CHECKPOINT:-$ROOT_DIR/data/unitree-lerobot/unitree_lerobot/lerobot/outputs/train/r1a7_pi05_single_gpu_bs32_20260826-010706/checkpoints/008000/pretrained_model_legacy}"
TOKENIZER_ROOT="${R1A7_PI05_TOKENIZER:-}"
EPISODE=0
FRAME=0
TASK=""
DEVICE=cuda
SEED=1000
DRY_RUN=0

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --dataset-root|--checkpoint|--tokenizer-root|--episode|--frame|--task|--device|--seed)
      (($# >= 2)) || {
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
      }
      case "$1" in
        --dataset-root) DATASET_ROOT="$2" ;;
        --checkpoint) CHECKPOINT="$2" ;;
        --tokenizer-root) TOKENIZER_ROOT="$2" ;;
        --episode) EPISODE="$2" ;;
        --frame) FRAME="$2" ;;
        --task) TASK="$2" ;;
        --device) DEVICE="$2" ;;
        --seed) SEED="$2" ;;
      esac
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

[[ -x "$PYTHON_BIN" ]] || {
  printf 'Missing LeRobot Python: %s\n' "$PYTHON_BIN" >&2
  exit 1
}
[[ -d "$DATASET_ROOT" ]] || {
  printf 'Missing dataset: %s\n' "$DATASET_ROOT" >&2
  exit 1
}
[[ "$EPISODE" =~ ^[0-9]+$ && "$FRAME" =~ ^[0-9]+$ && "$SEED" =~ ^[0-9]+$ ]] || {
  printf 'episode, frame and seed must be non-negative integers.\n' >&2
  exit 2
}

args=(
  --dataset-root "$DATASET_ROOT"
  --episode "$EPISODE"
  --frame "$FRAME"
)

if [[ -n "$TASK" ]]; then
  args+=(--task "$TASK")
fi

if ((DRY_RUN)); then
  args+=(--dry-run)
else
  [[ -d "$CHECKPOINT" ]] || {
    printf 'Missing PI0.5 checkpoint: %s\n' "$CHECKPOINT" >&2
    exit 1
  }
  if [[ -z "$TOKENIZER_ROOT" ]]; then
    hf_model_root="/home/user/.cache/huggingface/hub/models--google--paligemma-3b-pt-224"
    hf_ref="$hf_model_root/refs/main"
    [[ -f "$hf_ref" ]] || {
      printf 'Missing PaliGemma tokenizer cache ref: %s\n' "$hf_ref" >&2
      exit 1
    }
    IFS= read -r hf_revision < "$hf_ref" || [[ -n "$hf_revision" ]]
    [[ "$hf_revision" =~ ^[0-9a-f]{40}$ ]] || {
      printf 'Invalid Hugging Face revision in %s: %s\n' "$hf_ref" "$hf_revision" >&2
      exit 1
    }
    TOKENIZER_ROOT="$hf_model_root/snapshots/$hf_revision"
  fi
  "$PYTHON_BIN" - "$CHECKPOINT" "$TOKENIZER_ROOT" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).expanduser().resolve()
tokenizer = Path(sys.argv[2]).expanduser().resolve()
required = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
missing = [name for name in required if not (checkpoint / name).is_file()]
if missing:
    raise SystemExit(f"Checkpoint is incomplete; missing: {missing}")

tokenizer_required = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
tokenizer_missing = [name for name in tokenizer_required if not (tokenizer / name).is_file()]
if tokenizer_missing:
    raise SystemExit(
        "PI0.5 tokenizer asset is incomplete at "
        f"{tokenizer}; missing: {tokenizer_missing}."
    )
PY
  args+=(
    --policy-path "$CHECKPOINT"
    --tokenizer-path "$TOKENIZER_ROOT"
    --device "$DEVICE"
    --seed "$SEED"
  )
fi

env \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PACKAGE_ROOT:$LEROBOT_SOURCE${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m unitree_lerobot.r1a7_shadow_inference "${args[@]}"
