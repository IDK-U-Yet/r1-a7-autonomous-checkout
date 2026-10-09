#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: train-pi05-unitree-legacy.sh {smoke|production|resume} [options]

Options:
  --cuda-device N        Physical CUDA device index (default: 1).
  --output-dir PATH      New output directory for smoke/production.
  --checkpoint PATH      pretrained_model directory for resume mode.
  --dataset-root PATH    Exact-stats R1-A7 LeRobot dataset.
  --base-checkpoint PATH Local PI0.5 base checkpoint.
  --tokenizer-root PATH  Local PaliGemma tokenizer.
  --steps N              Total optimizer steps.
  --batch-size N         Per-step batch size (new runs only).
  --num-workers N        DataLoader workers.
  --save-freq N          Checkpoint frequency.
  --log-freq N           Metric logging frequency.
  --wandb                Enable online W&B metric logging.
  --wandb-project NAME   W&B project (default: r1a7-pi05-legacy).
  --dry-run              Validate and print without starting training.
  -h, --help             Show this help.
EOF
}

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "production" && "$MODE" != "resume" ]]; then
  usage >&2
  exit 2
fi
shift

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly CONDA_ROOT="${CONDA_ROOT:-/home/jieguo/anaconda3}"
readonly PYTHON_BIN="$CONDA_ROOT/envs/unitree_lerobot/bin/python"
readonly TRAIN_BIN="$CONDA_ROOT/envs/unitree_lerobot/bin/lerobot-train"
readonly OLD_SOURCE="$ROOT_DIR/data/unitree-lerobot/unitree_lerobot/lerobot/src"
readonly PACKAGE_ROOT="$ROOT_DIR/data/unitree-lerobot"

DATASET_ROOT="${R1A7_PI05_DATASET_ROOT:-/home/jieguo/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30_pi05_stats}"
BASE_CHECKPOINT="${R1A7_PI05_BASE:-/media/jieguo/disk/zyc/lerobot/pretrained/pi05_base}"
TOKENIZER_ROOT="${R1A7_PI05_TOKENIZER:-/media/jieguo/disk/zyc/lerobot/pretrained/google/paligemma-3b-pt-224}"
CUDA_DEVICE="${R1A7_LEGACY_CUDA_DEVICE:-1}"
WANDB_PROJECT="${R1A7_PI05_WANDB_PROJECT:-r1a7-pi05-legacy}"
WANDB_ENABLE=0
DRY_RUN=0
OUTPUT_DIR=""
CHECKPOINT=""

if [[ "$MODE" == "smoke" ]]; then
  STEPS=2
  BATCH_SIZE=1
  NUM_WORKERS=0
  SAVE_FREQ=1
  LOG_FREQ=1
elif [[ "$MODE" == "production" ]]; then
  STEPS="${R1A7_PI05_STEPS:-30000}"
  BATCH_SIZE="${R1A7_PI05_BATCH_SIZE:-1}"
  NUM_WORKERS="${R1A7_PI05_NUM_WORKERS:-4}"
  SAVE_FREQ="${R1A7_PI05_SAVE_FREQ:-4000}"
  LOG_FREQ="${R1A7_PI05_LOG_FREQ:-10}"
else
  STEPS="${R1A7_PI05_STEPS:-30000}"
  BATCH_SIZE=0
  NUM_WORKERS="${R1A7_PI05_NUM_WORKERS:-4}"
  SAVE_FREQ="${R1A7_PI05_SAVE_FREQ:-4000}"
  LOG_FREQ="${R1A7_PI05_LOG_FREQ:-10}"
fi

while (($#)); do
  case "$1" in
    --cuda-device|--output-dir|--checkpoint|--dataset-root|--base-checkpoint|--tokenizer-root|--steps|--batch-size|--num-workers|--save-freq|--log-freq|--wandb-project)
      (($# >= 2)) || {
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
      }
      case "$1" in
        --cuda-device) CUDA_DEVICE="$2" ;;
        --output-dir) OUTPUT_DIR="$2" ;;
        --checkpoint) CHECKPOINT="$2" ;;
        --dataset-root) DATASET_ROOT="$2" ;;
        --base-checkpoint) BASE_CHECKPOINT="$2" ;;
        --tokenizer-root) TOKENIZER_ROOT="$2" ;;
        --steps) STEPS="$2" ;;
        --batch-size) BATCH_SIZE="$2" ;;
        --num-workers) NUM_WORKERS="$2" ;;
        --save-freq) SAVE_FREQ="$2" ;;
        --log-freq) LOG_FREQ="$2" ;;
        --wandb-project) WANDB_PROJECT="$2" ;;
      esac
      shift 2
      ;;
    --wandb)
      WANDB_ENABLE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

for executable in "$PYTHON_BIN" "$TRAIN_BIN"; do
  [[ -x "$executable" ]] || {
    printf 'Missing executable: %s\n' "$executable" >&2
    exit 1
  }
done
[[ -d "$OLD_SOURCE/lerobot" ]] || {
  printf 'Missing vendored LeRobot source: %s\n' "$OLD_SOURCE" >&2
  exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || {
  printf 'Missing executable: nvidia-smi\n' >&2
  exit 1
}

for pair in \
  "steps:$STEPS" \
  "batch-size:$BATCH_SIZE" \
  "num-workers:$NUM_WORKERS" \
  "save-freq:$SAVE_FREQ" \
  "log-freq:$LOG_FREQ"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  [[ "$value" =~ ^[0-9]+$ ]] || {
    printf '%s must be a non-negative integer, got %s\n' "$name" "$value" >&2
    exit 2
  }
done
[[ "$CUDA_DEVICE" =~ ^[0-9]+$ ]] || {
  printf 'cuda-device must be a non-negative integer, got %s\n' "$CUDA_DEVICE" >&2
  exit 2
}
((STEPS > 0 && SAVE_FREQ > 0 && LOG_FREQ > 0)) || {
  printf 'steps, save-freq, and log-freq must be positive\n' >&2
  exit 2
}
if [[ "$MODE" != "resume" ]] && ((BATCH_SIZE == 0)); then
  printf 'batch-size must be positive for a new run\n' >&2
  exit 2
fi

mapfile -t available_gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
gpu_found=0
for gpu_id in "${available_gpu_ids[@]}"; do
  if [[ "$CUDA_DEVICE" == "$gpu_id" ]]; then
    gpu_found=1
    break
  fi
done
((gpu_found)) || {
  printf 'CUDA device %s is not reported by nvidia-smi\n' "$CUDA_DEVICE" >&2
  exit 1
}

DATASET_ROOT="$(realpath -e -- "$DATASET_ROOT")"
BASE_CHECKPOINT="$(realpath -e -- "$BASE_CHECKPOINT")"
TOKENIZER_ROOT="$(realpath -e -- "$TOKENIZER_ROOT")"

if [[ "$MODE" == "resume" ]]; then
  [[ -n "$CHECKPOINT" ]] || {
    printf 'resume mode requires --checkpoint PRETRAINED_MODEL_DIR\n' >&2
    exit 2
  }
  CHECKPOINT="$(realpath -e -- "$CHECKPOINT")"
  [[ -f "$CHECKPOINT/train_config.json" ]] || {
    printf 'Checkpoint is missing train_config.json: %s\n' "$CHECKPOINT" >&2
    exit 1
  }
  [[ -f "$CHECKPOINT/model.safetensors" ]] || {
    printf 'Checkpoint is missing model.safetensors: %s\n' "$CHECKPOINT" >&2
    exit 1
  }
  if [[ -n "$OUTPUT_DIR" ]]; then
    printf 'resume mode uses the output_dir saved in train_config.json; do not pass --output-dir\n' >&2
    exit 2
  fi
else
  if [[ -z "$OUTPUT_DIR" ]]; then
    timestamp="$(date +%Y%m%d-%H%M%S)"
    OUTPUT_DIR="$ROOT_DIR/outputs/train/r1a7_pi05_legacy_${MODE}_${timestamp}"
  fi
  OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
  [[ ! -e "$OUTPUT_DIR" ]] || {
    printf 'Refusing to reuse existing output directory: %s\n' "$OUTPUT_DIR" >&2
    exit 1
  }
fi

env \
  CUDA_VISIBLE_DEVICES='' \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$OLD_SOURCE:$PACKAGE_ROOT" \
  "$PYTHON_BIN" - "$DATASET_ROOT" "$BASE_CHECKPOINT" "$TOKENIZER_ROOT" "$OLD_SOURCE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import lerobot
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from unitree_lerobot.r1a7_contract import load_dataset_info

dataset_root, base_root, tokenizer_root, expected_source = map(Path, sys.argv[1:])
source_file = Path(lerobot.__file__).resolve()
if not source_file.is_relative_to(expected_source.resolve()):
    raise RuntimeError(
        f"expected vendored LeRobot under {expected_source}, imported {source_file}"
    )
if getattr(lerobot, "__version__", None) != "0.4.1":
    raise RuntimeError(f"expected LeRobot 0.4.1, got {lerobot.__version__!r}")

info = load_dataset_info(dataset_root)
if info.get("total_tasks") != 2:
    raise RuntimeError(f"expected two R1-A7 tasks, got {info.get('total_tasks')!r}")
if not (dataset_root / "meta/r1a7_exact_stats_audit.json").is_file():
    raise RuntimeError("training dataset is missing exact-stats audit metadata")

with (base_root / "config.json").open(encoding="utf-8") as stream:
    base_config = json.load(stream)
if base_config.get("type") != "pi05":
    raise RuntimeError(f"expected PI0.5 base, got {base_config.get('type')!r}")
for required in (base_root / "model.safetensors", tokenizer_root / "tokenizer.json"):
    if not required.is_file():
        raise FileNotFoundError(required)

config = PI05Config(
    device="cpu",
    dtype="bfloat16",
    gradient_checkpointing=True,
    push_to_hub=False,
    tokenizer_name=str(tokenizer_root),
    n_action_steps=10,
)
if config.max_state_dim != 32 or config.max_action_dim != 32:
    raise RuntimeError("legacy PI0.5 internal padding contract must remain 32D")
print(
    "Legacy training preflight passed: LeRobot 0.4.1, four cameras, "
    "16D state/action, two tasks, PI0.5 base, local tokenizer"
)
PY

if [[ "$MODE" == "resume" ]]; then
  COMMAND=(
    "$TRAIN_BIN"
    --config_path="$CHECKPOINT/train_config.json"
    --resume=true
    --steps="$STEPS"
    --num_workers="$NUM_WORKERS"
    --save_freq="$SAVE_FREQ"
    --log_freq="$LOG_FREQ"
    --wandb.enable="$([[ "$WANDB_ENABLE" == 1 ]] && printf true || printf false)"
    --wandb.project="$WANDB_PROJECT"
    --wandb.mode=online
    --wandb.disable_artifact=true
  )
else
  COMMAND=(
    "$TRAIN_BIN"
    --dataset.repo_id=local/r1a7_black_truffle_4cam_30_pi05_stats
    --dataset.root="$DATASET_ROOT"
    --dataset.video_backend=pyav
    --dataset.use_imagenet_stats=false
    --policy.type=pi05
    --policy.pretrained_path="$BASE_CHECKPOINT"
    --policy.device=cuda
    --policy.dtype=bfloat16
    --policy.use_amp=true
    --policy.tokenizer_name="$TOKENIZER_ROOT"
    --policy.n_action_steps=10
    --policy.gradient_checkpointing=true
    --policy.compile_model=false
    --policy.push_to_hub=false
    --batch_size="$BATCH_SIZE"
    --steps="$STEPS"
    --num_workers="$NUM_WORKERS"
    --output_dir="$OUTPUT_DIR"
    --job_name="r1a7_pi05_legacy_${MODE}"
    --eval_freq=0
    --save_checkpoint=true
    --save_freq="$SAVE_FREQ"
    --log_freq="$LOG_FREQ"
    --wandb.enable="$([[ "$WANDB_ENABLE" == 1 ]] && printf true || printf false)"
    --wandb.project="$WANDB_PROJECT"
    --wandb.mode=online
    --wandb.disable_artifact=true
  )
fi

printf 'Mode: %s\n' "$MODE"
printf 'Physical GPU: %s (visible as cuda:0 inside the process)\n' "$CUDA_DEVICE"
printf 'Dataset: %s\n' "$DATASET_ROOT"
if [[ "$MODE" == "resume" ]]; then
  printf 'Resume checkpoint: %s\n' "$CHECKPOINT"
else
  printf 'Base checkpoint: %s\n' "$BASE_CHECKPOINT"
  printf 'Output: %s\n' "$OUTPUT_DIR"
  printf 'Batch size: %s\n' "$BATCH_SIZE"
fi
printf 'Total steps: %s\n' "$STEPS"
printf 'Command:\n  '
printf '%q ' env \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  ACCELERATE_MIXED_PRECISION=bf16 \
  PYTHONPATH="$OLD_SOURCE:$PACKAGE_ROOT" \
  "${COMMAND[@]}"
printf '\n'

if ((DRY_RUN)); then
  printf 'Dry run complete; training was not started.\n'
  exit 0
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export ACCELERATE_MIXED_PRECISION=bf16
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$OLD_SOURCE:$PACKAGE_ROOT"

exec "${COMMAND[@]}"
