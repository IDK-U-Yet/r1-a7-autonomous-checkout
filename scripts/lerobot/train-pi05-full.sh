#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly SOURCE_DATASET_DEFAULT="/home/jieguo/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30"

usage() {
  cat <<'EOF'
Usage: train-pi05-full.sh {smoke|production} [options] [-- LeRobot options]

Options:
  --output-dir PATH          New output directory (must not already exist).
  --dataset-root PATH        Exact-stats derived LeRobot dataset.
  --base-checkpoint PATH     Local LeRobot PI0.5 base checkpoint.
  --tokenizer-root PATH      Local PaliGemma tokenizer directory.
  --steps N                  Training micro-batch steps.
  --batch-size N             Per-GPU micro-batch size.
  --gradient-accumulation N  Accumulation steps per optimizer update.
  --num-workers N            DataLoader workers per rank.
  --save-freq N              Production checkpoint frequency.
  --log-freq N               Logging frequency.
  --cuda-devices A,B         Exactly two CUDA device indices.
  --dry-run                  Validate and print, but do not start training.
  -h, --help                 Show this help.

Additional LeRobot options after -- may tune non-contract settings such as
optimizer LR or image augmentation. Core full-finetune, four-camera, DDP,
dataset, checkpoint and output settings cannot be overridden there.
EOF
}

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "production" ]]; then
  usage >&2
  exit 2
fi
shift

readonly CONDA_ROOT="${CONDA_ROOT:-/home/jieguo/anaconda3}"
readonly PYTHON_BIN="$CONDA_ROOT/envs/lerobot/bin/python"
readonly TORCHRUN_BIN="$CONDA_ROOT/envs/lerobot/bin/torchrun"
readonly TRAIN_BIN="$CONDA_ROOT/envs/lerobot/bin/lerobot-train"

DATASET_ROOT="${R1A7_PI05_DATASET_ROOT:-/home/jieguo/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30_pi05_stats}"
BASE_CHECKPOINT="${R1A7_PI05_BASE:-/media/jieguo/disk/zyc/lerobot/pretrained/pi05_base}"
TOKENIZER_ROOT="${R1A7_PI05_TOKENIZER:-/media/jieguo/disk/zyc/lerobot/pretrained/google/paligemma-3b-pt-224}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DRY_RUN=0
OUTPUT_DIR=""
EXTRA_ARGS=()

if [[ "$MODE" == "smoke" ]]; then
  STEPS=2
  BATCH_SIZE=1
  GRADIENT_ACCUMULATION=1
  NUM_WORKERS=2
  SAVE_FREQ=0
  LOG_FREQ=1
  SAVE_CHECKPOINT=false
else
  STEPS="${R1A7_PI05_STEPS:-30000}"
  BATCH_SIZE="${R1A7_PI05_BATCH_SIZE:-1}"
  GRADIENT_ACCUMULATION="${R1A7_PI05_GRADIENT_ACCUMULATION:-8}"
  NUM_WORKERS="${R1A7_PI05_NUM_WORKERS:-4}"
  SAVE_FREQ="${R1A7_PI05_SAVE_FREQ:-1000}"
  LOG_FREQ="${R1A7_PI05_LOG_FREQ:-10}"
  SAVE_CHECKPOINT=true
fi

while (($#)); do
  case "$1" in
    --output-dir|--dataset-root|--base-checkpoint|--tokenizer-root|--steps|--batch-size|--gradient-accumulation|--num-workers|--save-freq|--log-freq|--cuda-devices)
      (($# >= 2)) || {
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
      }
      case "$1" in
        --output-dir) OUTPUT_DIR="$2" ;;
        --dataset-root) DATASET_ROOT="$2" ;;
        --base-checkpoint) BASE_CHECKPOINT="$2" ;;
        --tokenizer-root) TOKENIZER_ROOT="$2" ;;
        --steps) STEPS="$2" ;;
        --batch-size) BATCH_SIZE="$2" ;;
        --gradient-accumulation) GRADIENT_ACCUMULATION="$2" ;;
        --num-workers) NUM_WORKERS="$2" ;;
        --save-freq) SAVE_FREQ="$2" ;;
        --log-freq) LOG_FREQ="$2" ;;
        --cuda-devices) CUDA_DEVICES="$2" ;;
      esac
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown script option before --: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

for pair in \
  "steps:$STEPS" \
  "batch-size:$BATCH_SIZE" \
  "gradient-accumulation:$GRADIENT_ACCUMULATION" \
  "num-workers:$NUM_WORKERS" \
  "save-freq:$SAVE_FREQ" \
  "log-freq:$LOG_FREQ"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer, got %s\n' "$name" "$value" >&2
    exit 2
  fi
done
((STEPS > 0 && BATCH_SIZE > 0 && GRADIENT_ACCUMULATION > 0 && LOG_FREQ > 0)) || {
  printf 'steps, batch size, gradient accumulation and log frequency must be positive.\n' >&2
  exit 2
}

IFS=',' read -r -a gpu_ids <<<"$CUDA_DEVICES"
if ((${#gpu_ids[@]} != 2)) || [[ -z "${gpu_ids[0]}" || -z "${gpu_ids[1]}" ]]; then
  printf '--cuda-devices must name exactly two devices, got %s\n' "$CUDA_DEVICES" >&2
  exit 2
fi
if [[ ! "${gpu_ids[0]}" =~ ^[0-9]+$ || ! "${gpu_ids[1]}" =~ ^[0-9]+$ || "${gpu_ids[0]}" == "${gpu_ids[1]}" ]]; then
  printf '--cuda-devices must contain two distinct numeric indices, got %s\n' "$CUDA_DEVICES" >&2
  exit 2
fi

for executable in "$PYTHON_BIN" "$TORCHRUN_BIN" "$TRAIN_BIN"; do
  [[ -x "$executable" ]] || {
    printf 'Missing executable: %s\n' "$executable" >&2
    exit 1
  }
done
command -v nvidia-smi >/dev/null 2>&1 || {
  printf 'Missing executable: nvidia-smi\n' >&2
  exit 1
}
mapfile -t available_gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
for gpu_id in "${gpu_ids[@]}"; do
  gpu_found=0
  for available_gpu_id in "${available_gpu_ids[@]}"; do
    if [[ "$gpu_id" == "$available_gpu_id" ]]; then
      gpu_found=1
      break
    fi
  done
  ((gpu_found)) || {
    printf 'Requested CUDA device %s is not reported by nvidia-smi.\n' "$gpu_id" >&2
    exit 1
  }
done

DATASET_ROOT="$(realpath -e -- "$DATASET_ROOT")" || {
  printf 'Missing derived dataset: %s\n' "$DATASET_ROOT" >&2
  exit 1
}
BASE_CHECKPOINT="$(realpath -e -- "$BASE_CHECKPOINT")" || {
  printf 'Missing base checkpoint: %s\n' "$BASE_CHECKPOINT" >&2
  exit 1
}
TOKENIZER_ROOT="$(realpath -e -- "$TOKENIZER_ROOT")" || {
  printf 'Missing tokenizer: %s\n' "$TOKENIZER_ROOT" >&2
  exit 1
}
SOURCE_DATASET="$(realpath -e -- "$SOURCE_DATASET_DEFAULT")"
if [[ "$DATASET_ROOT" == "$SOURCE_DATASET" ]]; then
  printf 'Refusing to train on the source dataset with inexact quantiles: %s\n' "$DATASET_ROOT" >&2
  exit 1
fi
[[ -f "$DATASET_ROOT/meta/r1a7_exact_stats_audit.json" ]] || {
  printf 'Derived dataset is missing exact-stats audit metadata: %s\n' "$DATASET_ROOT" >&2
  exit 1
}

if [[ -z "$OUTPUT_DIR" ]]; then
  timestamp="$(date +%Y%m%d-%H%M%S)"
  OUTPUT_DIR="/media/jieguo/disk/zyc/lerobot/outputs/train/r1a7_pi05_${MODE}_${timestamp}"
fi
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
if [[ -e "$OUTPUT_DIR" ]]; then
  printf 'Refusing to overwrite an existing output path: %s\n' "$OUTPUT_DIR" >&2
  exit 1
fi

for argument in "${EXTRA_ARGS[@]}"; do
  option="${argument%%=*}"
  case "$option" in
    --config_path|--resume|--output_dir|--dataset|--dataset.root|--dataset.repo_id|--policy|--policy.path|--policy.pretrained_path|--policy.type|--policy.input_features|--policy.device|--policy.dtype|--policy.freeze_vision_encoder|--policy.train_expert_only|--policy.use_peft|--policy.tokenizer_name|--policy.n_action_steps|--peft|--peft.*|--rename_map|--parallelism|--parallelism.*|--accelerator.mixed_precision|--accelerator.ddp.*|--job.target)
      printf 'Additional argument would violate the fixed training contract: %s\n' "$argument" >&2
      exit 2
      ;;
  esac
done

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  "$ROOT_DIR/scripts/lerobot/check_r1a7_setup.py" \
  --training-only \
  --dataset-root "$DATASET_ROOT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --tokenizer-root "$TOKENIZER_ROOT"

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" -c '
import sys
from pathlib import Path

from lerobot.configs import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.feature_utils import dataset_to_policy_features

checkpoint, dataset_root, tokenizer_root = sys.argv[1:]
config = PI05Config(
    pretrained_path=Path(checkpoint),
    input_features=None,
    device="cpu",
    dtype="bfloat16",
    freeze_vision_encoder=False,
    train_expert_only=False,
    gradient_checkpointing=True,
    push_to_hub=False,
    tokenizer_name=tokenizer_root,
    n_action_steps=10,
)
metadata = LeRobotDatasetMetadata(
    "local/r1a7_black_truffle_4cam_30_pi05_stats",
    root=dataset_root,
)
features = dataset_to_policy_features(metadata.features)
config.output_features = {
    key: feature for key, feature in features.items() if feature.type is FeatureType.ACTION
}
if not config.input_features:
    config.input_features = {
        key: feature for key, feature in features.items() if key not in config.output_features
    }
config.validate_features()
camera_keys = sorted(
    key for key, feature in config.input_features.items() if feature.type is FeatureType.VISUAL
)
expected_cameras = sorted(
    [
        "observation.images.cam_global",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
        "observation.images.cam_scanner",
    ]
)
assert camera_keys == expected_cameras, camera_keys
assert config.input_features["observation.state"].shape == (16,)
assert config.output_features["action"].shape == (16,)
assert config.freeze_vision_encoder is False
assert config.train_expert_only is False
assert config.use_peft is False
assert config.pretrained_path == Path(checkpoint)
assert config.tokenizer_name == tokenizer_root
assert config.n_action_steps == 10
print("Full-finetune contract: 4 cameras, 16D state/action, no frozen towers, no PEFT")
' "$BASE_CHECKPOINT" "$DATASET_ROOT" "$TOKENIZER_ROOT"

JOB_NAME="r1a7_pi05_${MODE}"
TRAIN_ARGS=(
  --dataset.repo_id=local/r1a7_black_truffle_4cam_30_pi05_stats
  --dataset.root="$DATASET_ROOT"
  --dataset.video_backend=pyav
  --dataset.return_uint8=true
  --policy.type=pi05
  --policy.pretrained_path="$BASE_CHECKPOINT"
  --policy.input_features=null
  --policy.device=cuda
  --policy.dtype=bfloat16
  --policy.tokenizer_name="$TOKENIZER_ROOT"
  --policy.n_action_steps=10
  --policy.freeze_vision_encoder=false
  --policy.train_expert_only=false
  --policy.gradient_checkpointing=true
  --policy.compile_model=false
  --policy.push_to_hub=false
  --parallelism.dp_replicate=2
  --parallelism.dp_shard=1
  --accelerator.mixed_precision=bf16
  --accelerator.gradient_accumulation.steps="$GRADIENT_ACCUMULATION"
  --accelerator.ddp.gradient_as_bucket_view=true
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --num_workers="$NUM_WORKERS"
  --output_dir="$OUTPUT_DIR"
  --job_name="$JOB_NAME"
  --env_eval_freq=0
  --eval_steps=0
  --save_checkpoint="$SAVE_CHECKPOINT"
  --save_freq="$SAVE_FREQ"
  --log_freq="$LOG_FREQ"
  --ema.enable=false
  --wandb.enable=false
  "${EXTRA_ARGS[@]}"
)
COMMAND=(
  "$TORCHRUN_BIN"
  --standalone
  --nproc-per-node=2
  "$TRAIN_BIN"
  "${TRAIN_ARGS[@]}"
)

printf '\nMode: %s\n' "$MODE"
printf 'Dataset: %s\n' "$DATASET_ROOT"
printf 'Checkpoint: %s\n' "$BASE_CHECKPOINT"
printf 'Tokenizer: %s\n' "$TOKENIZER_ROOT"
printf 'Output: %s\n' "$OUTPUT_DIR"
printf 'Effective batch per optimizer update: %d (= %d per GPU x 2 GPUs x %d accumulation)\n' \
  "$((BATCH_SIZE * 2 * GRADIENT_ACCUMULATION))" "$BATCH_SIZE" "$GRADIENT_ACCUMULATION"
printf 'Command:\n  '
printf '%q ' "${COMMAND[@]}"
printf '\n'

if ((DRY_RUN)); then
  printf 'Dry run complete; training was not started.\n'
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

exec "${COMMAND[@]}"
