#!/usr/bin/env bash
set -Eeuo pipefail

readonly EXPECTED_LEROBOT_COMMIT="262d2fd07e32224d46dca8ba8bcdef5a0b3a9c3b"
readonly EXPECTED_DATASET_REPO_ID="local/r1a7_oreo_4cam_16d_v1_80_pi05_stats"

usage() {
  cat <<'EOF'
Usage: train-pi05-4x4090-smoke.sh [options] [-- LeRobot options]

Runs a bounded PI0.5 systems smoke on one node with four 24 GB GPUs. This is
not a production-training launcher.

Options:
  --strategy fsdp|ddp       Distribution strategy (default: fsdp).
  --steps N                 Micro-batch steps (default: 20 for FSDP, 1 for DDP).
  --output-dir PATH         New output directory. Existing paths are refused.
  --dataset-root PATH       Exact-stats LeRobot v3 dataset.
  --base-checkpoint PATH    Local lerobot/pi05_base snapshot.
  --tokenizer-root PATH     Local PaliGemma tokenizer snapshot.
  --lerobot-root PATH       Pinned LeRobot source checkout.
  --python PATH             Python from the isolated training environment.
  --cuda-devices A,B,C,D   Exactly four GPU indices (default: 0,1,2,3).
  --fsdp-wrap-modules JSON  Explicit FSDP module classes for PI0.5 full tuning.
  --fsdp-min-params N       Size-based fallback; used only when wrap-modules is empty.
  --save-checkpoint         Save the final smoke checkpoint.
  --checkpoint-format FMT  safetensors, dcp, or safetensors_dcp (default: dcp).
  --dry-run                 Run preflight and print the command only.
  -h, --help                Show this help.

DDP is diagnostic only: historical full-finetune DDP used 36.68 GB per GPU,
so a 24 GB RTX 4090 is expected to OOM. Use FSDP for the actual 20-50 step
systems smoke.
EOF
}

TRAINING_ROOT="${R1A7_TRAINING_ROOT:-/home/robot/r1a7-training}"
LEROBOT_ROOT="${R1A7_LEROBOT_ROOT:-$TRAINING_ROOT/lerobot}"
PYTHON_BIN="${R1A7_PI05_PYTHON:-/home/robot/miniconda3/envs/r1a7-pi05/bin/python}"
DATASET_ROOT="${R1A7_PI05_DATASET_ROOT:-$TRAINING_ROOT/datasets/r1a7_oreo_4cam_16d_v1_80_pi05_stats}"
FILTER_PATCH="${R1A7_FILTER_PATCH:-$TRAINING_ROOT/patches/lerobot/262d2fd-training-filter.patch}"
BASE_CHECKPOINT="${R1A7_PI05_BASE:-$TRAINING_ROOT/assets/pi05_base}"
TOKENIZER_ROOT="${R1A7_PI05_TOKENIZER:-$TRAINING_ROOT/assets/paligemma-3b-pt-224}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
STRATEGY="fsdp"
STEPS=""
OUTPUT_DIR=""
FSDP_MIN_PARAMS=10000000
FSDP_WRAP_MODULES='["SiglipEncoderLayer","Linear","GemmaTextScaledWordEmbedding"]'
SAVE_CHECKPOINT=false
CHECKPOINT_FORMAT="dcp"
DRY_RUN=0
EXTRA_ARGS=()

while (($#)); do
  case "$1" in
    --strategy|--steps|--output-dir|--dataset-root|--base-checkpoint|--tokenizer-root|--lerobot-root|--python|--cuda-devices|--fsdp-wrap-modules|--fsdp-min-params|--checkpoint-format)
      (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --strategy) STRATEGY="$2" ;;
        --steps) STEPS="$2" ;;
        --output-dir) OUTPUT_DIR="$2" ;;
        --dataset-root) DATASET_ROOT="$2" ;;
        --base-checkpoint) BASE_CHECKPOINT="$2" ;;
        --tokenizer-root) TOKENIZER_ROOT="$2" ;;
        --lerobot-root) LEROBOT_ROOT="$2" ;;
        --python) PYTHON_BIN="$2" ;;
        --cuda-devices) CUDA_DEVICES="$2" ;;
        --fsdp-wrap-modules) FSDP_WRAP_MODULES="$2" ;;
        --fsdp-min-params) FSDP_MIN_PARAMS="$2" ;;
        --checkpoint-format) CHECKPOINT_FORMAT="$2" ;;
      esac
      shift 2
      ;;
    --save-checkpoint)
      SAVE_CHECKPOINT=true
      shift
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
      printf 'Unknown option before --: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

case "$STRATEGY" in
  fsdp) [[ -n "$STEPS" ]] || STEPS=20 ;;
  ddp) [[ -n "$STEPS" ]] || STEPS=1 ;;
  *) printf 'strategy must be fsdp or ddp, got %s\n' "$STRATEGY" >&2; exit 2 ;;
esac
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { printf 'steps must be a positive integer\n' >&2; exit 2; }
[[ "$FSDP_MIN_PARAMS" =~ ^[1-9][0-9]*$ ]] || {
  printf 'fsdp-min-params must be a positive integer\n' >&2
  exit 2
}
case "$CHECKPOINT_FORMAT" in
  safetensors|dcp|safetensors_dcp) ;;
  *) printf 'Unsupported checkpoint format: %s\n' "$CHECKPOINT_FORMAT" >&2; exit 2 ;;
esac
if [[ "$STRATEGY" == "ddp" && "$CHECKPOINT_FORMAT" != "safetensors" && "$SAVE_CHECKPOINT" == true ]]; then
  printf 'DDP checkpoint format must be safetensors.\n' >&2
  exit 2
fi

IFS=',' read -r -a gpu_ids <<<"$CUDA_DEVICES"
if ((${#gpu_ids[@]} != 4)); then
  printf 'cuda-devices must contain exactly four indices, got %s\n' "$CUDA_DEVICES" >&2
  exit 2
fi
declare -A seen_gpu=()
for gpu_id in "${gpu_ids[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || { printf 'Invalid GPU index: %s\n' "$gpu_id" >&2; exit 2; }
  [[ -z "${seen_gpu[$gpu_id]:-}" ]] || { printf 'Duplicate GPU index: %s\n' "$gpu_id" >&2; exit 2; }
  seen_gpu[$gpu_id]=1
done

[[ -x "$PYTHON_BIN" ]] || { printf 'Missing Python: %s\n' "$PYTHON_BIN" >&2; exit 1; }
TORCHRUN_BIN="$(dirname -- "$PYTHON_BIN")/torchrun"
TRAIN_BIN="$(dirname -- "$PYTHON_BIN")/lerobot-train"
[[ -x "$TORCHRUN_BIN" ]] || { printf 'Missing torchrun: %s\n' "$TORCHRUN_BIN" >&2; exit 1; }
[[ -x "$TRAIN_BIN" ]] || { printf 'Missing lerobot-train: %s\n' "$TRAIN_BIN" >&2; exit 1; }
[[ -d "$LEROBOT_ROOT/.git" ]] || { printf 'Missing LeRobot checkout: %s\n' "$LEROBOT_ROOT" >&2; exit 1; }
actual_commit="$(git -C "$LEROBOT_ROOT" rev-parse HEAD)"
[[ "$actual_commit" == "$EXPECTED_LEROBOT_COMMIT" ]] || {
  printf 'LeRobot commit mismatch: expected %s, got %s\n' "$EXPECTED_LEROBOT_COMMIT" "$actual_commit" >&2
  exit 1
}
[[ -f "$FILTER_PATCH" ]] || { printf 'Missing training-filter patch: %s\n' "$FILTER_PATCH" >&2; exit 1; }
git -C "$LEROBOT_ROOT" apply --reverse --check "$FILTER_PATCH" >/dev/null 2>&1 || {
  printf 'Training-filter patch is not applied to the pinned LeRobot checkout: %s\n' "$FILTER_PATCH" >&2
  exit 1
}

for required in \
  "$DATASET_ROOT/meta/info.json" \
  "$DATASET_ROOT/meta/stats.json" \
  "$DATASET_ROOT/meta/r1a7_exact_stats_audit.json" \
  "$DATASET_ROOT/meta/training_filter.json" \
  "$BASE_CHECKPOINT/config.json" \
  "$BASE_CHECKPOINT/model.safetensors" \
  "$BASE_CHECKPOINT/policy_preprocessor.json" \
  "$BASE_CHECKPOINT/policy_postprocessor.json" \
  "$TOKENIZER_ROOT/tokenizer.json" \
  "$TOKENIZER_ROOT/tokenizer.model" \
  "$TOKENIZER_ROOT/tokenizer_config.json"; do
  [[ -f "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$TRAINING_ROOT/outputs/pi05_${STRATEGY}_smoke_$(date +%Y%m%d-%H%M%S)"
fi
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
[[ ! -e "$OUTPUT_DIR" ]] || { printf 'Output already exists: %s\n' "$OUTPUT_DIR" >&2; exit 1; }

for argument in "${EXTRA_ARGS[@]}"; do
  option="${argument%%=*}"
  case "$option" in
    --dataset|--dataset.*|--policy|--policy.*|--parallelism|--parallelism.*|--accelerator.fsdp.*|--accelerator.mixed_precision|--output_dir|--steps|--batch_size|--save_checkpoint|--checkpoint_format|--resume|--config_path)
      printf 'Additional argument would override the smoke contract: %s\n' "$argument" >&2
      exit 2
      ;;
  esac
done

"$PYTHON_BIN" - "$DATASET_ROOT" "$EXPECTED_DATASET_REPO_ID" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
assert info["codebase_version"] == "v3.0", info["codebase_version"]
assert info["total_episodes"] == 160, info["total_episodes"]
assert info["total_frames"] == 36630, info["total_frames"]
assert info["fps"] == 15, info["fps"]
assert info["features"]["observation.state"]["shape"] == [16]
assert info["features"]["action"]["shape"] == [16]
camera_keys = sorted(key for key in info["features"] if key.startswith("observation.images."))
assert camera_keys == sorted([
    "observation.images.cam_global",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
    "observation.images.cam_scanner",
]), camera_keys
audit = json.loads((root / "meta/r1a7_exact_stats_audit.json").read_text(encoding="utf-8"))
training_filter = json.loads((root / "meta/training_filter.json").read_text(encoding="utf-8"))
assert audit["method"] == "numpy.quantile over training-filter-allowed Parquet rows"
assert audit["sample_selection"]["selected_frames"] == 35759
assert audit["sample_selection"]["excluded_frames"] == 871
assert training_filter["allowed_training_starts"] == 35759
assert training_filter["excluded_training_starts"] == 871
print("Dataset contract OK:", sys.argv[2], "4 cameras, 16D, 35759/36630 starts allowed")
PY

mapfile -t visible_gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
for gpu_id in "${gpu_ids[@]}"; do
  printf '%s\n' "${visible_gpu_ids[@]}" | grep -Fxq "$gpu_id" || {
    printf 'GPU %s is not reported by nvidia-smi.\n' "$gpu_id" >&2
    exit 1
  }
done

TRAIN_ARGS=(
  --dataset.repo_id="$EXPECTED_DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.video_backend=pyav
  --dataset.return_uint8=true
  --dataset.training_filter_path="$DATASET_ROOT/meta/training_filter.json"
  --policy.type=pi05
  --policy.pretrained_path="$BASE_CHECKPOINT"
  --policy.input_features=null
  --policy.device=cuda
  --policy.dtype=bfloat16
  --policy.text_tokenizer_name="$TOKENIZER_ROOT"
  --policy.n_action_steps=10
  --policy.freeze_vision_encoder=false
  --policy.train_expert_only=false
  --policy.use_peft=false
  --policy.gradient_checkpointing=true
  --policy.compile_model=false
  --policy.push_to_hub=false
  --parallelism.dp_replicate=1
  --accelerator.mixed_precision=bf16
  --accelerator.gradient_accumulation.steps=1
  --batch_size=1
  --steps="$STEPS"
  --num_workers=0
  --output_dir="$OUTPUT_DIR"
  --job_name="r1a7_pi05_${STRATEGY}_smoke"
  --env_eval_freq=0
  --eval_steps=0
  --save_checkpoint="$SAVE_CHECKPOINT"
  --save_freq=0
  --checkpoint_format="$CHECKPOINT_FORMAT"
  --log_freq=1
  --ema.enable=false
  --wandb.enable=false
)
if [[ "$STRATEGY" == "fsdp" ]]; then
  TRAIN_ARGS+=(
    --parallelism.dp_shard=4
    --accelerator.fsdp.reshard_after_forward=true
    --accelerator.fsdp.cpu_offload=false
  )
  if [[ -n "$FSDP_WRAP_MODULES" ]]; then
    TRAIN_ARGS+=(--accelerator.fsdp.wrap_modules="$FSDP_WRAP_MODULES")
  else
    TRAIN_ARGS+=(--accelerator.fsdp.min_num_params="$FSDP_MIN_PARAMS")
  fi
else
  TRAIN_ARGS+=(
    --parallelism.dp_shard=1
    --accelerator.ddp.gradient_as_bucket_view=true
  )
fi
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

COMMAND=("$TORCHRUN_BIN" --standalone --nproc-per-node=4 "$TRAIN_BIN" "${TRAIN_ARGS[@]}")
printf 'Strategy: %s\nSteps: %s\nDataset: %s\nBase: %s\nTokenizer: %s\nOutput: %s\n' \
  "$STRATEGY" "$STEPS" "$DATASET_ROOT" "$BASE_CHECKPOINT" "$TOKENIZER_ROOT" "$OUTPUT_DIR"
printf 'Effective batch per optimizer update: 4 (= 1 per GPU x 4 GPUs x 1 accumulation)\n'
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
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

exec "${COMMAND[@]}"
