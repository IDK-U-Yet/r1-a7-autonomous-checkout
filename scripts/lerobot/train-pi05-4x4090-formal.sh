#!/usr/bin/env bash
set -Eeuo pipefail

# Formal PI0.5 full-finetune launcher for the 4x4090 FSDP2 topology. Unlike the smoke
# launcher, this consumes meta/episode_split.json (leakage-free train/val/test grouping by
# original teaching episode) and adds disk/GPU prechecks, periodic checkpoints, offline val
# loss, and resume. Steps/batch below are defaults, not a decision: see --help.

readonly EXPECTED_LEROBOT_COMMIT="262d2fd07e32224d46dca8ba8bcdef5a0b3a9c3b"
readonly EXPECTED_DATASET_REPO_ID="local/r1a7_oreo_4cam_16d_v1_80_pi05_stats"
readonly EXPECTED_EPISODE_SPLIT_SHA256="53c12e7768cf53db0201641ba77147590dcb546a9130baf7d15b5fc87efe487b"
readonly EXPECTED_TRAINING_FILTER_SHA256="589182c4a4aa4ab0236417fea30c8c232570bf4a13a90e5e35f0503809c946b7"
readonly NUM_GPUS=4
readonly TRAIN_ALLOWED_STARTS=31930

usage() {
  cat <<'EOF'
Usage: train-pi05-4x4090-formal.sh [options]

Formal PI0.5 full-parameter FSDP2 training on four 24 GB GPUs, driven by the
leakage-free episode split manifest (train/val from meta/episode_split.json;
the test split is never touched here).

Options:
  --steps N                 Micro-batch steps (default: 24000). One optimizer update
                            consumes NUM_GPUS x batch_size x grad_accum micro-batches.
  --batch-size N            Per-GPU micro-batch (default: 2; probe memory first!).
  --grad-accum N            Gradient accumulation steps (default: 1).
  --peak-lr X               PI0.5 peak learning rate (default: 2.5e-5).
  --decay-lr X               PI0.5 final learning rate (default: 2.5e-6).
  --warmup-steps N           Scheduler warmup steps (default: 1000).
  --decay-steps N            Scheduler decay horizon (default: 24000, matching the default run).
  --save-freq N             Checkpoint every N steps (default: 2000, ~40 GiB each).
  --checkpoint-keep-last N  Keep the newest N completed checkpoints (default: 3).
  --checkpoint-keep-steps J JSON list of milestone steps to retain, e.g. '[8000,16000]'.
  --eval-steps N            Offline val loss every N steps (0 = off; default: 1000).
  --num-workers N           Dataloader workers (default: 4).
  --output-dir PATH         Run directory (default: outputs/pi05_oreo_fsdp2_formal_<ts>).
  --resume-from PATH        Resume: path to a checkpoint's train_config.json or
                            pretrained_model/ dir. Reuses its config; --output-dir may
                            be omitted to continue in the original run dir.
  --save-checkpoint/--no-save-checkpoint
                            Save the final checkpoint (default: save).
  --min-free-gb N           Refuse to start with less free disk (default: 180).
  --skip-gpu-idle-check     Skip the "GPUs look busy" preflight (debugging only).
  --dry-run                 Run all preflight checks and print the command only.
  -h, --help                Show this help.

Effective batch = batch_size x 4 GPUs x grad_accum. Memory probe before first use:
run with --steps 8 --batch-size 2 --grad-accum 1 --eval-steps 0 --save-freq 0
--no-save-checkpoint --output-dir outputs/probe_b2 and watch nvidia-smi; if it OOMs,
rerun the probe with --batch-size 1 (then use --batch-size 1 --grad-accum 2).
EOF
}

TRAINING_ROOT="${R1A7_TRAINING_ROOT:-/home/wzr/r1a7-training}"
LEROBOT_ROOT="${R1A7_LEROBOT_ROOT:-$TRAINING_ROOT/lerobot}"
PYTHON_BIN="${R1A7_PI05_PYTHON:-/home/robot/miniconda3/envs/r1a7-pi05/bin/python}"
DATASET_ROOT="${R1A7_PI05_DATASET_ROOT:-$TRAINING_ROOT/datasets/r1a7_oreo_4cam_16d_v1_80_pi05_stats}"
BASE_CHECKPOINT="${R1A7_PI05_BASE:-$TRAINING_ROOT/assets/pi05_base}"
TOKENIZER_ROOT="${R1A7_PI05_TOKENIZER:-$TRAINING_ROOT/assets/paligemma-3b-pt-224}"
PATCH_DIR="$TRAINING_ROOT"
FILTER_PATCH="$TRAINING_ROOT/patches/lerobot/262d2fd-training-filter.patch"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
STEPS=24000
BATCH_SIZE=2
GRAD_ACCUM=1
PEAK_LR="2.5e-5"
DECAY_LR="2.5e-6"
WARMUP_STEPS=1000
DECAY_STEPS=24000
SAVE_FREQ=2000
CHECKPOINT_KEEP_LAST=3
CHECKPOINT_KEEP_STEPS='[]'
EVAL_STEPS=1000
NUM_WORKERS=4
OUTPUT_DIR=""
RESUME_FROM=""
SAVE_CHECKPOINT=true
MIN_FREE_GB=180
SKIP_GPU_IDLE_CHECK=0
DRY_RUN=0

while (($#)); do
  case "$1" in
    --steps|--batch-size|--grad-accum|--warmup-steps|--decay-steps|--save-freq|--checkpoint-keep-last|--checkpoint-keep-steps|--eval-steps|--num-workers|--output-dir|--resume-from|--min-free-gb|--peak-lr|--decay-lr)
      (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --steps) STEPS="$2" ;;
        --batch-size) BATCH_SIZE="$2" ;;
        --grad-accum) GRAD_ACCUM="$2" ;;
        --peak-lr) PEAK_LR="$2" ;;
        --decay-lr) DECAY_LR="$2" ;;
        --warmup-steps) WARMUP_STEPS="$2" ;;
        --decay-steps) DECAY_STEPS="$2" ;;
        --save-freq) SAVE_FREQ="$2" ;;
        --checkpoint-keep-last) CHECKPOINT_KEEP_LAST="$2" ;;
        --checkpoint-keep-steps) CHECKPOINT_KEEP_STEPS="$2" ;;
        --eval-steps) EVAL_STEPS="$2" ;;
        --num-workers) NUM_WORKERS="$2" ;;
        --output-dir) OUTPUT_DIR="$2" ;;
        --resume-from) RESUME_FROM="$2" ;;
        --min-free-gb) MIN_FREE_GB="$2" ;;
      esac
      shift 2
      ;;
    --save-checkpoint) SAVE_CHECKPOINT=true; shift ;;
    --no-save-checkpoint) SAVE_CHECKPOINT=false; shift ;;
    --skip-gpu-idle-check) SKIP_GPU_IDLE_CHECK=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

for int_arg in "steps:$STEPS" "batch-size:$BATCH_SIZE" "grad-accum:$GRAD_ACCUM" \
               "warmup-steps:$WARMUP_STEPS" "decay-steps:$DECAY_STEPS" \
               "save-freq:$SAVE_FREQ" "checkpoint-keep-last:$CHECKPOINT_KEEP_LAST" \
               "eval-steps:$EVAL_STEPS" "num-workers:$NUM_WORKERS" \
               "min-free-gb:$MIN_FREE_GB"; do
  [[ "${int_arg#*:}" =~ ^[0-9]+$ ]] || { printf '%s must be a non-negative integer (got %s)\n' "${int_arg%%:*}" "${int_arg#*:}" >&2; exit 2; }
done
((STEPS > 0)) || { printf 'steps must be positive\n' >&2; exit 2; }
((BATCH_SIZE > 0 && GRAD_ACCUM > 0)) || { printf 'batch-size and grad-accum must be positive\n' >&2; exit 2; }
((CHECKPOINT_KEEP_LAST > 0)) || { printf 'checkpoint-keep-last must be positive\n' >&2; exit 2; }
((WARMUP_STEPS <= DECAY_STEPS)) || { printf 'warmup-steps must not exceed decay-steps\n' >&2; exit 2; }
[[ "$PEAK_LR" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || { printf 'peak-lr must be a positive decimal (got %s)\n' "$PEAK_LR" >&2; exit 2; }
[[ "$DECAY_LR" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || { printf 'decay-lr must be a positive decimal (got %s)\n' "$DECAY_LR" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { printf 'Missing Python: %s\n' "$PYTHON_BIN" >&2; exit 1; }
"$PYTHON_BIN" - "$CHECKPOINT_KEEP_STEPS" <<'PY'
import json
import sys

try:
    steps = json.loads(sys.argv[1])
except json.JSONDecodeError as exc:
    raise SystemExit(f"checkpoint-keep-steps must be a JSON list: {exc}")
if not isinstance(steps, list) or any(not isinstance(step, int) or step < 0 for step in steps):
    raise SystemExit("checkpoint-keep-steps must be a JSON list of non-negative integers")
PY
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

# Non-overlapping pinned patches use reverse-check. The later eval-filter patch intentionally
# relocates one line introduced by the training-filter patch, so that earlier patch is verified by
# source-contract markers instead of an exact reverse application.
[[ -f "$FILTER_PATCH" ]] || { printf 'Missing patch file: %s\n' "$FILTER_PATCH" >&2; exit 1; }
for patch in \
  "$PATCH_DIR/262d2fd-pi05-fsdp2-forward-hooks.patch" \
  "$PATCH_DIR/262d2fd-fsdp2-lazy-optimizer-resume.patch" \
  "$PATCH_DIR/262d2fd-episode-split.patch" \
  "$PATCH_DIR/262d2fd-episode-eval-filter.patch" \
  "$PATCH_DIR/262d2fd-checkpoint-retention.patch"; do
  [[ -f "$patch" ]] || { printf 'Missing patch file: %s\n' "$patch" >&2; exit 1; }
  git -C "$LEROBOT_ROOT" apply --reverse --check "$patch" >/dev/null 2>&1 || {
    printf 'Patch is not applied to the pinned LeRobot checkout: %s\n' "$patch" >&2
    exit 1
  }
done
grep -Fq 'training_filter_path: str | None = None' "$LEROBOT_ROOT/src/lerobot/configs/default.py" || {
  printf 'Training-filter DatasetConfig field is missing from the pinned LeRobot checkout.\n' >&2
  exit 1
}
grep -Fq 'excluded_ranges=excluded_ranges' "$LEROBOT_ROOT/src/lerobot/scripts/lerobot_train.py" || {
  printf 'Training-filter sampler wiring is missing from the pinned LeRobot checkout.\n' >&2
  exit 1
}
grep -Fq 'def test_excluded_absolute_ranges' "$LEROBOT_ROOT/tests/datasets/test_sampler.py" || {
  printf 'Training-filter sampler contract test is missing from the pinned LeRobot checkout.\n' >&2
  exit 1
}

SPLIT_PATH="$DATASET_ROOT/meta/episode_split.json"
for required in \
  "$DATASET_ROOT/meta/info.json" \
  "$DATASET_ROOT/meta/stats.json" \
  "$DATASET_ROOT/meta/r1a7_exact_stats_audit.json" \
  "$DATASET_ROOT/meta/training_filter.json" \
  "$SPLIT_PATH" \
  "$BASE_CHECKPOINT/config.json" \
  "$BASE_CHECKPOINT/model.safetensors" \
  "$BASE_CHECKPOINT/policy_preprocessor.json" \
  "$BASE_CHECKPOINT/policy_postprocessor.json" \
  "$TOKENIZER_ROOT/tokenizer.json" \
  "$TOKENIZER_ROOT/tokenizer.model" \
  "$TOKENIZER_ROOT/tokenizer_config.json"; do
  [[ -f "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
actual_split_sha="$(sha256sum "$SPLIT_PATH" | cut -d' ' -f1)"
[[ "$actual_split_sha" == "$EXPECTED_EPISODE_SPLIT_SHA256" ]] || {
  printf 'episode_split.json hash mismatch:\n  expected %s\n  got      %s\n' \
    "$EXPECTED_EPISODE_SPLIT_SHA256" "$actual_split_sha" >&2
  exit 1
}
actual_filter_sha="$(sha256sum "$DATASET_ROOT/meta/training_filter.json" | cut -d' ' -f1)"
[[ "$actual_filter_sha" == "$EXPECTED_TRAINING_FILTER_SHA256" ]] || {
  printf 'training_filter.json hash mismatch:\n  expected %s\n  got      %s\n' \
    "$EXPECTED_TRAINING_FILTER_SHA256" "$actual_filter_sha" >&2
  exit 1
}

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
training_filter = json.loads((root / "meta/training_filter.json").read_text(encoding="utf-8"))
assert training_filter["allowed_training_starts"] == 35759, training_filter["allowed_training_starts"]
split = json.loads((root / "meta/episode_split.json").read_text(encoding="utf-8"))
assert split["validation"]["each_source_in_one_split"] is True
assert split["validation"]["filter_allowed_starts_match"] is True, split["validation"]
assert split["splits"]["train"]["allowed_training_starts"] == 31930
assert split["splits"]["val"]["allowed_training_starts"] == 1924
assert split["splits"]["test"]["allowed_training_starts"] == 1905
all_output = sorted(
    idx for name in ("train", "val", "test") for idx in split["splits"][name]["output_episode_indices"]
)
assert all_output == list(range(160)), "split must cover all 160 episodes exactly once"
print("Dataset contract OK:", sys.argv[2], "| 16D | 35759/36630 starts | split covers 160 episodes")
PY

# Disk preflight: each checkpoint is ~40 GiB; keep room for periodic saves before pruning.
free_kb=$(df -Pk "$TRAINING_ROOT" | awk 'NR==2 {print $4}')
free_gb=$((free_kb / 1024 / 1024))
if ((free_gb < MIN_FREE_GB)); then
  printf 'Only %s GiB free under %s (need >= %s GiB for checkpoints).\n' "$free_gb" "$TRAINING_ROOT" "$MIN_FREE_GB" >&2
  printf 'Archive or remove old run outputs first (each checkpoint ~40 GiB), then retry.\n' >&2
  exit 1
fi

if ((SKIP_GPU_IDLE_CHECK == 0)); then
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 {n++} END {print n+0}')
  if ((busy > 0)); then
    printf '%s GPU(s) already have >2 GiB allocated; another training may be running.\n' "$busy" >&2
    printf 'Pass --skip-gpu-idle-check only if you are sure this is safe.\n' >&2
    exit 1
  fi
fi

RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
  resume_path="$(realpath -m -- "$RESUME_FROM")"
  [[ -e "$resume_path" ]] || { printf 'resume-from path does not exist: %s\n' "$resume_path" >&2; exit 1; }
  if [[ -f "$resume_path" && "$resume_path" != *"train_config.json" ]]; then
    printf 'resume-from file must be a train_config.json (got %s)\n' "$resume_path" >&2
    exit 1
  fi
  if [[ -f "$resume_path" ]]; then
    resume_config="$resume_path"
  elif [[ -f "$resume_path/train_config.json" ]]; then
    resume_config="$resume_path/train_config.json"
  else
    printf 'resume-from directory has no train_config.json: %s\n' "$resume_path" >&2
    exit 1
  fi
  "$PYTHON_BIN" - "$resume_config" "$BATCH_SIZE" "$GRAD_ACCUM" "$PEAK_LR" "$DECAY_LR" "$WARMUP_STEPS" "$DECAY_STEPS" "$CHECKPOINT_KEEP_LAST" "$CHECKPOINT_KEEP_STEPS" <<'PY'
import json
import math
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
cfg = json.loads(config_path.read_text(encoding="utf-8"))
expected = {
    "batch_size": int(sys.argv[2]),
    "accelerator.gradient_accumulation.steps": int(sys.argv[3]),
    "policy.optimizer_lr": float(sys.argv[4]),
    "policy.scheduler_decay_lr": float(sys.argv[5]),
    "policy.scheduler_warmup_steps": int(sys.argv[6]),
    "policy.scheduler_decay_steps": int(sys.argv[7]),
    "checkpoint_keep_last": int(sys.argv[8]),
    "checkpoint_keep_steps": json.loads(sys.argv[9]),
}

def get_path(payload, dotted):
    value = payload
    for key in dotted.split("."):
        value = value[key]
    return value

errors = []
for key, wanted in expected.items():
    try:
        actual = get_path(cfg, key)
    except (KeyError, TypeError):
        errors.append(f"{key}: missing")
        continue
    if isinstance(wanted, float):
        matches = math.isclose(float(actual), wanted, rel_tol=1e-12, abs_tol=0.0)
    else:
        matches = actual == wanted
    if not matches:
        errors.append(f"{key}: checkpoint={actual!r}, requested={wanted!r}")
if errors:
    raise SystemExit("Resume contract mismatch in " + str(config_path) + ":\n  " + "\n  ".join(errors))
print("Resume contract OK:", config_path)
PY
  RESUME_ARGS=(--resume=true --config_path="$resume_path")
else
  if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$TRAINING_ROOT/outputs/pi05_oreo_fsdp2_formal_$(date +%Y%m%d-%H%M%S)"
  fi
  OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
  [[ ! -e "$OUTPUT_DIR" ]] || { printf 'Output already exists: %s\n' "$OUTPUT_DIR" >&2; exit 1; }
fi

TRAIN_ARGS=(
  --dataset.repo_id="$EXPECTED_DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.video_backend=pyav
  --dataset.return_uint8=true
  --dataset.training_filter_path="$DATASET_ROOT/meta/training_filter.json"
  --dataset.episode_split_path="$SPLIT_PATH"
  --dataset.episode_split_name=train
  --dataset.episode_eval_split_name=val
  --dataset.eval_split=0.05
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
  --policy.optimizer_lr="$PEAK_LR"
  --policy.scheduler_warmup_steps="$WARMUP_STEPS"
  --policy.scheduler_decay_steps="$DECAY_STEPS"
  --policy.scheduler_decay_lr="$DECAY_LR"
  --policy.gradient_checkpointing=true
  --policy.compile_model=false
  --policy.push_to_hub=false
  --parallelism.dp_replicate=1
  --parallelism.dp_shard=4
  --accelerator.mixed_precision=bf16
  --accelerator.gradient_accumulation.steps="$GRAD_ACCUM"
  --accelerator.fsdp.reshard_after_forward=true
  --accelerator.fsdp.cpu_offload=false
  --accelerator.fsdp.wrap_modules='["SiglipEncoderLayer","Linear","GemmaTextScaledWordEmbedding"]'
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --num_workers="$NUM_WORKERS"
  --env_eval_freq=0
  --eval_steps="$EVAL_STEPS"
  --save_checkpoint="$SAVE_CHECKPOINT"
  --save_freq="$SAVE_FREQ"
  --checkpoint_keep_last="$CHECKPOINT_KEEP_LAST"
  --checkpoint_keep_steps="$CHECKPOINT_KEEP_STEPS"
  --checkpoint_format=dcp
  --log_freq=10
  --ema.enable=false
  --wandb.enable=false
  --job_name="r1a7_pi05_fsdp2_formal"
  --seed=1000
)
[[ -n "$OUTPUT_DIR" ]] && TRAIN_ARGS+=(--output_dir="$OUTPUT_DIR")
TRAIN_ARGS+=("${RESUME_ARGS[@]}")

EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))
OPT_UPDATES=$((STEPS / GRAD_ACCUM))
TOTAL_TRAIN_SAMPLES=$((STEPS * BATCH_SIZE * NUM_GPUS))
EQUIVALENT_EPOCHS="$(awk -v samples="$TOTAL_TRAIN_SAMPLES" -v starts="$TRAIN_ALLOWED_STARTS" 'BEGIN { printf "%.2f", samples / starts }')"
COMMAND=("$TORCHRUN_BIN" --standalone --nproc-per-node=$NUM_GPUS "$TRAIN_BIN" "${TRAIN_ARGS[@]}")

printf 'Dataset: %s\nSplit:   %s (sha256 %s...)\nFilter:  %s (sha256 %s...)\n' \
  "$DATASET_ROOT" "$SPLIT_PATH" "${EXPECTED_EPISODE_SPLIT_SHA256:0:16}" \
  "$DATASET_ROOT/meta/training_filter.json" "${EXPECTED_TRAINING_FILTER_SHA256:0:16}"
printf 'Batch:   %s per GPU x %s GPUs x %s accum = effective %s | %s optimizer updates over %s steps\n' \
  "$BATCH_SIZE" "$NUM_GPUS" "$GRAD_ACCUM" "$EFFECTIVE_BATCH" "$OPT_UPDATES" "$STEPS"
printf 'Coverage: %s samples over %s allowed train starts = %s equivalent epochs\n' \
  "$TOTAL_TRAIN_SAMPLES" "$TRAIN_ALLOWED_STARTS" "$EQUIVALENT_EPOCHS"
printf 'Optimizer: peak_lr=%s decay_lr=%s warmup=%s decay=%s\n' \
  "$PEAK_LR" "$DECAY_LR" "$WARMUP_STEPS" "$DECAY_STEPS"
printf 'Checkpoint: save=%s freq=%s keep_last=%s milestones=%s (~40 GiB each) | val every %s steps\n' \
  "$SAVE_CHECKPOINT" "$SAVE_FREQ" "$CHECKPOINT_KEEP_LAST" "$CHECKPOINT_KEEP_STEPS" "$EVAL_STEPS"
printf 'Free disk: %s GiB | Output: %s\n' "$free_gb" "${OUTPUT_DIR:-<from checkpoint>}"
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
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

mkdir -p "$TRAINING_ROOT/logs"
LOG_FILE="$TRAINING_ROOT/logs/$(basename "${OUTPUT_DIR:-resume_$(date +%Y%m%d-%H%M%S)}").log"
exec > >(tee -a "$LOG_FILE") 2>&1
printf '[launcher] logging to %s\n' "$LOG_FILE"

exec "${COMMAND[@]}"
