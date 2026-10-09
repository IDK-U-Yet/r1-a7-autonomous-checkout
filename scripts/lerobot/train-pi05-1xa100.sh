#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: train-pi05-1xa100.sh probe|train|resume PATH

probe       Eight micro-steps, one full-parameter optimizer update; no checkpoint.
train       Full PI0.5 fine-tune: 192000 micro-steps / 8 accumulation = 24000 updates.
resume PATH Resume an existing run from its checkpoint train_config.json.

Environment overrides:
  R1A7_A100_ROOT         Default /data/project
  R1A7_A100_GPU          Physical GPU index; default 3
  R1A7_A100_OUTPUT_DIR   Optional new output directory for probe/train
  R1A7_A100_PROBE_BATCH  Per-GPU batch size for a probe; default 1
  R1A7_A100_PROBE_ACCUM  Gradient accumulation for a probe; default 8
  R1A7_A100_DRY_RUN      Set to 1 for preflight only
EOF
}

mode="${1:-}"
case "$mode" in
  probe|train) (($# == 1)) || { usage >&2; exit 2; } ;;
  resume) (($# == 2)) || { usage >&2; exit 2; } ;;
  *) usage >&2; exit 2 ;;
esac

root="${R1A7_A100_ROOT:-/data/project}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gpu="${R1A7_A100_GPU:-3}"
python_bin="$root/envs/r1a7-pi05/bin/python"
train_bin="$root/envs/r1a7-pi05/bin/lerobot-train"
source_root="$root/lerobot"
dataset_root="$root/datasets/r1a7_oreo_4cam_16d_v1_80_pi05_train_stats"
base_root="/data/hy20260424/lerobot/pi05_base"
tokenizer_root="/data/hy20260424/paligemma-3b-pt-224"
expected_commit="262d2fd07e32224d46dca8ba8bcdef5a0b3a9c3b"
expected_split_sha="53c12e7768cf53db0201641ba77147590dcb546a9130baf7d15b5fc87efe487b"
expected_filter_sha="589182c4a4aa4ab0236417fea30c8c232570bf4a13a90e5e35f0503809c946b7"
expected_stats_sha="c6438fe1acaa1c0bbd56feb3e7c93160a20bbeb1b9ffb3847ed4bcce52019877"

[[ "$gpu" =~ ^[0-9]+$ ]] || { printf 'Invalid GPU index: %s\n' "$gpu" >&2; exit 2; }
for executable in "$python_bin" "$train_bin"; do
  [[ -x "$executable" ]] || { printf 'Missing executable: %s\n' "$executable" >&2; exit 1; }
done
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$expected_commit" ]] || {
  printf 'LeRobot checkout is not at pinned commit %s\n' "$expected_commit" >&2; exit 1;
}
train_entry="$source_root/src/lerobot/scripts/lerobot_train.py"
tokenizer_override_marker='preprocessor_overrides["tokenizer_processor"]'
if ! grep -Fq "$tokenizer_override_marker" "$train_entry"; then
  git -C "$source_root" apply "$script_dir/patches/a100-local-tokenizer-override.patch"
fi
for required in \
  "$dataset_root/meta/info.json" "$dataset_root/meta/stats.json" \
  "$dataset_root/meta/episode_split.json" "$dataset_root/meta/training_filter.json" \
  "$base_root/config.json" "$base_root/model.safetensors" \
  "$tokenizer_root/tokenizer.json" "$tokenizer_root/tokenizer.model"; do
  [[ -f "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
[[ "$(sha256sum "$dataset_root/meta/episode_split.json" | cut -d' ' -f1)" == "$expected_split_sha" ]] || {
  printf 'Episode split SHA-256 mismatch\n' >&2; exit 1;
}
[[ "$(sha256sum "$dataset_root/meta/training_filter.json" | cut -d' ' -f1)" == "$expected_filter_sha" ]] || {
  printf 'Training filter SHA-256 mismatch\n' >&2; exit 1;
}
[[ "$(sha256sum "$dataset_root/meta/stats.json" | cut -d' ' -f1)" == "$expected_stats_sha" ]] || {
  printf 'Train-only stats SHA-256 mismatch\n' >&2; exit 1;
}

"$python_bin" - "$dataset_root" "$source_root" <<'PY'
import json
import sys
from pathlib import Path

import av
import draccus
import lerobot

root, source = map(Path, sys.argv[1:])
assert Path(lerobot.__file__).resolve().is_relative_to(source.resolve()), lerobot.__file__
assert av.__version__ == "15.1.0", av.__version__
info = json.loads((root / "meta/info.json").read_text())
split = json.loads((root / "meta/episode_split.json").read_text())
stats = json.loads((root / "meta/stats.json").read_text())
audit = json.loads((root / "meta/r1a7_exact_stats_audit.json").read_text())
assert info["codebase_version"] == "v3.0"
assert (info["total_episodes"], info["total_frames"], info["fps"]) == (160, 36630, 15)
assert info["features"]["observation.state"]["shape"] == [16]
assert info["features"]["action"]["shape"] == [16]
assert len([key for key in info["features"] if key.startswith("observation.images.")]) == 4
assert [split["splits"][name]["allowed_training_starts"] for name in ("train", "val", "test")] == [31930, 1924, 1905]
assert stats["observation.state"]["count"] == stats["action"]["count"] == [31930]
assert audit["sample_selection"]["episode_split_name"] == "train"
print("Dataset/source contract OK: four cameras, 16D, train-only stats, pinned split")
PY

gpu_line="$(nvidia-smi -i "$gpu" --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits)"
IFS=',' read -r gpu_index gpu_total gpu_used <<<"$gpu_line"
gpu_total="${gpu_total// /}"
gpu_used="${gpu_used// /}"
((gpu_total >= 40000)) || { printf 'GPU %s has only %s MiB, need an A100 40GB\n' "$gpu" "$gpu_total" >&2; exit 1; }
((gpu_used < 2048)) || { printf 'GPU %s is already using %s MiB\n' "$gpu" "$gpu_used" >&2; exit 1; }
printf 'Using physical GPU %s: %s/%s MiB currently allocated\n' "$gpu_index" "$gpu_used" "$gpu_total"

export CUDA_VISIBLE_DEVICES="$gpu"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

if [[ "$mode" == resume ]]; then
  config_path="$(realpath -- "$2")"
  [[ "$(basename -- "$config_path")" == train_config.json && -f "$config_path" ]] || {
    printf 'Resume requires a checkpoint train_config.json, got %s\n' "$config_path" >&2; exit 1;
  }
  printf 'Resuming from %s\n' "$config_path"
  if [[ "${R1A7_A100_DRY_RUN:-0}" == 1 ]]; then
    printf 'Dry run complete; training was not started.\n'
    exit 0
  fi
  exec "$train_bin" \
    --resume=true --config_path="$config_path"
fi

if [[ "$mode" == probe ]]; then
  batch_size="${R1A7_A100_PROBE_BATCH:-1}"
  grad_accum="${R1A7_A100_PROBE_ACCUM:-8}"
  [[ "$batch_size" =~ ^[1-9][0-9]*$ && "$grad_accum" =~ ^[1-9][0-9]*$ ]] || {
    printf 'Probe batch and accumulation must be positive integers\n' >&2; exit 2;
  }
  steps="$grad_accum"
  eval_steps=0
  eval_batch_size=1
  save_freq=0
  save_checkpoint=false
  num_workers=0
  log_freq=1
  scheduler_warmup_steps=1000
  scheduler_decay_steps=24000
  default_output="$root/outputs/pi05_oreo_a100_probe_$(date +%Y%m%d-%H%M%S)"
else
  batch_size=32
  grad_accum=1
  steps=6000
  eval_steps=1000
  eval_batch_size=1
  save_freq=2000
  save_checkpoint=true
  num_workers=4
  log_freq=100
  scheduler_warmup_steps=250
  scheduler_decay_steps=6000
  default_output="$root/outputs/pi05_oreo_a100_full_01"
fi
output_dir="${R1A7_A100_OUTPUT_DIR:-$default_output}"
[[ ! -e "$output_dir" ]] || { printf 'Output already exists: %s\n' "$output_dir" >&2; exit 1; }

printf 'Mode=%s, micro-steps=%s, optimizer updates=%s, per-GPU batch=%s, accumulation=%s, effective batch=%s, output=%s\n' \
  "$mode" "$steps" "$((steps / grad_accum))" "$batch_size" "$grad_accum" "$((batch_size * grad_accum))" "$output_dir"
if [[ "${R1A7_A100_DRY_RUN:-0}" == 1 ]]; then
  printf 'Dry run complete; training was not started.\n'
  exit 0
fi
exec "$train_bin" \
  --dataset.repo_id=local/r1a7_oreo_4cam_16d_v1_80_pi05_train_stats \
  --dataset.root="$dataset_root" \
  --dataset.video_backend=pyav \
  --dataset.return_uint8=true \
  --dataset.training_filter_path="$dataset_root/meta/training_filter.json" \
  --dataset.episode_split_path="$dataset_root/meta/episode_split.json" \
  --dataset.episode_split_name=train \
  --dataset.episode_eval_split_name=val \
  --dataset.eval_split=0.05 \
  --policy.type=pi05 \
  --policy.pretrained_path="$base_root" \
  --policy.input_features=null \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.text_tokenizer_name="$tokenizer_root" \
  --policy.n_action_steps=10 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.use_peft=false \
  --policy.optimizer_lr=2.5e-5 \
  --policy.scheduler_warmup_steps="$scheduler_warmup_steps" \
  --policy.scheduler_decay_steps="$scheduler_decay_steps" \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --parallelism.dp_replicate=1 \
  --parallelism.dp_shard=1 \
  --accelerator.mixed_precision=bf16 \
  --accelerator.gradient_accumulation.steps="$grad_accum" \
  --batch_size="$batch_size" \
  --steps="$steps" \
  --num_workers="$num_workers" \
  --env_eval_freq=0 \
  --eval_steps="$eval_steps" \
  --eval.batch_size="$eval_batch_size" \
  --save_checkpoint="$save_checkpoint" \
  --save_freq="$save_freq" \
  --checkpoint_keep_last=3 \
  --checkpoint_keep_steps='[]' \
  --checkpoint_format=safetensors \
  --log_freq="$log_freq" \
  --ema.enable=false \
  --wandb.enable=false \
  --job_name="r1a7_pi05_a100_${mode}" \
  --seed=1000 \
  --output_dir="$output_dir"
