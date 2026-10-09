#!/usr/bin/env bash
set -Eeuo pipefail

# A100 GPU 3, full-parameter PI0.5 training on the 80 standard Oreo trajectories.
# All 160 scan/place episodes are used for training. There is no validation split.

usage() {
  cat <<'EOF'
Usage: train-pi05-1xa100-v2-standard.sh check|probe|train|resume CHECKPOINT_TRAIN_CONFIG

  check   Verify data, source and GPU availability without starting training.
  probe   Run two full-parameter updates at batch size 24, without saving a checkpoint.
  train   Run 8000 updates at batch size 24; save at 2000, 4000, 6000, 8000.
  resume  Continue this run from a saved checkpoint's train_config.json.

Optional: R1A7_A100_ROOT=/data/project, R1A7_A100_OUTPUT_DIR=<new run directory>
EOF
}

mode="${1:-}"
case "$mode" in
  check|probe|train) (($# == 1)) || { usage >&2; exit 2; } ;;
  resume) (($# == 2)) || { usage >&2; exit 2; } ;;
  *) usage >&2; exit 2 ;;
esac

root="${R1A7_A100_ROOT:-/data/project}"
dataset="$root/datasets/r1a7_oreo_4cam_16d_v2_80_standard_train"
repo_id="local/r1a7_oreo_4cam_16d_v2_80_standard_train"
source="$root/lerobot"
python_bin="$root/envs/r1a7-pi05/bin/python"
train_bin="$root/envs/r1a7-pi05/bin/lerobot-train"
base_model="/data/hy20260424/lerobot/pi05_base"
tokenizer="/data/hy20260424/paligemma-3b-pt-224"
gpu=3

for file in \
  "$dataset/meta/info.json" "$dataset/meta/stats.json" \
  "$dataset/meta/source_selection.json" \
  "$base_model/config.json" "$base_model/model.safetensors" \
  "$tokenizer/tokenizer.json" "$tokenizer/tokenizer.model"; do
  [[ -f "$file" ]] || { printf 'Missing file: %s\n' "$file" >&2; exit 1; }
done
[[ -x "$python_bin" && -x "$train_bin" ]] || {
  printf 'Missing PI0.5 environment under %s/envs/r1a7-pi05\n' "$root" >&2; exit 1;
}
[[ "$(git -C "$source" rev-parse HEAD)" == 262d2fd07e32224d46dca8ba8bcdef5a0b3a9c3b ]] || {
  printf 'Unexpected LeRobot checkout under %s\n' "$source" >&2; exit 1;
}

"$python_bin" - "$dataset" "$source" <<'PY'
import json
import sys
from pathlib import Path

import lerobot

root, source = map(Path, sys.argv[1:])
assert Path(lerobot.__file__).resolve().is_relative_to(source.resolve()), lerobot.__file__
info = json.loads((root / "meta/info.json").read_text())
selection = json.loads((root / "meta/source_selection.json").read_text())
stats = json.loads((root / "meta/stats.json").read_text())
assert info["codebase_version"] == "v3.0"
assert (info["total_episodes"], info["total_frames"], info["fps"]) == (160, 25928, 15)
assert info["features"]["observation.state"]["shape"] == [16]
assert info["features"]["action"]["shape"] == [16]
assert len([k for k in info["features"] if k.startswith("observation.images.")]) == 4
assert selection["selection"] == {
    "episode_type": "standard_pick", "collection_ordinal_inclusive": [1, 80]
}
assert (selection["source_episodes"], selection["lerobot_episodes"]) == (80, 160)
assert stats["observation.state"]["count"] == [25928]
assert stats["action"]["count"] == [25928]
print("Dataset contract OK: 80 source trajectories, 160 episodes, 25928 frames, four cameras, 16D")
PY

gpu_line="$(nvidia-smi -i "$gpu" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits)"
IFS=',' read -r gpu_total gpu_used <<<"$gpu_line"
gpu_total="${gpu_total// /}"
gpu_used="${gpu_used// /}"
((gpu_total >= 40000 && gpu_used < 2048)) || {
  printf 'GPU %s unavailable: %s/%s MiB allocated\n' "$gpu" "$gpu_used" "$gpu_total" >&2; exit 1;
}
printf 'Physical GPU %s available: %s/%s MiB allocated\n' "$gpu" "$gpu_used" "$gpu_total"

export CUDA_VISIBLE_DEVICES="$gpu"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "$mode" == check ]]; then
  exit 0
fi

if [[ "$mode" == resume ]]; then
  checkpoint_config="$(realpath -- "$2")"
  [[ "$(basename -- "$checkpoint_config")" == train_config.json && -f "$checkpoint_config" ]] || {
    printf 'Expected a checkpoint train_config.json: %s\n' "$checkpoint_config" >&2; exit 1;
  }
  "$python_bin" - "$checkpoint_config" "$dataset" "$repo_id" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
assert config["dataset"]["root"] == sys.argv[2]
assert config["dataset"]["repo_id"] == sys.argv[3]
assert config["dataset"]["eval_split"] == 0.0
assert config["policy"]["train_expert_only"] is False
assert config["policy"]["freeze_vision_encoder"] is False
assert config["batch_size"] == 24
assert config["steps"] == 8000
print("Resume contract OK: v2 standard dataset, all episodes, full parameters")
PY
  exec "$train_bin" --resume=true --config_path="$checkpoint_config"
fi

if [[ "$mode" == probe ]]; then
  steps=2
  workers=0
  save_checkpoint=false
  save_freq=0
  log_freq=1
  default_output="$root/outputs/pi05_oreo_v2_standard_gpu3_probe_$(date +%Y%m%d-%H%M%S)"
else
  steps=8000
  workers=4
  save_checkpoint=true
  save_freq=2000
  log_freq=100
  default_output="$root/outputs/pi05_oreo_v2_standard_a100_gpu3_full_b24_lr1e5_8k_01"
fi
output="${R1A7_A100_OUTPUT_DIR:-$default_output}"
[[ ! -e "$output" ]] || { printf 'Output directory already exists: %s\n' "$output" >&2; exit 1; }
printf 'Mode=%s, GPU=3, batch=24, steps=%s, peak LR=1e-5, eval=off, output=%s\n' \
  "$mode" "$steps" "$output"
if [[ "${R1A7_A100_DRY_RUN:-0}" == 1 ]]; then
  printf 'Dry run complete; training was not started.\n'
  exit 0
fi

exec "$train_bin" \
  --dataset.repo_id="$repo_id" \
  --dataset.root="$dataset" \
  --dataset.video_backend=pyav \
  --dataset.return_uint8=true \
  --dataset.eval_split=0 \
  --policy.type=pi05 \
  --policy.pretrained_path="$base_model" \
  --policy.input_features=null \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.text_tokenizer_name="$tokenizer" \
  --policy.n_action_steps=10 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.use_peft=false \
  --policy.optimizer_lr=1e-5 \
  --policy.scheduler_warmup_steps=250 \
  --policy.scheduler_decay_steps=8000 \
  --policy.scheduler_decay_lr=1e-6 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --parallelism.dp_replicate=1 \
  --parallelism.dp_shard=1 \
  --accelerator.mixed_precision=bf16 \
  --accelerator.gradient_accumulation.steps=1 \
  --batch_size=24 \
  --steps="$steps" \
  --num_workers="$workers" \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --save_checkpoint="$save_checkpoint" \
  --save_freq="$save_freq" \
  --checkpoint_keep_last=4 \
  --checkpoint_keep_steps='[]' \
  --checkpoint_format=safetensors \
  --log_freq="$log_freq" \
  --ema.enable=false \
  --wandb.enable=false \
  --job_name="r1a7_pi05_a100_gpu3_v2_standard_${mode}" \
  --seed=1000 \
  --output_dir="$output"
