#!/usr/bin/env bash
set -Eeuo pipefail

# Same full-parameter PI0.5 baseline as v2/80, with the merged 160 raw trajectories.
mode="${1:-}"
[[ "$mode" == check || "$mode" == train ]] || {
  printf 'Usage: %s check|train\n' "$0" >&2; exit 2;
}
root="${R1A7_A100_ROOT:-/data/project}"
dataset="$root/datasets/r1a7_oreo_4cam_16d_v2v3_standard160_train"
repo_id="local/r1a7_oreo_4cam_16d_v2v3_standard160_train"
source="$root/lerobot"
python_bin="$root/envs/r1a7-pi05/bin/python"
train_bin="$root/envs/r1a7-pi05/bin/lerobot-train"
base_model="/data/hy20260424/lerobot/pi05_base"
tokenizer="/data/hy20260424/paligemma-3b-pt-224"
output="${R1A7_A100_OUTPUT_DIR:-$root/outputs/pi05_oreo_v2v3_standard160_a100_gpu3_full_b24_lr1e5_8k_01}"

[[ "$(git -C "$source" rev-parse HEAD)" == 262d2fd07e32224d46dca8ba8bcdef5a0b3a9c3b ]]
for file in "$dataset/meta/source_selection.json" "$base_model/model.safetensors" "$tokenizer/tokenizer.model"; do
  [[ -f "$file" ]] || { printf 'Missing: %s\n' "$file" >&2; exit 1; }
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$python_bin" - "$dataset" "$repo_id" "$source" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

import lerobot
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root, repo_id, source = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
assert importlib.metadata.version("lerobot") == "0.6.2"
assert Path(lerobot.__file__).resolve().is_relative_to(source.resolve())
info = json.loads((root / "meta/info.json").read_text())
selection = json.loads((root / "meta/source_selection.json").read_text())
stats = json.loads((root / "meta/stats.json").read_text())
assert info["codebase_version"] == "v3.0"
assert (info["total_episodes"], info["fps"], info["splits"]) == (320, 15, {"train": "0:320"})
assert (selection["source_episodes"], selection["lerobot_episodes"]) == (160, 320)
assert selection["selection"] == {"episode_type": "standard_pick", "source_batch_counts": {"v2": 80, "v3": 80}}
for key in ("observation.state", "action"):
    assert info["features"][key]["shape"] == [16]
    assert stats[key]["count"] == [info["total_frames"]]
    assert np.isfinite(stats[key]["q01"]).all() and np.isfinite(stats[key]["q99"]).all()
assert sum(v["dtype"] == "video" for v in info["features"].values()) == 4
assert sum(pq.ParquetFile(p).metadata.num_rows for p in (root / "data").rglob("*.parquet")) == info["total_frames"]
dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
assert len(dataset) == info["total_frames"]
for index in (0, info["total_frames"] // 2, info["total_frames"] - 1):
    sample = dataset[index]
    assert sample["observation.state"].shape == (16,) and sample["action"].shape == (16,)
    for key, feature in info["features"].items():
        if feature["dtype"] == "video":
            assert sample[key].shape[0] == 3, (index, key)
print(f"Dataset OK: 160 raw trajectories, 320 episodes, {len(dataset)} frames, 4 cameras, 16D", flush=True)
PY

gpu_line="$(nvidia-smi -i 3 --query-gpu=memory.total,memory.used --format=csv,noheader,nounits)"
IFS=',' read -r gpu_total gpu_used <<<"$gpu_line"
gpu_total="${gpu_total// /}"; gpu_used="${gpu_used// /}"
((gpu_total >= 40000 && gpu_used < 2048)) || {
  printf 'GPU 3 busy: %s/%s MiB\n' "$gpu_used" "$gpu_total" >&2; exit 1;
}
[[ "$mode" == train ]] || exit 0
[[ ! -e "$output" ]] || { printf 'Output exists: %s\n' "$output" >&2; exit 1; }
printf 'Starting full PI0.5: GPU=3, batch=24, steps=8000, lr=1e-5, output=%s\n' "$output"
exec "$train_bin" \
  --dataset.repo_id="$repo_id" --dataset.root="$dataset" \
  --dataset.video_backend=pyav --dataset.return_uint8=true --dataset.eval_split=0 \
  --policy.type=pi05 --policy.pretrained_path="$base_model" --policy.input_features=null \
  --policy.device=cuda --policy.dtype=bfloat16 --policy.text_tokenizer_name="$tokenizer" \
  --policy.n_action_steps=10 --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --policy.use_peft=false --policy.optimizer_lr=1e-5 \
  --policy.scheduler_warmup_steps=250 --policy.scheduler_decay_steps=8000 --policy.scheduler_decay_lr=1e-6 \
  --policy.gradient_checkpointing=true --policy.compile_model=false --policy.push_to_hub=false \
  --parallelism.dp_replicate=1 --parallelism.dp_shard=1 \
  --accelerator.mixed_precision=bf16 --accelerator.gradient_accumulation.steps=1 \
  --batch_size=24 --steps=8000 --num_workers=4 --env_eval_freq=0 --eval_steps=0 \
  --save_checkpoint=true --save_freq=2000 --checkpoint_keep_last=4 --checkpoint_keep_steps='[]' \
  --checkpoint_format=safetensors --log_freq=100 --ema.enable=false --wandb.enable=false \
  --job_name=r1a7_pi05_a100_gpu3_v2v3_standard160 --seed=1000 --output_dir="$output"
