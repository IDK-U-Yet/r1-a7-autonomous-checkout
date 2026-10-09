#!/usr/bin/env python3
"""Run read-only PI0.5 inference on recorded R1-A7 LeRobot episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "runtime/checkpoints/pi05_oreo_a100_b24_6ep_02/checkpoints/004000/pretrained_model"
)
DEFAULT_DATASET = (
    Path.home()
    / ".cache/huggingface/lerobot/local/r1a7_oreo_4cam_16d_v1_80_pi05_stats"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--repo-id", help="LeRobot dataset ID; defaults to local/<dataset directory name>")
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--frame-offset", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="Check dataset samples without loading model")
    args = parser.parse_args()
    if any(index < 0 for index in args.episodes) or len(set(args.episodes)) != len(args.episodes):
        parser.error("--episodes must contain unique non-negative episode indices")
    if args.frame_offset < 0:
        parser.error("--frame-offset must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "meta/info.json").is_file():
        raise FileNotFoundError(f"LeRobot dataset not found: {dataset_root}")
    if not args.dry_run:
        weights_file = (
            "adapter_model.safetensors"
            if (checkpoint / "adapter_config.json").is_file()
            else "model.safetensors"
        )
        required = (
            "config.json",
            weights_file,
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Checkpoint is incomplete at {checkpoint}: {missing}")

    repo_id = args.repo_id or f"local/{dataset_root.name}"
    dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav")
    camera_keys = tuple(dataset.meta.camera_keys)
    if len(camera_keys) != 4:
        raise ValueError(f"Expected four cameras, got {camera_keys}")

    selected = []
    for episode_index in args.episodes:
        if episode_index >= len(dataset.meta.episodes):
            raise IndexError(f"Episode {episode_index} is absent from {dataset_root}")
        episode = dataset.meta.episodes[episode_index]
        if args.frame_offset >= episode["length"]:
            raise IndexError(
                f"Episode {episode_index} has {episode['length']} frames; "
                f"offset {args.frame_offset} is outside it"
            )
        frame_index = episode["dataset_from_index"] + args.frame_offset
        sample = dataset[frame_index]
        if sample["task"] not in episode["tasks"]:
            raise ValueError(f"Task mismatch for episode {episode_index}: {sample['task']}")
        if tuple(sample["observation.state"].shape) != (16,):
            raise ValueError(f"Episode {episode_index} does not have 16D state")
        selected.append((episode_index, frame_index, sample))

    if args.dry_run:
        for episode_index, frame_index, sample in selected:
            print(json.dumps({
                "episode_index": episode_index,
                "frame_index": frame_index,
                "task": sample["task"],
                "camera_keys": camera_keys,
                "state_dim": 16,
                "model_loaded": False,
                "hardware_commands_sent": False,
            }, ensure_ascii=False))
        return

    config = PreTrainedConfig.from_pretrained(
        checkpoint,
        cli_overrides=[f"--device={args.device}", "--gradient_checkpointing=false"],
    )
    if config.type != "pi05":
        raise ValueError(f"Expected PI0.5 checkpoint, got {config.type}")
    if tuple(config.input_features["observation.state"].shape) != (16,):
        raise ValueError("Checkpoint observation.state is not 16D")
    if tuple(config.output_features["action"].shape) != (16,):
        raise ValueError("Checkpoint action is not 16D")
    if set(camera_keys) != {
        key for key in config.input_features if key.startswith("observation.images.")
    }:
        raise ValueError("Checkpoint and dataset camera keys differ")

    config.pretrained_path = checkpoint
    policy = make_policy(cfg=config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    policy.eval()
    torch.manual_seed(args.seed)
    for episode_index, frame_index, sample in selected:
        observation = {
            key: sample[key]
            for key in (*camera_keys, "observation.state", "task")
        }
        policy.reset()
        with torch.inference_mode():
            action = postprocessor(policy.select_action(preprocessor(observation)))
        if tuple(action.shape) != (1, 16) or not torch.isfinite(action).all():
            raise RuntimeError(f"Invalid action for episode {episode_index}: {tuple(action.shape)}")
        print(json.dumps({
            "episode_index": episode_index,
            "frame_index": frame_index,
            "task": sample["task"],
            "action_shape": list(action.shape),
            "action": action.squeeze(0).detach().cpu().tolist(),
            "hardware_commands_sent": False,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
