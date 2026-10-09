#!/usr/bin/env python3
"""Validate an R1-A7 PI0.5 checkpoint and optionally run offline inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file


DEFAULT_DATASET = Path(
    "/home/jieguo/.cache/huggingface/lerobot/local/"
    "r1a7_black_truffle_4cam_30_pi05_stats"
)
EXPECTED_CAMERAS = {
    "observation.images.cam_global": [3, 360, 640],
    "observation.images.cam_left_wrist": [3, 480, 640],
    "observation.images.cam_right_wrist": [3, 480, 640],
    "observation.images.cam_scanner": [3, 480, 640],
}
EXPECTED_JOINTS = [
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
    "kLeftGripper",
    "kRightGripper",
]
EXPECTED_MODEL_TENSOR_COUNT = 812
EXPECTED_STORED_PARAMETER_COUNT = 3_616_757_520
TIED_EMBEDDING_KEY = (
    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
)
TIED_LM_HEAD_KEY = "model.paligemma_with_expert.paligemma.lm_head.weight"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _processor_step(payload: dict[str, Any], registry_name: str) -> dict[str, Any]:
    matches = [step for step in payload.get("steps", []) if step.get("registry_name") == registry_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {registry_name!r} processor step, got {len(matches)}")
    return matches[0]


def _flatten_names(names: Any) -> list[str]:
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"invalid feature names: {names!r}")
    return names


def validate_checkpoint(checkpoint: Path, dataset_root: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing required files: {missing}")

    config = _read_json(checkpoint / "config.json")
    train_config = _read_json(checkpoint / "train_config.json")
    preprocessor = _read_json(checkpoint / "policy_preprocessor.json")
    postprocessor = _read_json(checkpoint / "policy_postprocessor.json")
    info = _read_json(dataset_root / "meta/info.json")
    stats = _read_json(dataset_root / "meta/stats.json")

    if config.get("type") != "pi05":
        raise ValueError(f"expected policy type 'pi05', got {config.get('type')!r}")
    input_features = config.get("input_features", {})
    output_features = config.get("output_features", {})
    actual_cameras = {
        key: value.get("shape")
        for key, value in input_features.items()
        if value.get("type") == "VISUAL"
    }
    if actual_cameras != EXPECTED_CAMERAS:
        raise ValueError(f"four-camera contract mismatch: {actual_cameras}")
    if input_features.get("observation.state", {}).get("shape") != [16]:
        raise ValueError("checkpoint observation.state is not 16D")
    if output_features.get("action", {}).get("shape") != [16]:
        raise ValueError("checkpoint action is not 16D")

    full_finetune = {
        "freeze_vision_encoder": config.get("freeze_vision_encoder"),
        "train_expert_only": config.get("train_expert_only"),
        "use_peft": config.get("use_peft"),
    }
    if full_finetune != {
        "freeze_vision_encoder": False,
        "train_expert_only": False,
        "use_peft": False,
    }:
        raise ValueError(f"checkpoint is not a full-parameter finetune: {full_finetune}")
    if config.get("dtype") != "bfloat16":
        raise ValueError(f"expected bfloat16 checkpoint config, got {config.get('dtype')!r}")
    if not 1 <= int(config.get("n_action_steps", 0)) <= int(config.get("chunk_size", 0)):
        raise ValueError("invalid n_action_steps/chunk_size contract")

    dataset_features = info.get("features", {})
    state_names = _flatten_names(dataset_features.get("observation.state", {}).get("names"))
    action_names = _flatten_names(dataset_features.get("action", {}).get("names"))
    if state_names != EXPECTED_JOINTS or action_names != EXPECTED_JOINTS:
        raise ValueError("dataset state/action joint ordering does not match R1-A7")
    if config.get("action_feature_names") != EXPECTED_JOINTS:
        raise ValueError("checkpoint action_feature_names does not match R1-A7")

    tokenizer_step = _processor_step(preprocessor, "tokenizer_processor")
    tokenizer_name = tokenizer_step.get("config", {}).get("tokenizer_name")
    tokenizer_path = checkpoint / tokenizer_name if tokenizer_name else None
    if tokenizer_path is None or not (tokenizer_path / "tokenizer.json").is_file():
        raise FileNotFoundError(f"checkpoint-local tokenizer is missing: {tokenizer_path}")

    normalizer_step = _processor_step(preprocessor, "normalizer_processor")
    unnormalizer_step = _processor_step(postprocessor, "unnormalizer_processor")
    processor_features = normalizer_step.get("config", {}).get("features", {})
    if processor_features != {**input_features, **output_features}:
        raise ValueError("preprocessor features differ from checkpoint policy features")

    max_stats_error = 0.0
    for step in (normalizer_step, unnormalizer_step):
        state_file = step.get("state_file")
        if not state_file or not (checkpoint / state_file).is_file():
            raise FileNotFoundError(f"processor state file is missing: {state_file!r}")
        tensors = load_file(checkpoint / state_file)
        for feature in ("observation.state", "action"):
            for statistic in ("q01", "q99"):
                key = f"{feature}.{statistic}"
                actual = tensors[key].cpu().numpy().astype(np.float64)
                expected = np.asarray(stats[feature][statistic], dtype=np.float64)
                error = float(np.max(np.abs(actual - expected)))
                max_stats_error = max(max_stats_error, error)
    if max_stats_error > 1e-6:
        raise ValueError(f"checkpoint processor quantiles differ from dataset by {max_stats_error:.6g}")

    model_path = checkpoint / "model.safetensors"
    with safe_open(model_path, framework="pt", device="cpu") as tensors:
        tensor_names = list(tensors.keys())
        tensor_count = len(tensor_names)
        stored_parameter_count = sum(
            int(np.prod(tensors.get_slice(name).get_shape(), dtype=np.int64))
            for name in tensor_names
        )
        safetensors_metadata = tensors.metadata() or {}
    if tensor_count != EXPECTED_MODEL_TENSOR_COUNT:
        raise ValueError(
            f"expected {EXPECTED_MODEL_TENSOR_COUNT} stored model tensors, got {tensor_count}"
        )
    if stored_parameter_count != EXPECTED_STORED_PARAMETER_COUNT:
        raise ValueError(
            "stored model parameter count mismatch: "
            f"expected {EXPECTED_STORED_PARAMETER_COUNT}, got {stored_parameter_count}"
        )
    tied_alias = safetensors_metadata.get(TIED_EMBEDDING_KEY)
    if tied_alias != TIED_LM_HEAD_KEY or TIED_LM_HEAD_KEY not in tensor_names:
        raise ValueError(
            "PaliGemma input/output embeddings are not stored as tied weights: "
            f"{TIED_EMBEDDING_KEY!r} -> {tied_alias!r}"
        )

    train_policy = train_config.get("policy", {})
    if train_policy.get("input_features") != input_features or train_policy.get("output_features") != output_features:
        raise ValueError("train_config policy features differ from config.json")

    return {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "tensor_count": tensor_count,
        "stored_parameter_count": stored_parameter_count,
        "model_bytes": model_path.stat().st_size,
        "tied_embedding_alias": tied_alias,
        "camera_keys": sorted(actual_cameras),
        "state_dim": 16,
        "action_dim": 16,
        "n_action_steps": config["n_action_steps"],
        "max_processor_stats_error": max_stats_error,
        "full_parameter_finetune": True,
    }


def run_inference(
    checkpoint: Path,
    dataset_root: Path,
    frame_index: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    # These imports are intentionally local. This path uses only LeRobot, the dataset, and the GPU.
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_policy, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(
        checkpoint,
        cli_overrides=[f"--device={device}", "--gradient_checkpointing=false"],
    )
    config.pretrained_path = checkpoint
    dataset = LeRobotDataset(
        "local/r1a7_black_truffle_4cam_30_pi05_stats",
        root=dataset_root,
        video_backend="pyav",
    )
    if not 0 <= frame_index < len(dataset):
        raise IndexError(f"frame index {frame_index} is outside [0, {len(dataset)})")
    policy = make_policy(cfg=config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    sample = dataset[frame_index]
    observation = {
        key: sample[key]
        for key in (*dataset.meta.camera_keys, "observation.state", "task")
    }
    policy.reset()
    torch.manual_seed(seed)
    with torch.inference_mode():
        action = postprocessor(policy.select_action(preprocessor(observation)))
    if tuple(action.shape) != (1, 16) or not torch.isfinite(action).all():
        raise RuntimeError(f"invalid offline action: shape={tuple(action.shape)}")
    return {
        "frame_index": frame_index,
        "seed": seed,
        "action_shape": list(action.shape),
        "action_min": float(action.min()),
        "action_max": float(action.max()),
        "action": action.squeeze(0).detach().cpu().tolist(),
        "hardware_commands_sent": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    result = validate_checkpoint(checkpoint, dataset_root)
    if args.run_inference:
        result["offline_inference"] = run_inference(
            checkpoint, dataset_root, args.frame_index, args.device, args.seed
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
