# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Offline R1-A7 inference against a frame from a local LeRobot v3 dataset.

This command has no real-robot mode and never imports Unitree/xr-teleoperate
hardware modules.  ``--dry-run`` validates a dataset frame and prints its
recorded reference action without loading a policy.
"""

from __future__ import annotations

import argparse
import importlib
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from unitree_lerobot.r1a7_contract import (
    CAMERA_KEYS,
    ROBOT_TYPE,
    R1A7Action,
    R1A7ContractError,
    decode_action,
    load_dataset_info,
    validate_state_vector,
)


@dataclass(frozen=True)
class DatasetFrame:
    episode: int
    frame: int
    global_index: int
    task_index: int
    task: str
    state: np.ndarray
    reference_action: np.ndarray


def _load_task(dataset_root: Path, task_index: int) -> str:
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read a local LeRobot v3 dataset"
        ) from exc

    task_path = dataset_root / "meta" / "tasks.parquet"
    if not task_path.is_file():
        raise R1A7ContractError(f"task metadata does not exist: {task_path}")
    table = pq.read_table(task_path)
    if "task_index" not in table.column_names:
        raise R1A7ContractError("tasks.parquet does not contain task_index")
    matches = table.filter(pc.equal(table["task_index"], task_index))
    if matches.num_rows != 1:
        raise R1A7ContractError(
            f"task_index {task_index} matched {matches.num_rows} task rows, expected one"
        )
    row = matches.to_pylist()[0]
    for key in ("task", "__index_level_0__"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    string_values = [value for value in row.values() if isinstance(value, str) and value]
    if len(string_values) == 1:
        return string_values[0]
    raise R1A7ContractError(
        f"cannot identify task text for task_index {task_index} in {task_path}"
    )


def load_local_dataset_frame(
    dataset_root: str | Path,
    episode: int,
    frame: int,
    *,
    task_override: str | None = None,
) -> DatasetFrame:
    """Read one non-image frame directly from local v3 parquet files."""

    if episode < 0 or frame < 0:
        raise R1A7ContractError("episode and frame must be non-negative")
    root = Path(dataset_root).expanduser().resolve()
    load_dataset_info(root)

    try:
        import pyarrow.dataset as pads
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read a local LeRobot v3 dataset"
        ) from exc

    data_dir = root / "data"
    if not data_dir.is_dir():
        raise R1A7ContractError(f"dataset data directory does not exist: {data_dir}")
    dataset = pads.dataset(data_dir, format="parquet")
    required_columns = {
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "index",
        "task_index",
    }
    missing_columns = required_columns - set(dataset.schema.names)
    if missing_columns:
        raise R1A7ContractError(
            f"dataset parquet is missing columns: {sorted(missing_columns)}"
        )
    selected = dataset.to_table(
        columns=sorted(required_columns),
        filter=(pads.field("episode_index") == episode)
        & (pads.field("frame_index") == frame),
    )
    if selected.num_rows != 1:
        raise R1A7ContractError(
            f"episode={episode}, frame={frame} matched {selected.num_rows} rows, expected one"
        )
    row = selected.to_pylist()[0]
    state = validate_state_vector(row["observation.state"])
    reference_action = decode_action(row["action"]).as_vector()
    task_index = int(row["task_index"])
    task = task_override if task_override is not None else _load_task(root, task_index)
    if not isinstance(task, str) or not task.strip():
        raise R1A7ContractError("task must be a non-empty string")
    return DatasetFrame(
        episode=episode,
        frame=frame,
        global_index=int(row["index"]),
        task_index=task_index,
        task=task,
        state=state,
        reference_action=reference_action,
    )


def _policy_action(
    dataset_root: Path,
    frame: DatasetFrame,
    policy_path: Path,
    tokenizer_path: Path | None,
    device_name: str,
    seed: int,
) -> np.ndarray:
    """Load LeRobot/policy dependencies only for actual shadow inference."""

    if not policy_path.is_dir():
        raise FileNotFoundError(f"policy directory does not exist: {policy_path}")
    required_policy_files = ("config.json", "model.safetensors")
    missing = [name for name in required_policy_files if not (policy_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"policy directory {policy_path} is missing required files: {missing}"
        )

    try:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import make_policy, make_pre_post_processors
    except ImportError as exc:
        raise RuntimeError(
            "shadow policy inference dependencies are unavailable in this environment; "
            "install the old LeRobot policy extras (including its pinned transformers build)"
        ) from exc

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    policy_config = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_config.device = str(device)
    policy_config.pretrained_path = str(policy_path)
    if tokenizer_path is not None:
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"tokenizer directory does not exist: {tokenizer_path}")
        policy_config.tokenizer_name = str(tokenizer_path)
    dataset = LeRobotDataset(
        repo_id=dataset_root.name,
        root=dataset_root,
        video_backend="pyav",
    )
    if frame.global_index < 0 or frame.global_index >= len(dataset):
        raise R1A7ContractError(
            f"global frame index {frame.global_index} is outside dataset length {len(dataset)}"
        )
    step = dataset[frame.global_index]
    step_episode = int(step["episode_index"].item())
    step_frame = int(step["frame_index"].item())
    if (step_episode, step_frame) != (frame.episode, frame.frame):
        raise R1A7ContractError(
            "dataset index mismatch while loading images: "
            f"expected {(frame.episode, frame.frame)}, got {(step_episode, step_frame)}"
        )

    policy = make_policy(cfg=policy_config, ds_meta=dataset.meta)
    policy.eval()
    if policy_config.type == "pi05":
        # The legacy factory loads saved pipelines before importing the policy-specific
        # processor module, so register its custom tokenizer step explicitly.
        importlib.import_module("lerobot.policies.pi05.processor_pi05")
    preprocessor_overrides = {
        "device_processor": {"device": str(device)},
        "rename_observations_processor": {"rename_map": {}},
    }
    if tokenizer_path is not None:
        preprocessor_overrides["tokenizer_processor"] = {
            "tokenizer_name": str(tokenizer_path)
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(policy_path),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides=preprocessor_overrides,
    )
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    observation: dict[str, Any] = {
        "observation.state": step["observation.state"].unsqueeze(0).to(device)
    }
    for key in CAMERA_KEYS:
        image = step[key]
        if image.ndim != 3:
            raise R1A7ContractError(f"decoded image {key!r} must be 3D, got {image.shape}")
        observation[key] = image.unsqueeze(0).to(device)
    observation["task"] = frame.task
    observation["robot_type"] = ROBOT_TYPE

    amp_context = (
        torch.autocast(device_type="cuda")
        if device.type == "cuda" and policy_config.use_amp
        else nullcontext()
    )
    torch.manual_seed(seed)
    with torch.inference_mode(), amp_context:
        batch = preprocessor(observation)
        action = policy.select_action(batch)
        action = postprocessor(action)
    action = action.squeeze(0).detach().to("cpu", dtype=torch.float64).numpy()
    return decode_action(action).as_vector()


def _result_payload(
    mode: str,
    dataset_root: Path,
    frame: DatasetFrame,
    action: R1A7Action,
    policy_path: Path | None,
    seed: int | None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "hardware_commands_sent": False,
        "dataset_root": str(dataset_root),
        "episode": frame.episode,
        "frame": frame.frame,
        "global_index": frame.global_index,
        "task_index": frame.task_index,
        "task": frame.task,
        "policy_path": None if policy_path is None else str(policy_path),
        "seed": seed,
        "action_order": list(action.as_named_dict()),
        "action": action.as_vector().tolist(),
        "decoded": {
            "left_arm": action.left_arm.tolist(),
            "right_arm": action.right_arm.tolist(),
            "grippers": action.grippers.tolist(),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run zero-publisher R1-A7 policy inference on a local dataset frame."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Override an absolute tokenizer path saved by the training machine.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the recorded reference action without loading a policy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path; stdout is always written as well.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset_root.expanduser().resolve()
    frame = load_local_dataset_frame(
        dataset_root,
        args.episode,
        args.frame,
        task_override=args.task,
    )
    policy_path = None
    if args.dry_run:
        action = decode_action(frame.reference_action)
        mode = "dry-run-dataset-reference"
        inference_seed = None
    else:
        if args.policy_path is None:
            raise R1A7ContractError("--policy-path is required unless --dry-run is set")
        policy_path = args.policy_path.expanduser().resolve()
        tokenizer_path = (
            None if args.tokenizer_path is None else args.tokenizer_path.expanduser().resolve()
        )
        action = decode_action(
            _policy_action(
                dataset_root,
                frame,
                policy_path,
                tokenizer_path,
                args.device,
                args.seed,
            )
        )
        mode = "shadow-policy-inference"
        inference_seed = args.seed

    payload = _result_payload(
        mode, dataset_root, frame, action, policy_path, inference_seed
    )
    rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
