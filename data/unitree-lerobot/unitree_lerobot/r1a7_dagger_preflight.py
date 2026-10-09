# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Check DAgger assets, writable storage and ports without creating controllers."""
import json
import os
import shutil
import socket
import tempfile
from pathlib import Path
from unitree_lerobot.r1a7_contract import CAMERA_KEYS
from unitree_lerobot.r1a7_live_dagger import build_dagger_parser
from unitree_lerobot.r1a7_dagger_recording import collection_path, collection_name


def preflight(args):
    args.collection_name = collection_name(args.dataset_root, args.collection_name)
    args.dagger_record_dir = collection_path(args.dagger_record_dir, args.collection_name)
    from lerobot.policies import make_policy  # register PI0.5 config; not instantiated
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    config = PreTrainedConfig.from_pretrained(args.policy_path)
    if config.type != "pi05":
        raise ValueError("expected PI0.5 checkpoint")
    if tuple(config.input_features["observation.state"].shape) != (16,) or tuple(config.output_features["action"].shape) != (16,):
        raise ValueError("16D state/action checkpoint required")
    cameras = {k for k in config.input_features if k.startswith("observation.images.")}
    if cameras != set(CAMERA_KEYS):
        raise ValueError("checkpoint does not use the four R1-A7 cameras")
    for path in (args.policy_path / "model.safetensors", args.tokenizer_path / "tokenizer.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset = LeRobotDataset(args.dataset_root.name, root=args.dataset_root, video_backend="pyav")
    if not {args.task_before, args.task_after}.issubset(set(dataset.meta.tasks.index)):
        raise ValueError("task text differs from checkpoint reference dataset")
    if set(dataset.meta.camera_keys) != cameras:
        raise ValueError("checkpoint and dataset camera keys differ")
    if args.manual_cycles or args.completion_wait_seconds or args.max_commands < 0:
        raise ValueError("DAgger uses explicit P/R restart, not automatic/manual inference resets")
    for port in (55580, 55582, 8012):
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", port))
    args.dagger_record_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=args.dagger_record_dir) as stream:
        stream.write(b"DAgger storage check")
        stream.flush()
        os.fsync(stream.fileno())
    free = shutil.disk_usage(args.dagger_record_dir).free
    if free < 1024 ** 3:
        raise RuntimeError("less than 1 GiB recording storage free; free space before starting")
    print(json.dumps({"event": "dagger_preflight_ok", "hardware_commands_sent": False,
                      "checkpoint": str(args.policy_path), "storage_free_gib": round(free / 1024 ** 3, 2)}))


def main():
    parser = build_dagger_parser()
    preflight(parser.parse_args())


if __name__ == "__main__":
    main()
