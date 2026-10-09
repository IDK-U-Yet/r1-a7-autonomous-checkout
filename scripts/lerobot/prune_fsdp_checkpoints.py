#!/usr/bin/env python3
"""Safely inspect or prune completed LeRobot/FSDP checkpoints.

The default is a dry-run. Deletion requires ``--apply`` and is limited to numeric
checkpoint directories below the selected run's ``checkpoints/`` directory. The
latest N checkpoints, explicitly protected steps, and the target of ``last`` are
always retained. Incomplete directories are reported but never deleted.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


CHECKPOINT_NAME = re.compile(r"^(?P<step>[0-9]+)$")


@dataclass(frozen=True)
class Checkpoint:
    step: int
    path: Path
    complete: bool
    protected_by_last: bool


def _is_complete(path: Path) -> bool:
    """Require enough state to make the checkpoint resumable."""
    model_dir = path / "pretrained_model"
    if not (model_dir / "train_config.json").is_file():
        return False
    training_state = path / "training_state"
    if not (training_state / "training_step.json").is_file():
        return False
    optimizer_complete = (training_state / "optimizer_state.safetensors").is_file() or any(
        candidate.is_dir()
        and candidate.name.startswith("optimizer_")
        and any(child.is_file() for child in candidate.rglob("*"))
        for candidate in training_state.iterdir()
    )
    if not optimizer_complete:
        return False
    # DCP runs store sharded model state; safetensors runs store a single model file.
    if (model_dir / "model.safetensors").is_file():
        return True
    return model_dir.is_dir() and any(
        candidate.is_dir()
        and candidate.name.startswith("pytorch_model_fsdp_")
        and any(child.is_file() for child in candidate.rglob("*"))
        for candidate in model_dir.iterdir()
    )


def _last_target(checkpoints_dir: Path) -> Path | None:
    last = checkpoints_dir / "last"
    if not last.exists() and not last.is_symlink():
        return None
    try:
        return last.resolve(strict=True)
    except OSError:
        return None


def discover(run_dir: Path) -> list[Checkpoint]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"missing checkpoints directory: {checkpoints_dir}")
    last_target = _last_target(checkpoints_dir)
    published_step = int(last_target.name) if last_target is not None and last_target.name.isdigit() else None
    result: list[Checkpoint] = []
    for entry in checkpoints_dir.iterdir():
        match = CHECKPOINT_NAME.fullmatch(entry.name)
        if not match or not entry.is_dir():
            continue
        result.append(
            Checkpoint(
                step=int(match.group("step")),
                path=entry,
                complete=_is_complete(entry)
                and published_step is not None
                and int(match.group("step")) <= published_step,
                protected_by_last=last_target == entry.resolve(),
            )
        )
    return sorted(result, key=lambda item: item.step)


def plan(
    checkpoints: list[Checkpoint],
    keep_last: int,
    keep_steps: set[int],
) -> tuple[list[Checkpoint], list[Checkpoint]]:
    complete = [item for item in checkpoints if item.complete]
    keep: set[Path] = {item.path for item in complete[-keep_last:]}
    keep.update(item.path for item in complete if item.step in keep_steps or item.protected_by_last)
    delete = [item for item in complete if item.path not in keep]
    return [item for item in checkpoints if item.path in keep], delete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Formal training output directory")
    parser.add_argument("--keep-last", type=int, default=2, help="Retain the newest N complete checkpoints")
    parser.add_argument(
        "--keep-step", type=int, action="append", default=[], help="Explicitly retain this checkpoint step (repeatable)"
    )
    parser.add_argument("--apply", action="store_true", help="Actually remove planned checkpoints")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.keep_last < 1:
        raise SystemExit("--keep-last must be at least 1")
    run_dir = args.run_dir.resolve()
    checkpoints = discover(run_dir)
    keep, delete = plan(checkpoints, args.keep_last, set(args.keep_step))
    print(f"Run: {run_dir}")
    for item in checkpoints:
        if not item.complete:
            print(f"SKIP incomplete-or-unpublished step={item.step:06d} path={item.path}")
    for item in keep:
        reason = "last" if item.protected_by_last else "retained"
        print(f"KEEP {reason} step={item.step:06d} path={item.path}")
    for item in delete:
        print(f"DELETE step={item.step:06d} path={item.path}")
    if not args.apply:
        print("Dry run only; pass --apply after reviewing the list.")
        return
    for item in delete:
        shutil.rmtree(item.path)
        print(f"REMOVED step={item.step:06d}")


if __name__ == "__main__":
    main()
