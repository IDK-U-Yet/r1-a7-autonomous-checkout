# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Input and action-chunk conditioning for R1-A7 live policy inference."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from unitree_lerobot.r1a7_contract import (
    ACTION_LOWER_LIMITS,
    ACTION_UPPER_LIMITS,
    JOINT_NAMES,
    validate_action_vector,
    validate_state_vector,
)


ARM_INDICES = tuple(range(14))
GRIPPER_INDICES = (14, 15)


def load_task_start_reference(
    dataset_root: Path, task: str
) -> tuple[np.ndarray, int]:
    """Return the median state of frame zero for all episodes of one task."""

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    root = dataset_root.expanduser().resolve()
    tasks_path = root / "meta" / "tasks.parquet"
    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise ValueError(f"dataset has no parquet data files under {root / 'data'}")
    try:
        tasks = pq.read_table(tasks_path).to_pylist()
    except (OSError, pa.ArrowException) as exc:
        raise ValueError(f"cannot read task table {tasks_path}") from exc
    matching_indices = [
        int(row["task_index"])
        for row in tasks
        if row.get("task") == task or row.get("__index_level_0__") == task
    ]
    if len(matching_indices) != 1:
        raise ValueError(
            f"expected exactly one task_index for {task!r}, found {matching_indices}"
        )
    try:
        table = pq.read_table(
            data_files,
            columns=["observation.state", "frame_index", "task_index"],
        )
    except (OSError, pa.ArrowException) as exc:
        raise ValueError(f"cannot read dataset parquet files under {root}") from exc
    mask = pc.and_(
        pc.equal(table["frame_index"], 0),
        pc.equal(table["task_index"], matching_indices[0]),
    )
    starts = table.filter(mask)["observation.state"].to_pylist()
    if not starts:
        raise ValueError(f"dataset has no frame-zero states for task {task!r}")
    reference = validate_state_vector(np.median(np.asarray(starts), axis=0))
    return reference, len(starts)


def prepare_safe_real_action(
    values: Sequence[float] | np.ndarray,
    *,
    max_arm_overshoot: float = 0.05,
    clip_arm_overshoot: bool = False,
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    """Clip targets to hard limits; optionally allow finite arm overshoot of any size."""

    if not np.isfinite(max_arm_overshoot) or max_arm_overshoot < 0:
        raise ValueError("max arm overshoot must be finite and non-negative")
    action = np.asarray(values, dtype=np.float64)
    if action.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(action)):
        validate_action_vector(action)
    safe = action.copy()
    clipping: list[dict[str, float | str]] = []
    for index in (*ARM_INDICES, *GRIPPER_INDICES):
        raw_value = float(action[index])
        safe_value = float(
            np.clip(raw_value, ACTION_LOWER_LIMITS[index], ACTION_UPPER_LIMITS[index])
        )
        overshoot = abs(raw_value - safe_value)
        if overshoot == 0:
            continue
        if index in ARM_INDICES and overshoot > max_arm_overshoot and not clip_arm_overshoot:
            raise ValueError(
                f"{JOINT_NAMES[index]} exceeds its hard limit by {overshoot:.6f} rad; "
                f"maximum clippable overshoot is {max_arm_overshoot:.6f} rad"
            )
        safe[index] = safe_value
        clipping.append(
            {
                "joint": JOINT_NAMES[index],
                "raw": raw_value,
                "safe": safe_value,
                "delta": safe_value - raw_value,
            }
        )
    return validate_action_vector(safe), clipping


def build_start_pose_command(
    measured: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Build an arm/start command and open both grippers."""

    current = validate_state_vector(measured)
    start = validate_state_vector(reference)
    command = current.copy()
    command[:14] = start[:14]
    command[14:] = ACTION_UPPER_LIMITS[14:]
    return validate_action_vector(command)


def interpolate_start_pose(
    initial: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    progress: float,
) -> np.ndarray:
    """Smoothstep interpolation used only during explicit start-pose alignment."""

    start = validate_state_vector(initial)
    end = validate_action_vector(target)
    if not np.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("initialization progress must be in [0, 1]")
    alpha = progress * progress * (3.0 - 2.0 * progress)
    return validate_action_vector(start + alpha * (end - start))


def start_pose_errors(
    measured: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Return maximum arm-radian and gripper-position errors."""

    current = validate_state_vector(measured)
    start = validate_state_vector(reference)
    return (
        float(np.max(np.abs(current[:14] - start[:14]))),
        float(np.max(np.abs(current[14:] - start[14:]))),
    )


class ActionChunkBlender:
    """Blend the first few targets of each new policy chunk from the last command."""

    def __init__(self, steps: int = 3):
        if steps < 0:
            raise ValueError("chunk blend steps must be non-negative")
        self.steps = steps
        self._last_action: np.ndarray | None = None
        self._anchor: np.ndarray | None = None
        self._blend_index = steps

    def prime(self, values: Sequence[float] | np.ndarray) -> None:
        """Use an already executed target as the anchor for the next new chunk."""

        self._last_action = validate_action_vector(values)
        self._anchor = None
        self._blend_index = self.steps

    def apply(
        self, values: Sequence[float] | np.ndarray, *, new_chunk: bool
    ) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        action = validate_action_vector(values)
        if new_chunk and self.steps > 0 and self._last_action is not None:
            self._anchor = self._last_action.copy()
            self._blend_index = 0

        active = self._anchor is not None and self._blend_index < self.steps
        if active:
            alpha = (self._blend_index + 1) / self.steps
            action = self._anchor + alpha * (action - self._anchor)
            action = validate_action_vector(action)
            self._blend_index += 1
            if self._blend_index >= self.steps:
                self._anchor = None
        else:
            alpha = 1.0

        self._last_action = action.copy()
        return action, {
            "new_chunk": bool(new_chunk),
            "active": bool(active),
            "alpha": float(alpha),
            "steps": self.steps,
        }
