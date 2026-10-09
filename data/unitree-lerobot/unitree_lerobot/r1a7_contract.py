# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""R1-A7 dataset and action contract shared by training and inference tools.

The policy intentionally excludes waist yaw because the cashier demonstrations
do not control it. The hardware bridge holds waist yaw independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class R1A7ContractError(ValueError):
    """Raised when data cannot be represented by the R1-A7 16-DoF contract."""


ROBOT_TYPE = "Unitree_R1_A7_Dex1_4Cam"
CODEBASE_VERSION = "v3.0"

LEFT_ARM_JOINT_NAMES = (
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
)
RIGHT_ARM_JOINT_NAMES = (
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
)
JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES + (
    "kLeftGripper",
    "kRightGripper",
)

CAMERA_NAMES = (
    "cam_global",
    "cam_left_wrist",
    "cam_right_wrist",
    "cam_scanner",
)
CAMERA_KEYS = tuple(f"observation.images.{name}" for name in CAMERA_NAMES)
CAMERA_SHAPES = {
    "observation.images.cam_global": (360, 640, 3),
    "observation.images.cam_left_wrist": (480, 640, 3),
    "observation.images.cam_right_wrist": (480, 640, 3),
    "observation.images.cam_scanner": (480, 640, 3),
}

# These match the hardened R1_A7_ArmController and Dex1 motor-position API.
ACTION_LOWER_LIMITS = np.asarray(
    (
        -3.1416,
        -0.22689,
        -1.9199,
        -0.97564,
        -1.9199,
        -1.6144,
        -1.6144,
        -3.1416,
        -2.47849,
        -1.9199,
        -0.97564,
        -1.9199,
        -1.6144,
        -1.6144,
        0.0,
        0.0,
    ),
    dtype=np.float64,
)
ACTION_UPPER_LIMITS = np.asarray(
    (
        2.0944,
        2.4784,
        1.9199,
        2.1852,
        1.9199,
        1.6144,
        1.6144,
        2.0944,
        0.2268,
        1.9199,
        2.1852,
        1.9199,
        1.6144,
        1.6144,
        5.4,
        5.4,
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class R1A7Action:
    """Decoded 16-dimensional policy action."""

    left_arm: np.ndarray
    right_arm: np.ndarray
    grippers: np.ndarray

    @property
    def arms(self) -> np.ndarray:
        return np.concatenate((self.left_arm, self.right_arm))

    def as_vector(self) -> np.ndarray:
        return np.concatenate((self.left_arm, self.right_arm, self.grippers))

    def as_named_dict(self) -> dict[str, float]:
        return dict(zip(JOINT_NAMES, self.as_vector().tolist(), strict=True))


def _as_vector(values: Sequence[float] | np.ndarray, name: str, size: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,):
        raise R1A7ContractError(f"{name} must have shape ({size},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        invalid = np.flatnonzero(~np.isfinite(vector)).tolist()
        raise R1A7ContractError(f"{name} contains non-finite values at indices {invalid}")
    return vector.copy()


def validate_state_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate state shape and finite values without clipping measurements."""

    return _as_vector(values, "observation.state", len(JOINT_NAMES))


def validate_action_vector(
    values: Sequence[float] | np.ndarray,
    *,
    range_tolerance: float = 1e-4,
) -> np.ndarray:
    """Validate action shape, finite values, and hardware position limits."""

    if not np.isfinite(range_tolerance) or range_tolerance < 0:
        raise ValueError("range_tolerance must be finite and non-negative")
    action = _as_vector(values, "action", len(JOINT_NAMES))
    invalid = np.flatnonzero(
        (action < ACTION_LOWER_LIMITS - range_tolerance)
        | (action > ACTION_UPPER_LIMITS + range_tolerance)
    )
    if invalid.size:
        details = ", ".join(
            f"{JOINT_NAMES[index]}={action[index]:.8g} not in "
            f"[{ACTION_LOWER_LIMITS[index]:.8g}, {ACTION_UPPER_LIMITS[index]:.8g}]"
            for index in invalid
        )
        raise R1A7ContractError(f"action exceeds R1-A7 position limits: {details}")
    return action


def decode_action(
    values: Sequence[float] | np.ndarray,
    *,
    range_tolerance: float = 1e-4,
) -> R1A7Action:
    """Decode ``[left arm 7, right arm 7, left/right gripper]``."""

    action = validate_action_vector(values, range_tolerance=range_tolerance)
    return R1A7Action(
        left_arm=action[:7],
        right_arm=action[7:14],
        grippers=action[14:16],
    )


def encode_action(
    arms: Sequence[float] | np.ndarray,
    grippers: Sequence[float] | np.ndarray,
    *,
    range_tolerance: float = 1e-4,
) -> np.ndarray:
    """Encode controller targets into the policy/dataset ordering."""

    arm_vector = _as_vector(arms, "arms", 14)
    gripper_vector = _as_vector(grippers, "grippers", 2)
    return validate_action_vector(
        np.concatenate((arm_vector, gripper_vector)),
        range_tolerance=range_tolerance,
    )


def encode_observation_state(
    arms: Sequence[float] | np.ndarray,
    grippers: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Encode measured controller state in the same 16-dimensional ordering."""

    arm_vector = _as_vector(arms, "arms", 14)
    gripper_vector = _as_vector(grippers, "grippers", 2)
    return validate_state_vector(np.concatenate((arm_vector, gripper_vector)))


def _require_feature(
    features: Mapping[str, Any], key: str, *, dtype: str, shape: tuple[int, ...]
) -> Mapping[str, Any]:
    feature = features.get(key)
    if not isinstance(feature, Mapping):
        raise R1A7ContractError(f"dataset metadata is missing feature {key!r}")
    if feature.get("dtype") != dtype:
        raise R1A7ContractError(
            f"feature {key!r} must use dtype {dtype!r}, got {feature.get('dtype')!r}"
        )
    try:
        actual_shape = tuple(feature.get("shape", ()))
    except TypeError as exc:
        raise R1A7ContractError(f"feature {key!r} has an invalid shape") from exc
    if actual_shape != shape:
        raise R1A7ContractError(
            f"feature {key!r} must have shape {shape}, got {actual_shape}"
        )
    return feature


def validate_dataset_info(info: Mapping[str, Any]) -> None:
    """Strictly validate the R1-A7 four-camera LeRobot v3 metadata contract."""

    if not isinstance(info, Mapping):
        raise R1A7ContractError("dataset info must be a JSON object")
    if info.get("codebase_version") != CODEBASE_VERSION:
        raise R1A7ContractError(
            f"expected LeRobot {CODEBASE_VERSION}, got {info.get('codebase_version')!r}"
        )
    if info.get("robot_type") != ROBOT_TYPE:
        raise R1A7ContractError(
            f"expected robot_type {ROBOT_TYPE!r}, got {info.get('robot_type')!r}"
        )

    features = info.get("features")
    if not isinstance(features, Mapping):
        raise R1A7ContractError("dataset metadata must contain a features object")

    expected_names = [list(JOINT_NAMES)]
    for key in ("observation.state", "action"):
        feature = _require_feature(
            features, key, dtype="float32", shape=(len(JOINT_NAMES),)
        )
        if feature.get("names") != expected_names:
            raise R1A7ContractError(
                f"feature {key!r} has the wrong joint order; expected {expected_names!r}, "
                f"got {feature.get('names')!r}"
            )

    actual_camera_keys = {
        key
        for key, feature in features.items()
        if isinstance(feature, Mapping) and feature.get("dtype") in ("image", "video")
    }
    expected_camera_keys = set(CAMERA_KEYS)
    if actual_camera_keys != expected_camera_keys:
        missing = sorted(expected_camera_keys - actual_camera_keys)
        extra = sorted(actual_camera_keys - expected_camera_keys)
        raise R1A7ContractError(
            f"camera keys do not match R1-A7 4Cam contract; missing={missing}, extra={extra}"
        )

    for key in CAMERA_KEYS:
        feature = features[key]
        if feature.get("dtype") not in ("image", "video"):
            raise R1A7ContractError(f"camera feature {key!r} must be image or video")
        try:
            actual_shape = tuple(feature.get("shape", ()))
        except TypeError as exc:
            raise R1A7ContractError(f"camera feature {key!r} has an invalid shape") from exc
        if actual_shape != CAMERA_SHAPES[key]:
            raise R1A7ContractError(
                f"camera feature {key!r} must have shape {CAMERA_SHAPES[key]}, "
                f"got {actual_shape}"
            )


def load_dataset_info(dataset_root: str | Path) -> dict[str, Any]:
    """Load and validate ``meta/info.json`` from a local dataset root."""

    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise R1A7ContractError(f"dataset metadata does not exist: {info_path}")
    try:
        with info_path.open("r", encoding="utf-8") as stream:
            info = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise R1A7ContractError(f"cannot read dataset metadata {info_path}: {exc}") from exc
    validate_dataset_info(info)
    return info
