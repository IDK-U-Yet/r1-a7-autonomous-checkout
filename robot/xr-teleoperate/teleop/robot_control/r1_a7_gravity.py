# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Static gravity feedforward for the R1-A7 dual-arm joint order."""

from __future__ import annotations

from pathlib import Path

import numpy as np


R1_A7_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


class R1A7GravityModel:
    """Compute the same zero-velocity RNEA term used by official XR teleop."""

    def __init__(self, xr_teleoperate_root: str | Path):
        import pinocchio as pin

        root = Path(xr_teleoperate_root).expanduser().resolve()
        model_dir = root / "assets" / "r1"
        urdf_path = model_dir / "r1_a7.urdf"
        if not urdf_path.is_file():
            raise FileNotFoundError(f"R1-A7 URDF does not exist: {urdf_path}")

        self._pin = pin
        robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), [str(model_dir)])
        locked_joint_ids = [
            robot.model.getJointId(name)
            for name in ("waist_yaw_joint", "head_pitch_joint", "head_yaw_joint")
        ]
        self._robot = robot.buildReducedRobot(
            locked_joint_ids,
            np.zeros(robot.model.nq, dtype=np.float64),
        )
        joint_names = tuple(self._robot.model.names[1:])
        if joint_names != R1_A7_ARM_JOINT_NAMES:
            raise RuntimeError(
                "unexpected R1-A7 gravity-model joint order: "
                f"expected={R1_A7_ARM_JOINT_NAMES}, received={joint_names}"
            )

    def gravity(self, arm_q) -> np.ndarray:
        q = np.asarray(arm_q, dtype=np.float64)
        if q.shape != (14,) or not np.all(np.isfinite(q)):
            raise ValueError("R1-A7 gravity input must be a finite 14D joint vector")
        tau = np.asarray(
            self._pin.rnea(
                self._robot.model,
                self._robot.data,
                q,
                np.zeros(self._robot.model.nv, dtype=np.float64),
                np.zeros(self._robot.model.nv, dtype=np.float64),
            ),
            dtype=np.float64,
        )
        if tau.shape != (14,) or not np.all(np.isfinite(tau)):
            raise RuntimeError("R1-A7 gravity model returned an invalid torque vector")
        return tau
