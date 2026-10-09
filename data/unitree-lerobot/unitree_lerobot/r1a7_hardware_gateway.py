# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Explicitly gated, delayed access to the hardened xr-teleoperate controllers.

Importing this module is side-effect free.  It intentionally does not construct
controllers: their constructors create DDS publishers and subscribers.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REAL_ROBOT_CONFIRMATION = "I_UNDERSTAND_R1A7_COMMANDS_MOVE_REAL_HARDWARE"


class RealRobotAuthorizationError(PermissionError):
    """Raised before any hardware module is imported when authorization is absent."""


@dataclass(frozen=True)
class RealRobotAuthorization:
    """Two independent, explicit values required before resolving hardware code."""

    allow_real_robot: bool = False
    confirmation: str = ""

    def require(self) -> None:
        if not self.allow_real_robot or self.confirmation != REAL_ROBOT_CONFIRMATION:
            raise RealRobotAuthorizationError(
                "real R1-A7 access is locked; both allow_real_robot=True and the exact "
                f"confirmation {REAL_ROBOT_CONFIRMATION!r} are required"
            )


@dataclass(frozen=True)
class HardenedControllerTypes:
    arm: type[Any]
    gripper: type[Any]


def load_hardened_controller_types(
    xr_teleoperate_root: str | Path,
    authorization: RealRobotAuthorization,
) -> HardenedControllerTypes:
    """Resolve controller classes after authorization, without instantiating them."""

    authorization.require()
    root = Path(xr_teleoperate_root).expanduser().resolve()
    package_dir = root / "teleop"
    if not package_dir.is_dir():
        raise FileNotFoundError(f"xr-teleoperate package does not exist: {package_dir}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    arm_module = importlib.import_module("teleop.robot_control.robot_arm")
    gripper_module = importlib.import_module("teleop.robot_control.robot_hand_unitree")
    return HardenedControllerTypes(
        arm=arm_module.R1_A7_ArmController,
        gripper=gripper_module.Dex1_1_Gripper_Controller,
    )
