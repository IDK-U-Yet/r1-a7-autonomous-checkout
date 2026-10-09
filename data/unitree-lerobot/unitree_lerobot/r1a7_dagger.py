# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Hardware-independent ownership and relative PICO alignment contracts."""
from __future__ import annotations

import time
import numpy as np
from scipy.spatial.transform import Rotation


class OwnershipRejected(RuntimeError):
    """Recoverable stale/unauthorized command; must not fault the hardware."""


class Ownership:
    def __init__(self):
        self.source = "policy"
        self.epoch = 0
        self.human_started = False
        self.changed_ns = time.monotonic_ns()

    def transition(self, source):
        if source not in {"policy", "hold", "human"}:
            raise ValueError(source)
        if source == "policy" and self.human_started:
            raise OwnershipRejected("human takeover is terminal for this trial")
        self.epoch += 1
        self.source = source
        self.human_started |= source == "human"
        self.changed_ns = time.monotonic_ns()

    def restart_trial(self):
        """Bridge must first gate this on explicit between-trial restart."""
        self.human_started = False
        self.transition("policy")

    def check(self, source, epoch):
        if source != self.source or epoch != self.epoch:
            raise OwnershipRejected(
                f"stale owner: requested {source}/{epoch}, active {self.source}/{self.epoch}"
            )

    def snapshot(self):
        return {"control_source": self.source, "owner_epoch": self.epoch,
                "owner_changed_monotonic_ns": self.changed_ns}


def valid_pose(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("invalid wrist pose")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("invalid homogeneous wrist pose")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3):
        raise ValueError("invalid wrist rotation")
    return value.copy()


def takeover_blend_seconds(value):
    seconds = float(value)
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("takeover blend seconds must be finite and non-negative")
    return seconds


class RelativeAlignment:
    """No movement at activation; deltas live in the shared robot basis.

    Optional fading removes the initial offset into ordinary XR mapping.
    Gripper takeover uses a pickup latch: preserve its previous goal until
    the trigger reaches that goal, then use absolute trigger mapping. This
    avoids both accidental opening and a limited opening range from offsets.
    """
    def __init__(self, robot_poses, xr_poses, gripper_goals, *, blend_seconds=0.0, now=None):
        if not np.isfinite(blend_seconds) or blend_seconds < 0:
            raise ValueError("blend_seconds must be finite and non-negative")
        self.robot = [valid_pose(p) for p in robot_poses]
        self.xr = [valid_pose(p) for p in xr_poses]
        if len(self.robot) != 2 or len(self.xr) != 2:
            raise ValueError("two robot and XR wrist poses required")
        self.rotation_offsets = [Rotation.from_matrix(r[:3, :3].T @ x[:3, :3]).as_rotvec()
                                 for r, x in zip(self.robot, self.xr)]
        distance = max(np.linalg.norm(r[:3, 3] - x[:3, 3]) for r, x in zip(self.robot, self.xr))
        angle = max(np.linalg.norm(v) for v in self.rotation_offsets)
        # Quintic smoothstep has a maximum derivative of 1.875.
        self.blend_duration = float(max(float(blend_seconds), 1.875 * distance / 0.10,
                                        1.875 * angle / 0.6)) if blend_seconds else 0.0
        self.blend_elapsed = 0.0
        self.blend_weight = 0.0
        self.last_blend_time = time.monotonic() if now is None else float(now)
        if not np.isfinite(self.last_blend_time):
            raise ValueError("invalid alignment timestamp")
        self.grip = np.asarray(gripper_goals, dtype=float).copy()
        if self.grip.shape != (2,) or not np.isfinite(self.grip).all():
            raise ValueError("invalid gripper goals")
        self.trigger_latched = [False, False]
        self.previous_trigger_goals = [None, None]

    def blend_snapshot(self):
        return {"mode": "fade-to-ordinary" if self.blend_duration else "fixed-relative",
                # Status crosses JSON/ZMQ and recording boundaries: never expose NumPy scalars.
                "weight": float(self.blend_weight), "offset_weight": float(1 - self.blend_weight),
                "duration_seconds": float(self.blend_duration),
                "expert_action_valid": bool(not self.blend_duration or self.blend_weight >= 1.0)}

    def poses(self, xr_poses, *, now=None):
        xr_poses = [valid_pose(p) for p in xr_poses]
        if len(xr_poses) != 2:
            raise ValueError("two current XR wrist poses required")
        stamp = time.monotonic() if now is None else float(now)
        if not np.isfinite(stamp) or stamp < self.last_blend_time:
            raise ValueError("invalid blend clock")
        if self.blend_duration:
            self.blend_elapsed += min(stamp - self.last_blend_time, 1 / 30)
            fraction = min(self.blend_elapsed / self.blend_duration, 1.0)
            self.blend_weight = 1.0 if fraction >= 1 else fraction**3 * (10 + fraction * (-15 + 6 * fraction))
        self.last_blend_time = stamp
        result = []
        for robot, anchor, current, rotvec in zip(self.robot, self.xr, xr_poses, self.rotation_offsets):
            pose = robot.copy()
            pose[:3, 3] = current[:3, 3] + (1 - self.blend_weight) * (robot[:3, 3] - anchor[:3, 3])
            blended_rotation = robot[:3, :3] @ Rotation.from_rotvec(self.blend_weight * rotvec).as_matrix()
            pose[:3, :3] = current[:3, :3] @ anchor[:3, :3].T @ blended_rotation
            if np.linalg.norm(pose[:3, 3] - robot[:3, 3]) > 0.6:
                raise ValueError("XR displacement exceeded 0.6 m; realignment required")
            result.append(pose)
        return result

    def grippers(self, triggers):
        triggers = np.asarray(triggers, dtype=float)
        if triggers.shape != (2,) or not np.isfinite(triggers).all() or np.any((triggers < 0) | (triggers > 10)):
            raise ValueError("invalid trigger values")
        values = triggers * 0.54
        for i, value in enumerate(values):
            previous = self.previous_trigger_goals[i]
            crossed = previous is not None and (previous - self.grip[i]) * (value - self.grip[i]) <= 0
            if abs(value - self.grip[i]) <= 0.12 or crossed:
                self.trigger_latched[i] = True
            self.previous_trigger_goals[i] = float(value)
        return np.array([float(v) if self.trigger_latched[i] else self.grip[i]
                         for i, v in enumerate(values)])
