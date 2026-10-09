# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
from enum import Enum
import math
import time

import numpy as np


class SafetyViolation(ValueError):
    pass


class TeleopState(str, Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    HOLD = "hold"
    FAULT = "fault"


class ControllerButtonEdges:
    """Report one event per physical press, not once per control-loop sample."""

    def __init__(self):
        self._previous_a = False
        self._previous_b = False
        self._previous_x = False
        self._previous_y = False

    def update(self, a_pressed, b_pressed, x_pressed=False, y_pressed=False):
        a_pressed = bool(a_pressed)
        b_pressed = bool(b_pressed)
        x_pressed = bool(x_pressed)
        y_pressed = bool(y_pressed)
        a_rising = a_pressed and not self._previous_a
        b_rising = b_pressed and not self._previous_b
        x_rising = x_pressed and not self._previous_x
        y_rising = y_pressed and not self._previous_y
        self._previous_a = a_pressed
        self._previous_b = b_pressed
        self._previous_x = x_pressed
        self._previous_y = y_pressed
        return a_rising, b_rising, x_rising, y_rising


class XRWatchdog:
    """Two-stage XR watchdog with a latched fault state."""

    def __init__(self, hold_timeout=0.25, fault_timeout=1.0, clock=time.monotonic):
        if not 0.0 < hold_timeout < fault_timeout:
            raise ValueError("watchdog requires 0 < hold_timeout < fault_timeout")
        self.hold_timeout = float(hold_timeout)
        self.fault_timeout = float(fault_timeout)
        self.clock = clock
        self.state = TeleopState.WAITING
        self.fault_reason = None

    def activate(self, event_timestamp):
        age = self.age(event_timestamp)
        if self.state == TeleopState.FAULT:
            raise SafetyViolation("XR watchdog fault is latched")
        if not math.isfinite(age) or age > self.hold_timeout:
            raise SafetyViolation("cannot activate with stale XR data")
        self.state = TeleopState.ACTIVE

    def age(self, event_timestamp):
        if not event_timestamp:
            return math.inf
        return max(0.0, self.clock() - float(event_timestamp))

    def update(self, event_timestamp):
        age = self.age(event_timestamp)
        if self.state == TeleopState.FAULT:
            return self.state, age
        if self.state == TeleopState.WAITING:
            return self.state, age
        if age > self.fault_timeout:
            self.state = TeleopState.FAULT
            self.fault_reason = (
                f"XR data timeout ({age:.3f}s > {self.fault_timeout:.3f}s)"
            )
        elif age > self.hold_timeout:
            self.state = TeleopState.HOLD
        else:
            self.state = TeleopState.ACTIVE
        return self.state, age

    def pause(self):
        if self.state != TeleopState.FAULT:
            self.state = TeleopState.WAITING


def _finite_pose(value, name):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise SafetyViolation(f"{name} must be a finite 4x4 pose")
    return pose.copy()


def _project_rotation(rotation):
    u, _, vh = np.linalg.svd(rotation)
    projected = u @ vh
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return projected


def _axis_angle(rotation):
    rotation = _project_rotation(rotation)
    angle = math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        return np.array([1.0, 0.0, 0.0]), 0.0
    axis = np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]
    )
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        values, vectors = np.linalg.eig(rotation)
        axis = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
        norm = np.linalg.norm(axis)
    return axis / norm, angle


def _rodrigues(axis, angle):
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


class ActivationPoseEnvelope:
    """Limit controller displacement relative to the operator activation pose."""

    def __init__(self, horizontal_limit=0.15, vertical_down_limit=0.15,
                 vertical_up_limit=0.30, rotation_limit_degrees=35.0):
        self.horizontal_limit = float(horizontal_limit)
        self.vertical_down_limit = float(vertical_down_limit)
        self.vertical_up_limit = float(vertical_up_limit)
        self.rotation_limit = math.radians(float(rotation_limit_degrees))
        if (
            self.horizontal_limit <= 0.0
            or self.vertical_down_limit <= 0.0
            or self.vertical_up_limit <= 0.0
            or self.rotation_limit <= 0.0
        ):
            raise ValueError("pose envelope limits must be positive")
        self._anchors = None
        self.last_status = None

    @property
    def engaged(self):
        return self._anchors is not None

    def engage(self, poses):
        if len(poses) != 2:
            raise SafetyViolation("pose envelope requires left and right poses")
        self._anchors = tuple(_finite_pose(value, f"pose[{index}]") for index, value in enumerate(poses))
        self.last_status = None

    def disengage(self):
        self._anchors = None
        self.last_status = None

    def apply(self, poses):
        if not self.engaged:
            raise SafetyViolation("pose envelope is not engaged")
        output = []
        status = []
        for index, (value, anchor) in enumerate(zip(poses, self._anchors)):
            current = _finite_pose(value, f"pose[{index}]")
            delta = np.linalg.inv(anchor) @ current
            translation_delta = current[:3, 3] - anchor[:3, 3]
            applied_translation_delta = np.clip(
                translation_delta,
                [-self.horizontal_limit, -self.horizontal_limit, -self.vertical_down_limit],
                [self.horizontal_limit, self.horizontal_limit, self.vertical_up_limit],
            )
            clipped_translation = ~np.isclose(
                translation_delta, applied_translation_delta, rtol=0.0, atol=1e-12
            )
            axis, angle = _axis_angle(delta[:3, :3])
            applied_angle = min(angle, self.rotation_limit)
            result = anchor @ delta
            result[:3, 3] = anchor[:3, 3] + applied_translation_delta
            result[:3, :3] = anchor[:3, :3] @ _rodrigues(axis, applied_angle)
            output.append(result)
            status.append(
                {
                    "translation_delta_m": translation_delta,
                    "translation_applied_m": applied_translation_delta,
                    "translation_clipped": clipped_translation,
                    "rotation_delta_degrees": math.degrees(angle),
                    "rotation_applied_degrees": math.degrees(applied_angle),
                    "rotation_clipped": angle > self.rotation_limit + 1e-12,
                    "clipped": bool(np.any(clipped_translation))
                    or angle > self.rotation_limit + 1e-12,
                }
            )
        self.last_status = tuple(status)
        return tuple(output)


def apply_enabled_arms(q_target, tau_target, measured_q, enabled_arms):
    q_target = np.asarray(q_target, dtype=float).copy()
    tau_target = np.asarray(tau_target, dtype=float).copy()
    measured_q = np.asarray(measured_q, dtype=float)
    if q_target.shape != (14,) or tau_target.shape != (14,) or measured_q.shape != (14,):
        raise SafetyViolation("R1-A7 arm vectors must contain 14 joints")
    if enabled_arms not in ("left", "right", "both"):
        raise SafetyViolation(f"invalid enabled-arms value: {enabled_arms}")
    if enabled_arms == "left":
        q_target[7:] = measured_q[7:]
        tau_target[7:] = 0.0
    elif enabled_arms == "right":
        q_target[:7] = measured_q[:7]
        tau_target[:7] = 0.0
    return q_target, tau_target


class JointCommandSafety:
    def __init__(self, lower_limits, upper_limits, velocity_limit, control_dt,
                 joint_limit_margin, torque_limits=None):
        self.lower = np.asarray(lower_limits, dtype=float)
        self.upper = np.asarray(upper_limits, dtype=float)
        self.velocity_limit = float(velocity_limit)
        self.control_dt = float(control_dt)
        self.margin = float(joint_limit_margin)
        self.torque_limits = (
            None if torque_limits is None else np.asarray(torque_limits, dtype=float)
        )
        if self.lower.shape != self.upper.shape or self.lower.ndim != 1:
            raise ValueError("joint limits must be equally sized vectors")
        if np.any(self.lower + self.margin >= self.upper - self.margin):
            raise ValueError("joint margin leaves an empty valid range")
        if self.torque_limits is not None and (
            self.torque_limits.shape != self.lower.shape
            or np.any(self.torque_limits <= 0.0)
        ):
            raise ValueError("torque limits must be a positive vector matching joints")
        if self.velocity_limit <= 0.0 or self.control_dt <= 0.0:
            raise ValueError("velocity_limit and control_dt must be positive")

    def sanitize(self, q_target, tau_target):
        q_target = np.asarray(q_target, dtype=float)
        tau_target = np.asarray(tau_target, dtype=float)
        if q_target.shape != self.lower.shape or tau_target.shape != self.lower.shape:
            raise SafetyViolation("joint command has the wrong shape")
        if not np.all(np.isfinite(q_target)) or not np.all(np.isfinite(tau_target)):
            raise SafetyViolation("joint command contains a non-finite value")
        if self.torque_limits is not None:
            tau_target = np.clip(tau_target, -self.torque_limits, self.torque_limits)
        return np.clip(q_target, self.lower + self.margin, self.upper - self.margin), tau_target.copy()

    def rate_limit(self, q_target, previous_command):
        q_target = np.asarray(q_target, dtype=float)
        previous_command = np.asarray(previous_command, dtype=float)
        step = self.velocity_limit * self.control_dt
        return previous_command + np.clip(q_target - previous_command, -step, step)
