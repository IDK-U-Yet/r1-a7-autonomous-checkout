# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
from __future__ import annotations

import importlib.util
import time
from multiprocessing import Value
from pathlib import Path

import numpy as np
import pytest


BRIDGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "robot"
    / "xr-teleoperate"
    / "teleop"
    / "tools"
    / "r1a7_real_command_bridge.py"
)


def _load_bridge_module():
    spec = importlib.util.spec_from_file_location("r1a7_real_command_bridge", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bridge = _load_bridge_module()


def test_dagger_hold_preserves_requested_grip_and_disarm_is_terminal():
    from types import SimpleNamespace
    from unitree_lerobot.r1a7_dagger import Ownership
    controller, *_ = make_controller()
    owner = Ownership()
    controller.dagger = SimpleNamespace(owner=owner, disarmed=False, enabled=True)
    controller.execute(np.r_[np.zeros(14), [0.3, 0.6]])
    held = controller.hold_current(preserve_grip=True)
    np.testing.assert_allclose(held["held_state"][14:], [0.3, 0.6])
    controller.disarm()
    assert controller.dagger.disarmed and not controller.dagger.enabled
    with pytest.raises(RuntimeError, match="disarmed"):
        controller.execute(np.r_[np.zeros(14), [0.3, 0.6]])
    assert not controller.active


def test_ordinary_policy_hold_retains_grip_across_repeated_waits():
    controller, _arm, left, right, *_ = make_controller()
    assert controller.dagger is None
    controller.execute(np.r_[np.zeros(14), [0.3, 0.6]])
    for _ in range(3):
        held = controller.hold_current(preserve_grip=True)
        np.testing.assert_allclose(held["held_state"][14:], [0.3, 0.6])
        assert left.value == pytest.approx(0.3 / 5.4 * 10)
        assert right.value == pytest.approx(0.6 / 5.4 * 10)


def test_preserved_grip_without_previous_target_uses_measured_opening():
    controller, *_ = make_controller()
    assert controller.last_action is None
    held = controller.hold_current(preserve_grip=True)
    np.testing.assert_allclose(held["held_state"][14:], [1, 2])


class FakeArm:
    def __init__(self):
        self.tracking_active = False
        self.arm_q = np.zeros(14, dtype=np.float64)
        self.waist_q = 0.0
        self.last_lowstate_time = time.monotonic()
        self.arm_commands = []
        self.waist_commands = []

    def check_health(self):
        return None

    def set_tracking_active(self, active):
        self.tracking_active = bool(active)

    def ctrl_dual_arm(self, q, tau):
        if not self.tracking_active:
            return False
        self.arm_commands.append((np.asarray(q).copy(), np.asarray(tau).copy()))
        return True

    def ctrl_waist_yaw(self, axis):
        self.waist_commands.append(float(axis))
        return self.tracking_active

    def get_current_dual_arm_q(self):
        return self.arm_q.copy()

    def get_current_waist_yaw_q(self):
        return self.waist_q


class FakeGripper:
    def _snapshot_gripper_state(self):
        return (
            np.asarray([1.0, 2.0]),
            np.zeros(2, dtype=np.float64),
            np.asarray([0.01, 0.02]),
        )


class FakeGravityModel:
    def __init__(self):
        self.inputs = []

    def gravity(self, q):
        q = np.asarray(q, dtype=np.float64)
        self.inputs.append(q.copy())
        return q + 1.0


def make_controller(command_timeout=0.5, pre_policy_timeout=5.0):
    left = Value("d", 0.0, lock=True)
    right = Value("d", 0.0, lock=True)
    ready = Value("b", False, lock=True)
    arm = FakeArm()
    gravity = FakeGravityModel()
    controller = bridge.R1A7RealCommandController(
        arm,
        FakeGripper(),
        left,
        right,
        ready,
        gravity,
        command_timeout=command_timeout,
        pre_policy_timeout=pre_policy_timeout,
    )
    return controller, arm, left, right, ready, gravity


def test_official_action_is_forwarded_without_extra_arm_or_gripper_reduction():
    controller, arm, left, right, ready, gravity = make_controller()
    action = np.zeros(16, dtype=np.float64)
    action[:14] = np.linspace(-0.2, 0.2, 14)
    action[14:] = (5.4, 2.7)

    reply = controller.execute(action)

    assert reply["hardware_commands_sent"] is True
    np.testing.assert_allclose(arm.arm_commands[-1][0], action[:14])
    np.testing.assert_allclose(gravity.inputs[-1], action[:14])
    np.testing.assert_allclose(arm.arm_commands[-1][1], action[:14] + 1.0)
    assert arm.waist_commands == [0.0]
    assert reply["waist_mode"] == "hold_at_activation"
    assert reply["waist_hold_target"] == pytest.approx(0.0)
    np.testing.assert_allclose(reply["executed_action"], action)
    assert left.value == pytest.approx(10.0)
    assert right.value == pytest.approx(5.0)
    assert bool(ready.value) is True


def test_waist_is_held_outside_policy_contract():
    controller, arm, _left, _right, _ready, _gravity = make_controller()
    arm.waist_q = 0.25
    first = np.zeros(16, dtype=np.float64)
    second = np.ones(16, dtype=np.float64) * 0.1

    first_reply = controller.execute(first)
    arm.waist_q = -0.1
    second_reply = controller.execute(second)

    assert arm.waist_commands == [0.0, 0.0]
    assert first_reply["waist_hold_target"] == pytest.approx(0.25)
    assert second_reply["waist_hold_target"] == pytest.approx(0.25)
    assert len(first_reply["executed_action"]) == 16
    assert len(second_reply["executed_action"]) == 16


def test_hardware_position_limit_violation_is_rejected_before_motion():
    controller, arm, _left, _right, ready, _gravity = make_controller()
    action = np.zeros(16, dtype=np.float64)
    action[14] = 5.41

    with pytest.raises(ValueError, match="kLeftGripper"):
        controller.execute(action)

    assert arm.arm_commands == []
    assert arm.tracking_active is False
    assert bool(ready.value) is False


def test_command_stream_timeout_holds_and_requires_restart():
    controller, arm, _left, _right, ready, _gravity = make_controller(command_timeout=0.01)
    controller.execute(np.zeros(16, dtype=np.float64))
    controller.last_command_monotonic -= 1.0

    controller.check_timeout()

    assert controller.fault_reason is not None
    assert "restart required" in controller.fault_reason
    assert arm.tracking_active is False
    assert bool(ready.value) is False
    with pytest.raises(RuntimeError, match="fault-locked"):
        controller.execute(np.zeros(16, dtype=np.float64))


def test_initialization_hold_refreshes_until_first_policy_command():
    controller, arm, _left, _right, ready, gravity = make_controller(
        command_timeout=0.01
    )
    initialization = np.linspace(-0.2, 0.2, 16)
    initialization[14:] = (5.4, 5.4)

    reply = controller.execute(initialization, policy_command=False)
    controller.last_command_monotonic -= 1.0
    controller.check_timeout()

    assert reply["policy_watchdog_active"] is False
    assert controller.fault_reason is None
    assert controller.policy_watchdog_active is False
    assert controller.pre_policy_hold_count == 1
    assert bool(ready.value) is True
    np.testing.assert_allclose(arm.arm_commands[-1][0], initialization[:14])
    np.testing.assert_allclose(gravity.inputs[-1], initialization[:14])

    policy_reply = controller.execute(initialization)
    assert policy_reply["policy_watchdog_active"] is True
    controller.last_command_monotonic -= 1.0
    controller.check_timeout()
    assert controller.fault_reason is not None
    assert arm.tracking_active is False


def test_initialization_hold_has_a_bounded_first_policy_deadline():
    controller, arm, _left, _right, ready, _gravity = make_controller(
        command_timeout=0.01,
        pre_policy_timeout=0.1,
    )
    controller.execute(np.zeros(16, dtype=np.float64), policy_command=False)
    controller.last_command_monotonic -= 1.0

    controller.check_timeout()

    assert "first policy command did not arrive" in controller.fault_reason
    assert arm.tracking_active is False
    assert bool(ready.value) is False


def test_manual_return_reenters_pre_policy_watchdog_without_clearing_faults():
    controller, arm, _left, _right, _ready, _gravity = make_controller(
        command_timeout=0.01,
        pre_policy_timeout=5.0,
    )
    action = np.zeros(16, dtype=np.float64)
    controller.execute(action, policy_command=True)
    assert controller.policy_watchdog_active is True

    controller.execute(action, policy_command=False)
    assert controller.policy_watchdog_active is False
    controller.last_command_monotonic -= 1.0
    controller.check_timeout()
    assert controller.fault_reason is None
    assert controller.pre_policy_hold_count == 1



def test_explicit_hold_freezes_measured_pose_and_refreshes_watchdog():
    controller, arm, left, right, ready, gravity = make_controller(command_timeout=0.01)
    command = np.zeros(16, dtype=np.float64)
    command[14:] = (5.4, 5.4)
    controller.execute(command)
    arm.arm_q = np.linspace(-0.1, 0.1, 14)
    controller.last_command_monotonic -= 1.0

    reply = controller.hold_current()
    controller.check_timeout()

    assert reply["status"] == "ok"
    assert reply["hold_count"] == 1
    assert reply["hardware_commands_sent"] is True
    np.testing.assert_allclose(arm.arm_commands[-1][0], arm.arm_q)
    np.testing.assert_allclose(gravity.inputs[-1], arm.arm_q)
    np.testing.assert_allclose(arm.arm_commands[-1][1], arm.arm_q + 1.0)
    assert left.value == pytest.approx(1.0 / 5.4 * 10.0)
    assert right.value == pytest.approx(2.0 / 5.4 * 10.0)
    assert controller.fault_reason is None
    assert bool(ready.value) is True


def test_state_snapshot_preserves_16d_policy_order_and_reports_waist_separately():
    controller, arm, _left, _right, _ready, _gravity = make_controller()
    arm.arm_q = np.arange(14, dtype=np.float64) / 10
    arm.waist_q = 0.3

    reply = controller.state_snapshot()

    assert reply["state"] == pytest.approx([*arm.arm_q.tolist(), 1.0, 2.0])
    assert reply["measured_waist_yaw"] == pytest.approx(0.3)
    assert set(reply["sample_monotonic_ns"]) == {
        "lowstate",
        "left_gripper",
        "right_gripper",
    }


@pytest.mark.parametrize("ticks", [0, 1, 400])
def test_bridge_reports_blend_training_gate_and_pause_clears_it(ticks):
    from types import SimpleNamespace
    from unitree_lerobot.r1a7_dagger import Ownership, RelativeAlignment
    from zmq.utils import jsonapi
    controller, *_ = make_controller()
    xr = np.eye(4)
    xr[0, 3] = 0.5  # NumPy offset term, rather than default seconds, wins max().
    alignment = RelativeAlignment([np.eye(4)] * 2, [xr] * 2,
                                  [1, 2], blend_seconds=3, now=0)
    for tick in range(1, ticks + 1):
        alignment.poses([xr] * 2, now=tick / 30)
    controller.dagger = SimpleNamespace(owner=Ownership(), enabled=True,
                                       reason="human", xr_ready=True, alignment=alignment)
    reply = controller.state_snapshot()
    snapshot = reply["takeover_blend"]
    assert type(snapshot["expert_action_valid"]) is bool
    assert snapshot["expert_action_valid"] is (ticks == 400)
    for name in ("weight", "offset_weight", "duration_seconds"):
        assert type(snapshot[name]) is float
    assert jsonapi.loads(jsonapi.dumps(reply)) == reply
    controller.dagger.alignment = None
    reply = controller.state_snapshot()
    assert reply["takeover_blend"] is None
    assert jsonapi.loads(jsonapi.dumps(reply)) == reply
