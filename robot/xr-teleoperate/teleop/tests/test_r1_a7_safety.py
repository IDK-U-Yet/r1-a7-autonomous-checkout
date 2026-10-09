# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from teleop.robot_control.safety import (
    ActivationPoseEnvelope,
    ControllerButtonEdges,
    JointCommandSafety,
    SafetyViolation,
    TeleopState,
    XRWatchdog,
    apply_enabled_arms,
)
from teleop.robot_control.robot_hand_unitree import Dex1TorqueLimiter, map_dex1_input
from teleop.robot_control.robot_arm import R1_A7_ArmController, R1_A7_JointIndex
from teleop.utils.diagnostics import DiagnosticsLogger
from teleop.utils.weighted_moving_filter import WeightedMovingFilter


def pose(x=0.0, y=0.0, z=0.0, angle=0.0):
    result = np.eye(4)
    result[:3, 3] = (x, y, z)
    result[:3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return result


class FakeClock:
    def __init__(self, now=10.0):
        self.now = now

    def __call__(self):
        return self.now


class XRWatchdogTest(unittest.TestCase):
    def test_hold_then_latched_fault(self):
        clock = FakeClock()
        watchdog = XRWatchdog(0.25, 1.0, clock=clock)
        watchdog.activate(10.0)
        self.assertEqual(watchdog.update(10.0)[0], TeleopState.ACTIVE)
        clock.now = 10.3
        self.assertEqual(watchdog.update(10.0)[0], TeleopState.HOLD)
        clock.now = 10.1
        self.assertEqual(watchdog.update(10.0)[0], TeleopState.ACTIVE)
        clock.now = 11.1
        self.assertEqual(watchdog.update(10.0)[0], TeleopState.FAULT)
        clock.now = 10.0
        self.assertEqual(watchdog.update(10.0)[0], TeleopState.FAULT)
        with self.assertRaises(SafetyViolation):
            watchdog.activate(10.0)

    def test_rejects_stale_activation(self):
        clock = FakeClock()
        watchdog = XRWatchdog(clock=clock)
        with self.assertRaises(SafetyViolation):
            watchdog.activate(0.0)


class ControllerButtonEdgesTest(unittest.TestCase):
    def test_long_press_emits_one_edge_and_repress_emits_another(self):
        buttons = ControllerButtonEdges()
        self.assertEqual(buttons.update(False, False, False), (False, False, False, False))
        self.assertEqual(buttons.update(True, False, False), (True, False, False, False))
        self.assertEqual(buttons.update(True, False, False), (False, False, False, False))
        self.assertEqual(buttons.update(False, False, False), (False, False, False, False))
        self.assertEqual(buttons.update(True, False, False), (True, False, False, False))

    def test_reports_simultaneous_edges_for_pause_priority(self):
        buttons = ControllerButtonEdges()
        self.assertEqual(buttons.update(True, True, True, True), (True, True, True, True))

    def test_left_x_long_press_emits_one_record_edge(self):
        buttons = ControllerButtonEdges()
        self.assertEqual(buttons.update(False, False, True), (False, False, True, False))
        self.assertEqual(buttons.update(False, False, True), (False, False, False, False))
        self.assertEqual(buttons.update(False, False, False), (False, False, False, False))
        self.assertEqual(buttons.update(False, False, True), (False, False, True, False))

    def test_left_y_long_press_emits_one_reject_edge(self):
        buttons = ControllerButtonEdges()
        self.assertEqual(buttons.update(False, False, False, True), (False, False, False, True))
        self.assertEqual(buttons.update(False, False, False, True), (False, False, False, False))
        self.assertEqual(buttons.update(False, False, False, False), (False, False, False, False))
        self.assertEqual(buttons.update(False, False, False, True), (False, False, False, True))


class ActivationPoseEnvelopeTest(unittest.TestCase):
    def test_clamps_translation_and_rotation_around_activation(self):
        envelope = ActivationPoseEnvelope(
            horizontal_limit=0.15,
            vertical_down_limit=0.15,
            vertical_up_limit=0.30,
            rotation_limit_degrees=35.0,
        )
        anchors = (pose(1.0, 2.0, 3.0), pose(-1.0, -2.0, -3.0))
        envelope.engage(anchors)
        output = envelope.apply((pose(1.3, 1.8, 3.4, np.deg2rad(50.0)), anchors[1]))
        np.testing.assert_allclose(output[0][:3, 3], [1.15, 1.85, 3.3], atol=1e-9)
        angle = np.arccos(np.clip((np.trace(output[0][:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
        self.assertAlmostEqual(angle, np.deg2rad(35.0))
        np.testing.assert_allclose(output[1], anchors[1])
        self.assertEqual(
            envelope.last_status[0]["translation_clipped"].tolist(),
            [True, True, True],
        )
        self.assertTrue(envelope.last_status[0]["rotation_clipped"])
        self.assertTrue(envelope.last_status[0]["clipped"])
        self.assertFalse(envelope.last_status[1]["clipped"])

    def test_uses_asymmetric_world_vertical_limits(self):
        anchor = pose(0.0, 0.0, 1.0, np.deg2rad(90.0))
        envelope = ActivationPoseEnvelope(0.15, 0.15, 0.30, 35.0)
        envelope.engage((anchor, anchor))
        up, down = envelope.apply((pose(0.0, 0.0, 1.5), pose(0.0, 0.0, 0.5)))
        self.assertAlmostEqual(up[2, 3], 1.30)
        self.assertAlmostEqual(down[2, 3], 0.85)


class ArmCommandTest(unittest.TestCase):
    def test_disabled_side_holds_measured_position(self):
        target = np.arange(14, dtype=float)
        torque = np.ones(14)
        measured = np.arange(14, dtype=float) + 100.0
        q, tau = apply_enabled_arms(target, torque, measured, "left")
        np.testing.assert_allclose(q[:7], target[:7])
        np.testing.assert_allclose(q[7:], measured[7:])
        np.testing.assert_allclose(tau[7:], 0.0)

    def test_joint_margin_and_per_step_velocity(self):
        safety = JointCommandSafety(
            [-1.0, -2.0], [1.0, 2.0], velocity_limit=0.5,
            control_dt=0.1, joint_limit_margin=0.1, torque_limits=[3.0, 4.0],
        )
        q, tau = safety.sanitize([1.0, -2.0], [9.0, -9.0])
        np.testing.assert_allclose(q, [0.9, -1.9])
        np.testing.assert_allclose(tau, [3.0, -4.0])
        np.testing.assert_allclose(safety.rate_limit(q, [0.0, 0.0]), [0.05, -0.05])

    def test_teach_gain_vectors_and_arm_selection(self):
        np.testing.assert_array_equal(
            R1_A7_ArmController._gain_vector(2.0, "kd"),
            np.full(14, 2.0),
        )
        np.testing.assert_array_equal(
            R1_A7_ArmController._arm_selection_mask("left"),
            np.array([True] * 7 + [False] * 7),
        )
        with self.assertRaises(ValueError):
            R1_A7_ArmController._gain_vector(-1.0, "kp")

    def test_waist_dataset_accessors_use_only_waist_yaw(self):
        motor_state = [SimpleNamespace(q=float(i), dq=float(i) / 10.0) for i in range(35)]
        controller = object.__new__(R1_A7_ArmController)
        controller.lowstate_buffer = SimpleNamespace(
            GetData=lambda: SimpleNamespace(motor_state=motor_state)
        )
        controller.msg = SimpleNamespace(
            motor_cmd=[SimpleNamespace(q=100.0 + i) for i in range(35)]
        )
        controller.ctrl_lock = threading.Lock()

        waist_index = R1_A7_JointIndex.kWaistYaw.value
        self.assertEqual(waist_index, 13)
        self.assertEqual(controller.get_current_waist_yaw_q(), 13.0)
        self.assertEqual(controller.get_current_waist_yaw_dq(), 1.3)
        self.assertEqual(controller.get_commanded_waist_yaw_q(), 113.0)

    def test_waist_thumbstick_deadzone_and_rescaling(self):
        controller = object.__new__(R1_A7_ArmController)
        controller.ctrl_lock = threading.Lock()
        controller.tracking_active = True
        controller.waist_deadzone = 0.10
        controller.waist_yaw_velocity_command = 99.0

        self.assertTrue(controller.ctrl_waist_yaw(0.05))
        self.assertEqual(controller.waist_yaw_velocity_command, 0.0)
        self.assertTrue(controller.ctrl_waist_yaw(0.55))
        self.assertAlmostEqual(controller.waist_yaw_velocity_command, 0.5)
        self.assertTrue(controller.ctrl_waist_yaw(-1.0))
        self.assertEqual(controller.waist_yaw_velocity_command, -1.0)

        controller.tracking_active = False
        self.assertFalse(controller.ctrl_waist_yaw(1.0))
        self.assertEqual(controller.waist_yaw_velocity_command, 0.0)

    def test_waist_target_integrates_velocity_and_respects_urdf_limit(self):
        controller = object.__new__(R1_A7_ArmController)
        controller.tracking_active = True
        controller.waist_yaw_target = 2.617
        controller.waist_yaw_velocity_command = 1.0
        controller.waist_velocity_limit = 1.0
        controller.control_dt = 0.004
        self.assertEqual(controller._advance_waist_yaw_target(), 2.618)

        controller.waist_yaw_target = 0.0
        controller.waist_yaw_velocity_command = -0.5
        self.assertAlmostEqual(controller._advance_waist_yaw_target(), -0.002)


class Dex1InputMappingTest(unittest.TestCase):
    def test_controller_uses_full_trigger_travel(self):
        self.assertAlmostEqual(map_dex1_input(10.0, "controller"), 5.4)
        self.assertAlmostEqual(map_dex1_input(5.0, "controller"), 2.7)
        self.assertAlmostEqual(map_dex1_input(0.0, "controller"), 0.0)

    def test_hand_mode_retains_official_pinch_mapping(self):
        self.assertAlmostEqual(map_dex1_input(7.0, "hand"), 5.4)
        self.assertAlmostEqual(map_dex1_input(5.0, "hand"), 0.0)

    def test_held_trigger_converges_without_other_trigger_moving(self):
        filter_ = WeightedMovingFilter(
            np.array([0.5, 0.3, 0.2]), 2, skip_duplicates=False
        )
        filter_.add_data(np.array([0.0, 0.0]))
        filter_.add_data(np.array([0.0, 10.0]))
        filter_.add_data(np.array([10.0, 10.0]))
        self.assertLess(filter_.filtered_data[0], 10.0)

        filter_.add_data(np.array([10.0, 10.0]))
        filter_.add_data(np.array([10.0, 10.0]))

        np.testing.assert_allclose(filter_.filtered_data, np.array([10.0, 10.0]))


class Dex1TorqueLimiterTest(unittest.TestCase):
    @staticmethod
    def limiter(**overrides):
        parameters = {
            "command_step": 0.1,
            "soft_torque_limit": 1.0,
            "hard_torque_limit": 2.0,
            "contact_samples": 2,
            "torque_hysteresis_ratio": 0.8,
            "release_step": 0.2,
            "state_timeout": 0.1,
        }
        parameters.update(overrides)
        return Dex1TorqueLimiter(**parameters)

    def test_monitor_only_mode_still_rate_limits_both_directions(self):
        limiter = self.limiter(
            soft_torque_limit=None,
            hard_torque_limit=None,
        )
        self.assertAlmostEqual(limiter.update(0.0, 3.0, 0.0, 0.01), 2.9)
        self.assertEqual(limiter.state, "monitoring")
        self.assertAlmostEqual(limiter.update(5.4, 3.0, 0.0, 0.01), 3.1)
        self.assertEqual(limiter.state, "opening")

    def test_soft_limit_requires_consecutive_samples_and_holds_command(self):
        limiter = self.limiter()
        self.assertAlmostEqual(limiter.update(0.0, 3.0, 1.1, 0.01), 2.9)
        self.assertEqual(limiter.state, "closing")
        self.assertAlmostEqual(limiter.update(0.0, 2.9, 1.1, 0.01), 2.9)
        self.assertEqual(limiter.state, "soft-limit")
        self.assertAlmostEqual(limiter.update(0.0, 2.85, 1.0, 0.01), 2.9)

    def test_soft_limit_uses_hysteresis_before_resuming_close(self):
        limiter = self.limiter()
        limiter.update(0.0, 3.0, 1.1, 0.01)
        limiter.update(0.0, 2.9, 1.1, 0.01)

        self.assertAlmostEqual(limiter.update(0.0, 2.9, 0.7, 0.01), 2.9)
        self.assertEqual(limiter.state, "soft-limit")
        self.assertAlmostEqual(limiter.update(0.0, 2.9, 0.7, 0.01), 2.8)
        self.assertEqual(limiter.state, "closing")

    def test_hard_limit_immediately_commands_opening_release(self):
        limiter = self.limiter()
        self.assertAlmostEqual(limiter.update(0.0, 2.5, 2.1, 0.01), 2.7)
        self.assertEqual(limiter.state, "hard-release")
        self.assertTrue(limiter.soft_limit_active)

    def test_opening_and_hard_release_never_reverse_an_open_command(self):
        limiter = self.limiter()
        limiter.last_command = 3.0
        self.assertAlmostEqual(limiter.update(5.4, 2.5, 0.0, 0.01), 3.0)
        self.assertAlmostEqual(limiter.update(0.0, 2.5, 2.1, 0.01), 3.0)

    def test_hard_limit_overrides_motion_hold(self):
        limiter = self.limiter()
        self.assertAlmostEqual(
            limiter.update(0.0, 2.5, 2.1, 0.01, motion_allowed=False),
            2.7,
        )
        self.assertEqual(limiter.state, "hard-release")

    def test_stale_state_blocks_close_but_allows_opening(self):
        limiter = self.limiter()
        self.assertAlmostEqual(limiter.update(0.0, 3.0, 0.0, 0.2), 3.0)
        self.assertEqual(limiter.state, "stale-hold")
        self.assertAlmostEqual(limiter.update(5.4, 3.0, 0.0, 0.2), 3.1)
        self.assertEqual(limiter.state, "stale-opening")

    def test_rejects_incomplete_or_reversed_torque_limits(self):
        with self.assertRaises(ValueError):
            Dex1TorqueLimiter(soft_torque_limit=1.0)
        with self.assertRaises(ValueError):
            Dex1TorqueLimiter(soft_torque_limit=2.0, hard_torque_limit=1.0)


class DiagnosticsTest(unittest.TestCase):
    def test_writes_jsonl_and_normalizes_non_finite_age(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diag.jsonl"
            logger = DiagnosticsLogger(path)
            logger.write(
                state="waiting",
                xr_age=np.inf,
                arm_q=np.array([1.0]),
                envelope_status={"left": {"translation_clipped": np.array([True, False, False])}},
            )
            logger.close()
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["state"], "waiting")
            self.assertIsNone(record["xr_age"])
            self.assertEqual(record["arm_q"], [1.0])
            self.assertEqual(
                record["envelope_status"]["left"]["translation_clipped"],
                [True, False, False],
            )


if __name__ == "__main__":
    unittest.main()
