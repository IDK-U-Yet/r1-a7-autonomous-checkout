# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""PICO input adapter; no DDS publisher is created here."""
from __future__ import annotations
import time
import os
import numpy as np
from unitree_lerobot.r1a7_dagger import Ownership, OwnershipRejected, RelativeAlignment


class DaggerPico:
    def __init__(self, controller, xr_root, *, takeover_blend_seconds=3.0):
        from televuer import TeleVuerWrapper
        from teleop.robot_control.robot_arm_ik import R1_A7_ArmIK
        self.controller = controller
        self.takeover_blend_seconds = takeover_blend_seconds
        self.owner = Ownership()
        previous = os.getcwd()
        try:
            os.chdir(os.path.join(xr_root, "teleop"))
            self.ik = R1_A7_ArmIK()
        finally:
            os.chdir(previous)
        self.xr = TeleVuerWrapper(use_hand_tracking=False, binocular=False,
                                 img_shape=(360, 640), display_mode="pass-through",
                                 zmq=False, webrtc=False, arm_reference_mode="head_yaw")
        self.alignment = None
        self.last_buttons = (False, False)
        self.last_finish_buttons = (False, False)
        self.operator_finish = None
        self.next_tick = 0.0
        self.last_host_heartbeat = time.monotonic()
        self.enabled = False
        self.disarmed = False
        self.initializing = True
        self.reason = "standby"
        self.xr_ready = False
        self.right_squeeze = False
        self.start_press_sequence = 0
        self.restart_press_sequence = 0
        self.previous_start_squeeze = None
        self.between_trials = False
        self.finalization_deadline = 0.0

    def wait_between_trials(self):
        self.hold("trial ended; clear workspace then R to restart")
        self.between_trials = True
        self.operator_finish = None
        self.finalization_deadline = time.monotonic() + 12.0

    def prepare_restart(self):
        if (self.disarmed or self.controller.fault_reason is not None
                or not self.between_trials or not self.xr_ready):
            raise OwnershipRejected("restart requires healthy HOLD and fresh PICO")
        self.owner.restart_trial()
        self.between_trials = False
        self.initializing = True
        self.operator_finish = None
        self.alignment = None
        self.last_host_heartbeat = time.monotonic()
        self.reason = "explicit R restart"

    def update_start_button(self, tele, fresh):
        value = float(getattr(tele, "right_ctrl_squeezeValue", 0.0))
        # Some WebXR clients provide analog depth without the pressed boolean.
        pressed = bool(tele.right_ctrl_squeeze) or value >= 0.8
        if not pressed and value > 0.2:
            pressed = self.right_squeeze  # Hysteresis against a partially held grip.
        self.right_squeeze = pressed
        previous = getattr(self, "previous_start_squeeze", None)
        if fresh and not self.enabled and self.initializing and previous is False and pressed:
            self.start_press_sequence = getattr(self, "start_press_sequence", 0) + 1
            print(f"PICO RIGHT GRIP start edge={self.start_press_sequence}, value={value:.2f}", flush=True)
        if (fresh and self.enabled and not self.initializing and previous is False and pressed
                and (getattr(self, "between_trials", False) or self.owner.source == "hold")):
            self.restart_press_sequence = getattr(self, "restart_press_sequence", 0) + 1
            print(f"PICO RIGHT GRIP restart edge={self.restart_press_sequence}", flush=True)
        self.previous_start_squeeze = pressed if fresh else None

    def fk(self, q):
        import pinocchio as pin
        model, data = self.ik.reduced_robot.model, self.ik.reduced_robot.data
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return [data.oMf[i].homogeneous.copy() for i in (self.ik.L_hand_id, self.ik.R_hand_id)]

    def hold(self, reason):
        self.owner.transition("hold")
        self.reason = reason
        self.alignment = None
        self.controller.hold_current(preserve_grip=True)
        print(f"DAgger HOLD owner_epoch={self.owner.epoch}: {reason}", flush=True)

    def commit_ready(self, tele):
        """IK is synchronous: revalidate its captured inputs at commit time."""
        now = time.monotonic()
        if self.disarmed or self.controller.fault_reason is not None:
            return False
        if now - self.last_host_heartbeat > 0.75:
            self.controller.latch_fault("coordinator heartbeat expired during IK")
            return False
        latest = self.xr.get_tele_data()
        # The XR process can publish a new timestamp during the read. Sample
        # validation time AFTER reading, not before, to avoid false negative age.
        now = time.monotonic()
        if latest.left_ctrl_aButton or latest.left_ctrl_bButton:
            self.hold("completion button during IK")
            return False
        buttons = (latest.right_ctrl_aButton, latest.right_ctrl_bButton)
        # Releasing activation A is normal; only a new A press or B stops this solve.
        if buttons[1] or (buttons[0] and not self.last_buttons[0]):
            self.hold("operator button changed during IK; press A to realign")
            return False
        if (not latest.motion_data_ready or not 0 <= now - latest.motion_data_timestamp <= 0.25
                or not 0 <= now - tele.motion_data_timestamp <= 0.25):
            self.hold(f"XR expired during IK: latest_age={now - latest.motion_data_timestamp:.3f}s, "
                      f"input_age={now - tele.motion_data_timestamp:.3f}s; press A to realign")
            return False
        return True

    def tick(self):
        if self.disarmed:
            return
        now = time.monotonic()
        if now < self.next_tick:
            return
        self.next_tick = now + 1 / 30
        tele = self.xr.get_tele_data()
        now = time.monotonic()
        buttons = (tele.right_ctrl_aButton, tele.right_ctrl_bButton)
        a, b = [value and not old for value, old in zip(buttons, self.last_buttons)]
        self.last_buttons = buttons
        age = now - tele.motion_data_timestamp
        fresh = tele.motion_data_ready and 0 <= age <= 0.25
        self.xr_ready = bool(fresh)
        self.update_start_button(tele, fresh)
        finish_buttons = (tele.left_ctrl_aButton, tele.left_ctrl_bButton)
        x, y = [value and not old for value, old in zip(finish_buttons, self.last_finish_buttons)]
        self.last_finish_buttons = finish_buttons
        if not self.enabled:
            return
        finalizing = getattr(self, "between_trials", False) and now < self.finalization_deadline
        if now - self.last_host_heartbeat > 0.75 and not finalizing:
            self.controller.latch_fault("DAgger recorder/coordinator heartbeat lost")
            return
        if getattr(self, "between_trials", False):
            self.controller.hold_current(preserve_grip=True)
            return
        if fresh and not self.initializing and (x or y):
            # Sticky until coordinator acknowledgement; Y wins simultaneous presses.
            if self.operator_finish != "Q":
                self.operator_finish = "Q" if y else "S"
            self.hold("operator Y reject" if y else "operator X complete")
            return
        if self.operator_finish is not None:
            self.controller.hold_current(preserve_grip=True)
            return
        if b:
            self.hold("operator B")
        elif a and fresh:
            if self.initializing:
                self.hold("startup interrupted; restart launcher")
                return
            self.hold("takeover alignment")
            q = self.controller.arm.get_current_dual_arm_q()
            goals = (self.controller.last_action[14:].copy() if self.controller.last_action is not None
                     else self.controller.gripper._snapshot_gripper_state()[0])
            self.alignment = RelativeAlignment(
                self.fk(q), [tele.left_wrist_pose, tele.right_wrist_pose], goals,
                blend_seconds=getattr(self, "takeover_blend_seconds", 3.0))
            self.ik.init_data = q.copy()
            self.ik.smooth_filter._data_queue = []
            self.ik.smooth_filter.add_data(q.copy())
            self.owner.transition("human")
            self.reason = "human aligned; move triggers to held grip before they become active"
            print(f"HUMAN owner_epoch={self.owner.epoch}; trigger pickup alignment active", flush=True)
            return  # Never solve or move on the activation edge.
        if not fresh:
            if self.owner.source != "hold":
                self.hold(f"stale XR: ready={tele.motion_data_ready}, age={age:.3f}s; press A to realign")
            if tele.motion_data_ready and age > 1.0:
                self.controller.latch_fault("XR motion missing for >1 s during intervention")
            else:
                self.controller.hold_current(preserve_grip=True)
            return
        if self.owner.source == "policy":
            return
        if self.owner.source == "hold":
            self.controller.hold_current(preserve_grip=True)
            return
        try:
            poses = self.alignment.poses([tele.left_wrist_pose, tele.right_wrist_pose])
        except ValueError as exc:
            self.hold(f"invalid takeover target: {exc}; press A to realign")
            return
        q = self.controller.arm.get_current_dual_arm_q()
        ik_started = time.monotonic()
        try:
            solved, _ = self.ik.solve_ik(poses[0], poses[1], q, self.controller.arm.get_current_dual_arm_dq())
        finally:
            elapsed = time.monotonic() - ik_started
            if elapsed > 0.1:
                print(f"DAgger slow IK: {elapsed * 1000:.1f} ms; owner_epoch={self.owner.epoch}", flush=True)
        # Match ordinary collection: use solve_ik's smoothed/fallback result.
        # Do not latch HOLD solely for FK residual; command validation, hard
        # joint limits and the XR freshness check below remain in force.
        grip = self.alignment.grippers([tele.left_ctrl_triggerValue, tele.right_ctrl_triggerValue])
        if not self.commit_ready(tele):
            return
        self.controller.execute(np.r_[solved, grip], policy_command=True)

    def close(self):
        self.xr.close()
