#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Authorized local bridge from validated 16D actions to R1-A7 controllers."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from multiprocessing import Array, Lock, Value

import numpy as np
import zmq

from unitree_lerobot.r1a7_contract import validate_action_vector
from unitree_lerobot.r1a7_dagger import OwnershipRejected
from unitree_lerobot.r1a7_hardware_gateway import (
    RealRobotAuthorization,
    load_hardened_controller_types,
)


REAL_ENV_NAME = "R1A7_REAL_INFERENCE"
REAL_ENV_VALUE = "YES"
DEX1_PROFILES = {
    "legacy": {"kp": 2.0, "kd": 0.05, "command_step": 0.18},
    "matched-teleop": {"kp": 3.0, "kd": 0.05, "command_step": 0.24},
}


class R1A7RealCommandController:
    """Apply validated actions and latch on command-stream failure."""

    def __init__(
        self,
        arm,
        gripper,
        left_gripper_input,
        right_gripper_input,
        gripper_motion_ready,
        gravity_model,
        *,
        command_timeout: float,
        pre_policy_timeout: float,
    ):
        self.arm = arm
        self.gripper = gripper
        self.left_gripper_input = left_gripper_input
        self.right_gripper_input = right_gripper_input
        self.gripper_motion_ready = gripper_motion_ready
        self.gravity_model = gravity_model
        self.command_timeout = float(command_timeout)
        self.pre_policy_timeout = float(pre_policy_timeout)
        if self.pre_policy_timeout <= self.command_timeout:
            raise ValueError("pre-policy timeout must be greater than command timeout")
        self.active = False
        self.fault_reason: str | None = None
        self.last_command_monotonic = 0.0
        self.command_count = 0
        self.hold_count = 0
        self.pre_policy_hold_count = 0
        self.policy_watchdog_active = False
        self.pre_policy_arm_target: np.ndarray | None = None
        self.waist_hold_target: float | None = None
        self.last_action = None
        self.last_action_ns = 0
        self.dagger = None
        self.gripper_action = None
        self.gripper_data_lock = None
        self.last_action_source = "hold"
        self.last_action_epoch = 0

    def _hold(self) -> None:
        with self.gripper_motion_ready.get_lock():
            self.gripper_motion_ready.value = False
        self.arm.set_tracking_active(False)

    def latch_fault(self, reason: str) -> None:
        if self.fault_reason is None:
            self.fault_reason = str(reason)
        self.active = False
        self.policy_watchdog_active = False
        self._hold()

    def check_timeout(self) -> None:
        if self.active and not self.policy_watchdog_active:
            if (
                time.monotonic() - self.last_command_monotonic
                > self.pre_policy_timeout
            ):
                self.latch_fault(
                    "first policy command did not arrive within "
                    f"{self.pre_policy_timeout:.3f}s; restart required"
                )
                return
            try:
                self.arm.check_health()
                if self.pre_policy_arm_target is None:
                    raise RuntimeError("pre-policy hold target is unavailable")
                arm_tau = self.gravity_model.gravity(self.pre_policy_arm_target)
                if not self.arm.ctrl_dual_arm(self.pre_policy_arm_target, arm_tau):
                    raise RuntimeError("R1-A7 arm controller rejected pre-policy hold")
                self.arm.ctrl_waist_yaw(0.0)
                self.pre_policy_hold_count += 1
            except Exception as exc:
                self.latch_fault(f"pre-policy hold failed: {exc}")
            return
        if (
            self.policy_watchdog_active
            and time.monotonic() - self.last_command_monotonic > self.command_timeout
        ):
            self.latch_fault(
                f"policy command timeout exceeded {self.command_timeout:.3f}s; restart required"
            )

    def execute(self, values, *, policy_command: bool = True) -> dict:
        if self.dagger is not None and self.dagger.disarmed:
            raise RuntimeError("DAgger trial disarmed; restart required")
        if self.fault_reason is not None:
            raise RuntimeError(f"real command bridge is fault-locked: {self.fault_reason}")
        action = validate_action_vector(values)
        self.arm.check_health()

        # Stage gripper targets, but enable its thread only after the arm accepts.
        with self.left_gripper_input.get_lock():
            self.left_gripper_input.value = float(action[14] / 5.4 * 10.0)
        with self.right_gripper_input.get_lock():
            self.right_gripper_input.value = float(action[15] / 5.4 * 10.0)
        if not self.active:
            self.waist_hold_target = float(self.arm.get_current_waist_yaw_q())
            self.arm.set_tracking_active(True)

        arm_tau = self.gravity_model.gravity(action[:14])
        accepted = self.arm.ctrl_dual_arm(action[:14], arm_tau)
        if not accepted:
            raise RuntimeError("R1-A7 arm controller did not accept an active command")
        # Waist is deliberately outside the policy contract. Keep it at the
        # activation pose through the hardened controller's hold mode.
        self.arm.ctrl_waist_yaw(0.0)
        if not self.active:
            with self.gripper_motion_ready.get_lock():
                self.gripper_motion_ready.value = True
            self.active = True
        self.pre_policy_arm_target = action[:14].copy()
        self.policy_watchdog_active = bool(policy_command)
        self.last_command_monotonic = time.monotonic()
        self.command_count += 1
        self.last_action = action.copy()
        self.last_action_ns = time.monotonic_ns()
        self.last_action_source = self.dagger.owner.source if self.dagger else "policy"
        self.last_action_epoch = self.dagger.owner.epoch if self.dagger else 0
        return {
            "status": "ok",
            "command_count": self.command_count,
            "hardware_commands_sent": True,
            "policy_watchdog_active": self.policy_watchdog_active,
            "waist_mode": "hold_at_activation",
            "waist_hold_target": self.waist_hold_target,
            "executed_action": action.tolist(),
        }

    def hold_current(self, *, preserve_grip=False) -> dict:
        """Hold measured joints while keeping a healthy policy stream alive."""

        if self.fault_reason is not None:
            raise RuntimeError(f"real command bridge is fault-locked: {self.fault_reason}")
        self.arm.check_health()
        arm_q = self.arm.get_current_dual_arm_q()
        gripper_q, _torque, _ages = self.gripper._snapshot_gripper_state()
        grip_target = (self.last_action[14:].copy()
                       if preserve_grip and self.last_action is not None else gripper_q)
        if self.active:
            with self.left_gripper_input.get_lock():
                self.left_gripper_input.value = float(grip_target[0] / 5.4 * 10.0)
            with self.right_gripper_input.get_lock():
                self.right_gripper_input.value = float(grip_target[1] / 5.4 * 10.0)
            arm_tau = self.gravity_model.gravity(arm_q)
            accepted = self.arm.ctrl_dual_arm(arm_q, arm_tau)
            if not accepted:
                raise RuntimeError("R1-A7 arm controller did not accept hold command")
            self.arm.ctrl_waist_yaw(0.0)
            self.last_command_monotonic = time.monotonic()
            self.hold_count += 1
            self.last_action = np.r_[arm_q, grip_target]
            self.last_action_ns = time.monotonic_ns()
            self.last_action_source = "hold"
            self.last_action_epoch = self.dagger.owner.epoch if self.dagger else 0
            self.pre_policy_arm_target = arm_q.copy()
        return {
            "status": "ok",
            "active": self.active,
            "policy_watchdog_active": self.policy_watchdog_active,
            "pre_policy_hold_count": self.pre_policy_hold_count,
            "hold_count": self.hold_count,
            "hardware_commands_sent": self.active,
            "held_state": [*arm_q.tolist(), *grip_target.tolist()],
            "measured_waist_yaw": float(self.arm.get_current_waist_yaw_q()),
        }

    def state_snapshot(self) -> dict:
        self.arm.check_health()
        gripper_q, _torque, gripper_ages = self.gripper._snapshot_gripper_state()
        now = time.monotonic()
        state = np.concatenate(
            (
                self.arm.get_current_dual_arm_q(),
                gripper_q,
            )
        )
        reply = {
            "status": "ok",
            "state": state.tolist(),
            "sample_monotonic_ns": {
                "lowstate": int(self.arm.last_lowstate_time * 1e9),
                "left_gripper": int((now - gripper_ages[0]) * 1e9),
                "right_gripper": int((now - gripper_ages[1]) * 1e9),
            },
            "bridge_monotonic_ns": time.monotonic_ns(),
            "active": self.active,
            "policy_watchdog_active": self.policy_watchdog_active,
            "fault_reason": self.fault_reason,
            "waist_mode": "hold_at_activation",
            "waist_hold_target": self.waist_hold_target,
            "measured_waist_yaw": float(self.arm.get_current_waist_yaw_q()),
            "hardware_commands_sent": self.command_count > 0,
        }
        if self.dagger is not None:
            reply.update(self.dagger.owner.snapshot())
            reply["dagger_enabled"] = self.dagger.enabled
            reply["takeover_reason"] = self.dagger.reason
            reply["xr_ready"] = self.dagger.xr_ready
            reply["right_squeeze"] = getattr(self.dagger, "right_squeeze", False)
            reply["start_press_sequence"] = getattr(self.dagger, "start_press_sequence", 0)
            reply["restart_press_sequence"] = getattr(self.dagger, "restart_press_sequence", 0)
            reply["operator_finish"] = getattr(self.dagger, "operator_finish", None)
            reply["trigger_pickup"] = (self.dagger.alignment.trigger_latched
                                       if self.dagger.alignment else [False, False])
            reply["takeover_blend"] = (self.dagger.alignment.blend_snapshot()
                                       if self.dagger.alignment else None)
            if self.last_action is not None:
                action = self.last_action.copy()
                if self.gripper_action is not None:
                    with self.gripper_data_lock:
                        action[14:] = np.asarray(self.gripper_action[:])
                reply["commanded_action"] = action.tolist()
                reply["requested_action"] = self.last_action.tolist()
                reply["command_monotonic_ns"] = self.last_action_ns
                reply["command_source"] = self.last_action_source
                reply["command_owner_epoch"] = self.last_action_epoch
        return reply

    def disarm(self) -> dict:
        if self.dagger is not None:
            self.dagger.enabled = False
            self.dagger.disarmed = True
            self.dagger.owner.transition("hold")
        self.active = False
        self.policy_watchdog_active = False
        self.pre_policy_arm_target = None
        self._hold()
        return {
            "status": "ok",
            "command_count": self.command_count,
            "active": False,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xr-teleoperate-root", required=True)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--bind", default="tcp://127.0.0.1:55582")
    parser.add_argument("--command-timeout", type=float, default=0.5)
    parser.add_argument("--pre-policy-timeout", type=float, default=5.0)
    parser.add_argument("--dex1-profile", choices=DEX1_PROFILES, default="legacy")
    parser.add_argument("--confirm-real-robot", required=True)
    parser.add_argument("--dagger", action="store_true")
    from unitree_lerobot.r1a7_dagger import takeover_blend_seconds
    parser.add_argument("--takeover-blend-seconds", type=takeover_blend_seconds, default=3.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dex1 = DEX1_PROFILES[args.dex1_profile]
    if os.environ.get(REAL_ENV_NAME) != REAL_ENV_VALUE:
        raise PermissionError(f"set {REAL_ENV_NAME}={REAL_ENV_VALUE} for real inference")
    # A background process can inherit SIGTERM as ignored from its launching
    # shell. Keep startup stoppable before the graceful handler is installed.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    authorization = RealRobotAuthorization(
        allow_real_robot=True,
        confirmation=args.confirm_real_robot,
    )
    print('{"event":"bridge_initializing","phase":"load_controllers"}', flush=True)
    controller_types = load_hardened_controller_types(
        args.xr_teleoperate_root, authorization
    )
    from teleop.robot_control.r1_a7_gravity import R1A7GravityModel
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    print('{"event":"bridge_initializing","phase":"dds_domain"}', flush=True)
    ChannelFactoryInitialize(0, networkInterface=args.network_interface)
    left_input = Value("d", 0.0, lock=True)
    right_input = Value("d", 0.0, lock=True)
    gripper_motion_ready = Value("b", False, lock=True)
    gripper_data_lock = Lock()
    gripper_state = Array("d", 2, lock=False)
    gripper_action = Array("d", 2, lock=False)

    arm = None
    gripper = None
    controller = None
    socket = None
    dagger = None
    try:
        gravity_model = R1A7GravityModel(args.xr_teleoperate_root)
        print('{"event":"bridge_initializing","phase":"arm_lowstate"}', flush=True)
        arm = controller_types.arm(
            motion_mode=False,
            simulation_mode=False,
            lowstate_timeout=0.25,
            command_timeout=args.command_timeout,
            arm_velocity_limit=30.0,
            joint_limit_margin=0.0,
            rate_limit_mode="official",
            clip_feedforward_torque=False,
            waist_velocity_limit=1.0,
            waist_deadzone=0.1,
        )
        print('{"event":"bridge_initializing","phase":"dex1_state"}', flush=True)
        gripper = controller_types.gripper(
            left_input,
            right_input,
            gripper_data_lock,
            gripper_state,
            gripper_action,
            filter=True,
            fps=200.0,
            simulation_mode=False,
            xr_motion_data_ready_in=gripper_motion_ready,
            enabled_arms="both",
            input_mode="controller",
            kp=dex1["kp"],
            kd=dex1["kd"],
            command_step=dex1["command_step"],
            runtime_state_timeout=0.1,
        )
        controller = R1A7RealCommandController(
            arm,
            gripper,
            left_input,
            right_input,
            gripper_motion_ready,
            gravity_model,
            command_timeout=args.command_timeout,
            pre_policy_timeout=args.pre_policy_timeout,
        )
        if args.dagger:
            from r1a7_dagger_pico import DaggerPico
            dagger = DaggerPico(controller, args.xr_teleoperate_root,
                                takeover_blend_seconds=args.takeover_blend_seconds)
            controller.dagger = dagger
            controller.gripper_action = gripper_action
            controller.gripper_data_lock = gripper_data_lock

        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(args.bind)
        stopped = threading.Event()
        signal.signal(signal.SIGINT, lambda _s, _f: stopped.set())
        signal.signal(signal.SIGTERM, lambda _s, _f: stopped.set())
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        print(
            json.dumps(
                {
                    "status": "armed-hold",
                    "bind": args.bind,
                    "profile": "official",
                    "arm_velocity_limit": 30.0,
                    "joint_limit_margin": 0.0,
                    "waist_mode": "hold_at_activation",
                    "policy_dimension": 16,
                    "policy_waist_action": "not_in_policy_contract",
                    "arm_feedforward": "r1a7_model_gravity_rnea",
                    "policy_watchdog_active": False,
                    "pre_policy_timeout": args.pre_policy_timeout,
                    "dex1_profile": args.dex1_profile,
                    "dex1": dex1,
                    "hardware_commands_sent": False,
                }
            ),
            flush=True,
        )
        while not stopped.is_set():
            if dagger is not None and controller.fault_reason is None:
                try:
                    dagger.tick()
                except Exception as exc:
                    controller.latch_fault(f"PICO takeover failed: {exc}")
            controller.check_timeout()
            if socket not in dict(poller.poll(20)):
                continue
            request = socket.recv_json()
            try:
                kind = request.get("request")
                if kind == "state":
                    reply = controller.state_snapshot()
                elif kind == "dagger_heartbeat" and dagger is not None:
                    if dagger.disarmed:
                        raise RuntimeError("DAgger trial already disarmed")
                    dagger.last_host_heartbeat = time.monotonic()
                    dagger.enabled = True
                    reply = controller.state_snapshot()
                elif kind == "dagger_start_ready" and dagger is not None:
                    dagger.initializing = False
                    dagger.last_host_heartbeat = time.monotonic()
                    reply = controller.state_snapshot()
                elif kind == "dagger_between_trials" and dagger is not None:
                    dagger.wait_between_trials()
                    reply = controller.state_snapshot()
                elif kind == "dagger_prepare_restart" and dagger is not None:
                    dagger.prepare_restart()
                    reply = controller.state_snapshot()
                elif kind == "dagger_ack_finish" and dagger is not None:
                    dagger.operator_finish = None
                    reply = controller.state_snapshot()
                elif kind == "takeover_hold" and dagger is not None:
                    dagger.hold("operator console; press PICO A to align")
                    reply = controller.state_snapshot()
                elif kind == "command":
                    if dagger is not None:
                        dagger.owner.check("policy", request.get("owner_epoch"))
                        dagger.last_host_heartbeat = time.monotonic()
                    reply = controller.execute(
                        request.get("action"),
                        policy_command=bool(request.get("policy_command", True)),
                    )
                elif kind == "hold":
                    if dagger is not None:
                        dagger.owner.check("policy", request.get("owner_epoch"))
                        dagger.last_host_heartbeat = time.monotonic()
                    # Policy waits/pauses must not replace the closing target with
                    # measured contact opening and thereby relax an existing grasp.
                    reply = controller.hold_current(preserve_grip=True)
                elif kind == "disarm":
                    reply = controller.disarm()
                else:
                    reply = {"status": "error", "message": "invalid request"}
            except OwnershipRejected as exc:
                reply = {"status": "rejected", "message": str(exc),
                         "fault_locked": False, "takeover_reason": dagger.reason,
                         "xr_ready": dagger.xr_ready, **dagger.owner.snapshot()}
            except Exception as exc:
                controller.latch_fault(str(exc))
                reply = {"status": "error", "message": str(exc), "fault_locked": True}
            socket.send_json(reply)
    finally:
        if controller is not None:
            controller.disarm()
        if socket is not None:
            socket.close(linger=0)
        if dagger is not None:
            dagger.close()
        if gripper is not None:
            gripper.shutdown()
        if arm is not None:
            arm.shutdown(go_home=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
