#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Record an R1-A7 trajectory by manually guiding one or both arms.

This is a custom low-level impedance/gravity-compensation mode. Unitree's
published R1-A7 examples do not provide an official drag-teaching API.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return number


def selected_indices(enabled_arms):
    if enabled_arms == "left":
        return np.arange(0, 7)
    if enabled_arms == "right":
        return np.arange(7, 14)
    return np.arange(0, 14)


def next_episode_dir(task_dir):
    task_dir.mkdir(parents=True, exist_ok=True)
    used = []
    for candidate in task_dir.glob("episode_*"):
        try:
            used.append(int(candidate.name.removeprefix("episode_")))
        except ValueError:
            continue
    episode_id = max(used, default=-1) + 1
    episode_dir = task_dir / f"episode_{episode_id:04d}"
    episode_dir.mkdir()
    return episode_dir


def make_sample(index, q, dq, tau, monotonic_ns, wall_time_ns):
    return {
        "idx": index,
        "colors": {},
        "depths": {},
        "states": {
            "left_arm": {
                "qpos": q[:7].tolist(),
                "qvel": dq[:7].tolist(),
                "torque": [],
            },
            "right_arm": {
                "qpos": q[7:].tolist(),
                "qvel": dq[7:].tolist(),
                "torque": [],
            },
            "left_ee": {"qpos": [], "qvel": [], "torque": []},
            "right_ee": {"qpos": [], "qvel": [], "torque": []},
            "body": {"qpos": []},
        },
        "actions": {
            "left_arm": {
                "qpos": q[:7].tolist(),
                "qvel": [],
                "torque": tau[:7].tolist(),
            },
            "right_arm": {
                "qpos": q[7:].tolist(),
                "qvel": [],
                "torque": tau[7:].tolist(),
            },
            "left_ee": {"qpos": [], "qvel": [], "torque": []},
            "right_ee": {"qpos": [], "qvel": [], "torque": []},
            "body": {"qpos": []},
        },
        "tactiles": None,
        "audios": None,
        "sim_state": None,
        "metadata": {
            "record_monotonic_ns": monotonic_ns,
            "record_wall_time_ns": wall_time_ns,
            "recording_mode": "r1_drag_teach",
            "source": "manual_drag",
        },
    }


def save_episode(episode_dir, samples, args):
    payload = {
        "info": {
            "version": "1.0.0",
            "date": date.today().isoformat(),
            "author": "local-r1-a7",
            "recording_mode": "r1_drag_teach",
            "source": "manual_drag",
            "image": {"width": 0, "height": 0, "fps": args.frequency},
            "depth": {"width": 0, "height": 0, "fps": args.frequency},
            "audio": {"sample_rate": 0, "channels": 0, "format": "none", "bits": 0},
            "joint_names": {
                "left_arm": [],
                "left_ee": [],
                "right_arm": [],
                "right_ee": [],
                "body": [],
            },
            "teach_parameters": {
                "enabled_arms": args.enabled_arms,
                "teach_kp": args.teach_kp,
                "teach_kd": args.teach_kd,
                "gravity_scale": args.gravity_scale,
                "transition_duration_s": args.transition_duration,
            },
        },
        "text": {
            "goal": args.task_goal,
            "desc": "R1-A7 manually guided joint trajectory; no XR, cameras, or gripper commands.",
            "steps": "Support selected arm; guide it manually; hold final pose until recording ends.",
        },
        "data": samples,
    }
    final_path = episode_dir / "data.json"
    temporary_path = episode_dir / "data.json.tmp"
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(final_path)
    return final_path


class GravityModel:
    def __init__(self, urdf_path, model_dir):
        import pinocchio as pin

        self.pin = pin
        robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), [str(model_dir)])
        locked_joint_ids = [
            robot.model.getJointId(name)
            for name in ("waist_yaw_joint", "head_pitch_joint", "head_yaw_joint")
        ]
        self.robot = robot.buildReducedRobot(
            locked_joint_ids,
            np.zeros(robot.model.nq),
        )
        joint_names = list(self.robot.model.names)[1:]
        if len(joint_names) != 14 or joint_names[0] != "left_shoulder_pitch_joint":
            raise RuntimeError(f"unexpected R1-A7 gravity-model joint order: {joint_names}")

    def gravity(self, q):
        return np.asarray(
            self.pin.computeGeneralizedGravity(
                self.robot.model,
                self.robot.data,
                np.asarray(q, dtype=np.float64),
            ),
            dtype=np.float64,
        )


def wait_until(deadline):
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(remaining, 0.002))


def command_manual_mode(
    controller,
    gravity_model,
    held_q,
    indices,
    gravity_scale,
    follow_measured=True,
):
    q = controller.get_current_dual_arm_q().copy()
    dq = controller.get_current_dual_arm_dq().copy()
    command_q = held_q.copy()
    if follow_measured:
        command_q[indices] = q[indices]
    tau = np.zeros(14, dtype=np.float64)
    gravity = gravity_model.gravity(q) * gravity_scale
    tau[indices] = gravity[indices]
    if not controller.ctrl_dual_arm(command_q, tau):
        raise RuntimeError("R1-A7 tracking became inactive during manual guidance")
    controller.check_health()
    return q, dq, tau


def transition_gains(
    controller,
    gravity_model,
    held_q,
    indices,
    enabled_arms,
    start_kp,
    start_kd,
    end_kp,
    end_kd,
    start_gravity_scale,
    end_gravity_scale,
    duration,
    follow_measured,
    rate_hz=100.0,
):
    steps = max(1, int(math.ceil(duration * rate_hz)))
    started = time.monotonic()
    for step in range(1, steps + 1):
        ratio = step / steps
        kp = start_kp + (end_kp - start_kp) * ratio
        kd = start_kd + (end_kd - start_kd) * ratio
        gravity_scale = (
            start_gravity_scale
            + (end_gravity_scale - start_gravity_scale) * ratio
        )
        controller.set_arm_joint_gains(kp, kd, enabled_arms)
        command_manual_mode(
            controller,
            gravity_model,
            held_q,
            indices,
            gravity_scale,
            follow_measured=follow_measured,
        )
        wait_until(started + step / rate_hz)


def run_recording(args):
    repo_root = Path(__file__).resolve().parents[2]
    system_root = Path(
        os.environ.get("R1A7_ROOT", repo_root.parents[1])
    ).expanduser().resolve()
    sdk2_python_root = Path(
        os.environ.get(
            "UNITREE_SDK2_PYTHON_PATH",
            system_root / "vendor" / "unitree-sdk2-python",
        )
    ).expanduser().resolve()
    for path in (repo_root, sdk2_python_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from teleop.robot_control.robot_arm import (
        R1_A7_ARM_EFFORT_LIMITS,
        R1_A7_ArmController,
    )
    from teleop.utils.motion_switcher import MotionSwitcher
    from teleop.utils.network import resolve_dds_interface

    interface, renamed = resolve_dds_interface(args.network_interface)
    if renamed:
        print(f"DDS interface {args.network_interface!r} unavailable; using {interface!r}.")
    ChannelFactoryInitialize(0, networkInterface=interface)
    switcher = MotionSwitcher()
    status, result = switcher.Enter_Debug_Mode()
    if status != 0:
        raise RuntimeError(
            f"failed to enter exclusive debug control: status={status}, result={result}"
        )

    gravity_model = GravityModel(
        repo_root / "assets" / "r1" / "r1_a7.urdf",
        repo_root / "assets" / "r1",
    )
    indices = selected_indices(args.enabled_arms)
    controller = None
    samples = []
    normal_completion = False
    try:
        controller = R1_A7_ArmController(
            motion_mode=False,
            simulation_mode=False,
            arm_velocity_limit=30.0,
            joint_limit_margin=0.0,
            rate_limit_mode="official",
            clip_feedforward_torque=True,
        )
        controller.set_tracking_active(True)
        held_q = controller.get_current_dual_arm_q().copy()
        nominal_kp = controller.nominal_arm_kp.copy()
        nominal_kd = controller.nominal_arm_kd.copy()

        initial_gravity = gravity_model.gravity(held_q) * args.gravity_scale
        if np.any(np.abs(initial_gravity[indices]) > R1_A7_ARM_EFFORT_LIMITS[indices]):
            raise RuntimeError("gravity model exceeds an official arm effort limit")
        print("Selected arm joints are currently held with nominal position stiffness.")
        input(
            "Support the selected arm(s) with both hands. Press Enter to begin the "
            "impedance transition, or Ctrl+C to cancel: "
        )
        for second in range(args.countdown, 0, -1):
            print(f"Manual-guidance transition begins in {second}s.")
            time.sleep(1.0)
        controller.set_tracking_active(True)
        held_q = controller.get_current_dual_arm_q().copy()

        transition_gains(
            controller,
            gravity_model,
            held_q,
            indices,
            args.enabled_arms,
            nominal_kp,
            nominal_kd,
            np.full(14, args.teach_kp),
            np.full(14, args.teach_kd),
            0.0,
            args.gravity_scale,
            args.transition_duration,
            follow_measured=True,
        )

        print(
            f"RECORDING for {args.duration:.1f}s. Guide the arm slowly and keep "
            "supporting it; hold the final pose before time expires."
        )
        period = 1.0 / args.frequency
        started = time.monotonic()
        next_status = started
        sample_index = 0
        while True:
            sample_started = time.monotonic()
            if sample_started - started >= args.duration:
                break
            q, dq, tau = command_manual_mode(
                controller,
                gravity_model,
                held_q,
                indices,
                args.gravity_scale,
            )
            if np.any(np.abs(tau[indices]) > R1_A7_ARM_EFFORT_LIMITS[indices]):
                raise RuntimeError("gravity command exceeded an official arm effort limit")
            samples.append(
                make_sample(
                    sample_index,
                    q,
                    dq,
                    tau,
                    time.monotonic_ns(),
                    time.time_ns(),
                )
            )
            sample_index += 1
            if sample_started >= next_status:
                remaining = max(0.0, args.duration - (sample_started - started))
                print(f"Recording: {remaining:5.1f}s remaining")
                next_status = sample_started + 1.0
            wait_until(started + sample_index * period)

        final_q = controller.get_current_dual_arm_q().copy()
        held_q[indices] = final_q[indices]
        print("Recording complete. Keep supporting the arm while stiffness is restored.")
        transition_gains(
            controller,
            gravity_model,
            held_q,
            indices,
            args.enabled_arms,
            np.full(14, args.teach_kp),
            np.full(14, args.teach_kd),
            nominal_kp,
            nominal_kd,
            args.gravity_scale,
            0.0,
            args.transition_duration,
            follow_measured=False,
        )
        controller.restore_nominal_arm_joint_gains(args.enabled_arms)
        controller.ctrl_dual_arm(held_q, np.zeros(14))
        time.sleep(0.5)

        if len(samples) < 2:
            raise RuntimeError("recording produced fewer than two samples")
        episode_dir = next_episode_dir(args.task_dir / args.task_name)
        final_path = save_episode(episode_dir, samples, args)
        normal_completion = True
        print(f"Saved {len(samples)} samples to {final_path}")
        input("Arm stiffness is restored. Press Enter to stop command publishing: ")
        return final_path
    finally:
        if controller is not None:
            try:
                current_q = controller.get_current_dual_arm_q().copy()
                controller.restore_nominal_arm_joint_gains(args.enabled_arms)
                controller.ctrl_dual_arm(current_q, np.zeros(14))
                time.sleep(0.2)
            except Exception as exc:
                print(f"WARNING: failed to restore nominal arm hold: {exc}", file=sys.stderr)
            controller.shutdown(go_home=False)
        if samples and not normal_completion:
            print(
                "Recording was interrupted and was not saved; selected-arm stiffness "
                "restoration was attempted.",
                file=sys.stderr,
            )


def build_parser():
    repo_root = Path(__file__).resolve().parents[2]
    system_root = Path(
        os.environ.get("R1A7_ROOT", repo_root.parents[1])
    ).expanduser().resolve()
    parser = argparse.ArgumentParser(
        description=(
            "Record R1-A7 arm encoder trajectories by custom manual guidance; "
            "no PICO, cameras, or gripper commands are used."
        )
    )
    parser.add_argument("--enabled-arms", choices=("left", "right", "both"), default="left")
    parser.add_argument("--network-interface", default="enx00e04c187010")
    parser.add_argument("--duration", type=positive_float, default=15.0)
    parser.add_argument("--frequency", type=positive_float, default=100.0)
    parser.add_argument("--teach-kp", type=nonnegative_float, default=0.0)
    parser.add_argument("--teach-kd", type=nonnegative_float, default=2.0)
    parser.add_argument("--gravity-scale", type=nonnegative_float, default=1.0)
    parser.add_argument("--transition-duration", type=positive_float, default=2.0)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=system_root / "runtime" / "datasets" / "r1a7_drag_teach_raw",
    )
    parser.add_argument("--task-name", default="drag_teach_left_01")
    parser.add_argument(
        "--task-goal",
        default="Manually guide the selected R1-A7 arm along a fixed-point trajectory.",
    )
    parser.add_argument("--confirm-real-robot", action="store_true")
    parser.add_argument(
        "--confirm-custom-gravity-compensation",
        action="store_true",
        help="Acknowledge that this is not an official Unitree drag-teaching mode",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.confirm_real_robot:
        parser.error("real R1-A7 manual guidance requires --confirm-real-robot")
    if not args.confirm_custom_gravity_compensation:
        parser.error(
            "manual guidance requires --confirm-custom-gravity-compensation; "
            "Unitree does not publish an official R1-A7 drag-teaching API"
        )
    if args.countdown < 0:
        parser.error("--countdown must be non-negative")
    if args.frequency > 250.0:
        parser.error("--frequency cannot exceed the 250 Hz arm command loop")
    args.task_dir = args.task_dir.expanduser().resolve()
    try:
        run_recording(args)
    except KeyboardInterrupt:
        print("Cancelled by operator.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
