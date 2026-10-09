#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Inspect or replay an R1-A7 joint-only teleoperation episode.

The reported errors use the robot's joint encoders. They measure command
tracking and encoder repeatability, not independent Cartesian accuracy.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


JOINT_NAMES = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)


@dataclass
class Trajectory:
    path: Path
    q_target: np.ndarray
    tau_target: np.ndarray
    q_state: np.ndarray
    dq_state: np.ndarray
    time_s: np.ndarray
    configured_fps: float
    recording_mode: str
    recorded_enabled_arms: str | None
    recorded_torque: bool
    recorded_velocity: bool


def _vector(sample, root, field, *, required=True):
    values = []
    for side in ("left_arm", "right_arm"):
        value = (sample.get(root) or {}).get(side, {}).get(field)
        if value is None or len(value) == 0:
            if required:
                raise ValueError(f"missing {root}.{side}.{field}")
            return None
        values.extend(value)
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (14,):
        raise ValueError(
            f"{root} arm {field} has shape {vector.shape}, expected (14,)"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{root} arm {field} contains a non-finite value")
    return vector


def resolve_episode(path):
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    direct = path / "data.json"
    if direct.is_file():
        return direct
    candidates = sorted(path.glob("episode_*/data.json"))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(
            f"{path} contains {len(candidates)} episodes; select one episode directory"
        )
    raise ValueError(f"no data.json found at {path}")


def load_trajectory(path):
    json_path = resolve_episode(path)
    episode = json.loads(json_path.read_text(encoding="utf-8"))
    samples = episode.get("data")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("episode must contain at least two samples")

    fps_value = episode.get("info", {}).get("image", {}).get("fps", 30.0)
    try:
        fps = float(fps_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid configured fps: {fps_value!r}") from exc
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"configured fps must be positive, got {fps_value!r}")

    q_target = []
    tau_target = []
    q_state = []
    dq_state = []
    timestamps = []
    has_all_torque = True
    has_all_velocity = True
    for index, sample in enumerate(samples):
        try:
            q_target.append(_vector(sample, "actions", "qpos"))
            q_state.append(_vector(sample, "states", "qpos"))
            tau = _vector(sample, "actions", "torque", required=False)
            dq = _vector(sample, "states", "qvel", required=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"sample {index}: {exc}") from exc
        if tau is None:
            has_all_torque = False
            tau = np.zeros(14, dtype=np.float64)
        if dq is None:
            has_all_velocity = False
            dq = np.zeros(14, dtype=np.float64)
        tau_target.append(tau)
        dq_state.append(dq)
        timestamp = (sample.get("metadata") or {}).get("record_monotonic_ns")
        timestamps.append(timestamp)

    if all(isinstance(value, int) for value in timestamps):
        time_s = (np.asarray(timestamps, dtype=np.int64) - timestamps[0]) / 1e9
        if np.any(np.diff(time_s) <= 0.0):
            raise ValueError("record_monotonic_ns must be strictly increasing")
    else:
        time_s = np.arange(len(samples), dtype=np.float64) / fps

    if not has_all_torque:
        tau_target = np.zeros((len(samples), 14), dtype=np.float64)

    return Trajectory(
        path=json_path,
        q_target=np.vstack(q_target),
        tau_target=np.vstack(tau_target),
        q_state=np.vstack(q_state),
        dq_state=np.vstack(dq_state),
        time_s=np.asarray(time_s, dtype=np.float64),
        configured_fps=fps,
        recording_mode=str(episode.get("info", {}).get("recording_mode", "unmarked")),
        recorded_enabled_arms=(
            episode.get("info", {}).get("teach_parameters", {}).get("enabled_arms")
        ),
        recorded_torque=has_all_torque,
        recorded_velocity=has_all_velocity,
    )


def selected_indices(enabled_arms):
    if enabled_arms == "left":
        return np.arange(0, 7)
    if enabled_arms == "right":
        return np.arange(7, 14)
    return np.arange(0, 14)


def _joint_metrics(error_rad, indices):
    result = {}
    error_deg = np.rad2deg(error_rad)
    for index in indices:
        values = error_deg[:, index]
        result[JOINT_NAMES[index]] = {
            "rmse_deg": float(np.sqrt(np.mean(np.square(values)))),
            "max_abs_deg": float(np.max(np.abs(values))),
            "mean_deg": float(np.mean(values)),
        }
    selected = error_deg[:, indices]
    result["all_selected"] = {
        "rmse_deg": float(np.sqrt(np.mean(np.square(selected)))),
        "max_abs_deg": float(np.max(np.abs(selected))),
    }
    return result


def summarize_trajectory(trajectory, enabled_arms):
    indices = selected_indices(enabled_arms)
    intervals = np.diff(trajectory.time_s)
    target_velocity = np.diff(trajectory.q_target, axis=0) / intervals[:, None]
    duration = float(trajectory.time_s[-1] - trajectory.time_s[0])
    return {
        "episode": str(trajectory.path),
        "samples": int(trajectory.q_target.shape[0]),
        "duration_s": duration,
        "configured_fps": trajectory.configured_fps,
        "measured_fps": float((len(intervals) / duration) if duration > 0 else 0.0),
        "interval_ms": {
            "median": float(np.median(intervals) * 1000.0),
            "p95": float(np.percentile(intervals, 95) * 1000.0),
            "max": float(np.max(intervals) * 1000.0),
        },
        "enabled_arms": enabled_arms,
        "recording_mode": trajectory.recording_mode,
        "recorded_enabled_arms": trajectory.recorded_enabled_arms,
        "recorded_feedforward_torque": trajectory.recorded_torque,
        "recorded_joint_velocity": trajectory.recorded_velocity,
        "max_recorded_target_velocity_rad_s": float(
            np.max(np.abs(target_velocity[:, indices]))
        ),
        "recorded_same_sample_encoder_error": _joint_metrics(
            trajectory.q_state - trajectory.q_target, indices
        ),
        "measurement_scope": (
            "Joint command tracking from R1-A7 encoders only; this is not an "
            "independent measurement of Cartesian absolute accuracy."
        ),
    }


def _merge_selected(target, measured, indices):
    merged = np.asarray(measured, dtype=np.float64).copy()
    merged[indices] = np.asarray(target, dtype=np.float64)[indices]
    return merged


def _wait_until(deadline):
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(remaining, 0.005))


def _send_interpolation(controller, target, indices, duration, rate_hz=100.0):
    start = controller.get_current_dual_arm_q().copy()
    target = _merge_selected(target, start, indices)
    steps = max(1, int(math.ceil(duration * rate_hz)))
    start_time = time.monotonic()
    for step in range(1, steps + 1):
        scale = step / steps
        command = start + (target - start) * scale
        controller.ctrl_dual_arm(command, np.zeros(14))
        _wait_until(start_time + step / rate_hz)
        controller.check_health()


def _hold_and_sample(controller, target, tau, indices, duration, rate_hz=100.0):
    samples = []
    steps = max(1, int(math.ceil(duration * rate_hz)))
    start_time = time.monotonic()
    for step in range(1, steps + 1):
        current = controller.get_current_dual_arm_q()
        command = _merge_selected(target, current, indices)
        command_tau = np.zeros(14)
        command_tau[indices] = tau[indices]
        controller.ctrl_dual_arm(command, command_tau)
        _wait_until(start_time + step / rate_hz)
        controller.check_health()
        samples.append(controller.get_current_dual_arm_q().copy())
    return np.vstack(samples)


def run_replay(trajectory, args):
    if trajectory.recording_mode != "r1_joint_precision":
        if trajectory.recording_mode != "r1_drag_teach":
            raise ValueError(
                "episode is not marked as r1_joint_precision or r1_drag_teach; "
                "record it with a dedicated R1-A7 recording tool before replay"
            )
        if trajectory.recorded_enabled_arms != args.enabled_arms:
            raise ValueError(
                "drag-teach replay arm selection must match the recording: "
                f"recorded={trajectory.recorded_enabled_arms!r}, "
                f"requested={args.enabled_arms!r}"
            )
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
        R1_A7_ARM_LOWER_LIMITS,
        R1_A7_ARM_UPPER_LIMITS,
        R1_A7_ArmController,
    )
    from teleop.utils.motion_switcher import MotionSwitcher
    from teleop.utils.network import resolve_dds_interface

    indices = selected_indices(args.enabled_arms)
    max_interval = float(np.max(np.diff(trajectory.time_s)))
    if max_interval >= 0.20:
        raise ValueError(
            f"recorded interval {max_interval:.3f}s is too close to the 0.25s "
            "command watchdog; record a continuous trajectory again"
        )
    selected_targets = trajectory.q_target[:, indices]
    if np.any(selected_targets < R1_A7_ARM_LOWER_LIMITS[indices]) or np.any(
        selected_targets > R1_A7_ARM_UPPER_LIMITS[indices]
    ):
        raise ValueError(
            "recorded target exceeds an official R1-A7 hard joint limit; replay refused"
        )
    interface, renamed = resolve_dds_interface(args.network_interface)
    if renamed:
        print(
            f"DDS interface {args.network_interface!r} unavailable; using {interface!r}."
        )
    ChannelFactoryInitialize(0, networkInterface=interface)
    switcher = MotionSwitcher()
    status, result = switcher.Enter_Debug_Mode()
    if status != 0:
        raise RuntimeError(
            f"failed to enter exclusive debug control: status={status}, result={result}"
        )

    controller = None
    trace_path = args.trace_path or args.output.with_suffix(".trace.jsonl")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    repeat_results = []
    final_means = []
    try:
        controller = R1_A7_ArmController(
            motion_mode=False,
            simulation_mode=False,
            arm_velocity_limit=30.0,
            joint_limit_margin=0.0,
            rate_limit_mode="official",
            clip_feedforward_torque=False,
        )
        controller.set_tracking_active(True)
        for second in range(args.countdown, 0, -1):
            print(f"Replay begins in {second}s; keep the emergency stop ready.")
            time.sleep(1.0)

        with trace_path.open("w", encoding="utf-8") as trace:
            for repeat in range(args.repeats):
                print(f"Repeat {repeat + 1}/{args.repeats}: approaching recorded start pose.")
                _send_interpolation(
                    controller,
                    trajectory.q_target[0],
                    indices,
                    args.approach_duration,
                )
                _hold_and_sample(
                    controller,
                    trajectory.q_target[0],
                    trajectory.tau_target[0],
                    indices,
                    args.start_hold_duration,
                )

                measured = []
                started = time.monotonic()
                for sample_index in range(trajectory.q_target.shape[0]):
                    current = controller.get_current_dual_arm_q()
                    q_command = _merge_selected(
                        trajectory.q_target[sample_index], current, indices
                    )
                    tau_command = np.zeros(14)
                    tau_command[indices] = trajectory.tau_target[sample_index, indices]
                    controller.ctrl_dual_arm(q_command, tau_command)
                    if sample_index + 1 < trajectory.q_target.shape[0]:
                        deadline = started + trajectory.time_s[sample_index + 1]
                    else:
                        deadline = started + trajectory.time_s[-1] + 1.0 / trajectory.configured_fps
                    _wait_until(deadline)
                    controller.check_health()
                    feedback = controller.get_current_dual_arm_q().copy()
                    measured.append(feedback)
                    trace.write(
                        json.dumps(
                            {
                                "repeat": repeat + 1,
                                "sample": sample_index,
                                "elapsed_s": time.monotonic() - started,
                                "target_q": q_command.tolist(),
                                "measured_q": feedback.tolist(),
                                "error_q": (feedback - q_command).tolist(),
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

                measured = np.vstack(measured)
                settled = _hold_and_sample(
                    controller,
                    trajectory.q_target[-1],
                    trajectory.tau_target[-1],
                    indices,
                    args.settle_duration,
                )
                final_mean = np.mean(settled, axis=0)
                final_means.append(final_mean)
                repeat_results.append(
                    {
                        "repeat": repeat + 1,
                        "trajectory_tracking": _joint_metrics(
                            measured - trajectory.q_target, indices
                        ),
                        "settled_final_error_deg": {
                            JOINT_NAMES[index]: float(
                                np.rad2deg(
                                    final_mean[index] - trajectory.q_target[-1, index]
                                )
                            )
                            for index in indices
                        },
                        "settled_joint_std_deg": {
                            JOINT_NAMES[index]: float(
                                np.rad2deg(np.std(settled[:, index]))
                            )
                            for index in indices
                        },
                    }
                )
                print(f"Repeat {repeat + 1}/{args.repeats}: complete.")

        final_means = np.vstack(final_means)
        if args.repeats >= 2:
            repeatability = {
                JOINT_NAMES[index]: float(np.rad2deg(np.std(final_means[:, index])))
                for index in indices
            }
        else:
            repeatability = None
        return {
            "executed_at": datetime.now().astimezone().isoformat(),
            "network_interface": interface,
            "control_profile": "official",
            "enabled_arms": args.enabled_arms,
            "repeats": args.repeats,
            "approach_duration_s": args.approach_duration,
            "settle_duration_s": args.settle_duration,
            "trace": str(trace_path),
            "repeat_results": repeat_results,
            "final_encoder_repeatability_std_deg": repeatability,
            "measurement_scope": (
                "Encoder-based joint tracking and repeatability. Use an external "
                "fixture or metrology sensor for Cartesian absolute accuracy."
            ),
        }
    finally:
        if controller is not None:
            controller.shutdown(go_home=False)


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an R1-A7 joint episode, or replay it on the real robot and "
            "report encoder-based tracking/repeatability metrics."
        )
    )
    parser.add_argument("episode", type=Path, help="Episode directory or data.json")
    parser.add_argument("--enabled-arms", choices=("left", "right", "both"), default="both")
    parser.add_argument("--execute", action="store_true", help="Actually command the real robot")
    parser.add_argument(
        "--confirm-real-robot",
        action="store_true",
        help="Confirm clear workspace, physical emergency stop, and sole command publisher",
    )
    parser.add_argument("--network-interface", default="enx00e04c187010")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--approach-duration", type=positive_float, default=3.0)
    parser.add_argument("--start-hold-duration", type=positive_float, default=0.5)
    parser.add_argument("--settle-duration", type=positive_float, default=1.0)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-path", type=Path)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.countdown < 0:
        parser.error("--countdown must be non-negative")
    if args.confirm_real_robot and not args.execute:
        parser.error("--confirm-real-robot is only valid with --execute")
    if args.execute and not args.confirm_real_robot:
        parser.error("--execute requires --confirm-real-robot")

    try:
        trajectory = load_trajectory(args.episode)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    summary = summarize_trajectory(trajectory, args.enabled_arms)
    if not args.execute:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("Inspection only: no command was sent to the robot.")
        return

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = trajectory.path.parent / f"joint_replay_report_{stamp}.json"
    else:
        args.output = args.output.expanduser().resolve()
    if args.trace_path is not None:
        args.trace_path = args.trace_path.expanduser().resolve()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Only the selected arm joints will be replayed; grippers are never commanded.")
    execution = run_replay(trajectory, args)
    report = {"source_summary": summary, "execution": execution}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
