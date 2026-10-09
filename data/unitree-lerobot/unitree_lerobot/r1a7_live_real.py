# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Authorized PI0.5 inference loop that commands a real R1-A7 through ZMQ."""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import signal
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import zmq

from teleimager.image_client import ImageClient
from teleop.utils.task_events import TaskEventServer
from unitree_lerobot.r1a7_async_policy import (
    ActionChunkResult,
    AsyncActionChunkProducer,
    select_executable_window,
)
from unitree_lerobot.r1a7_contract import CAMERA_KEYS, JOINT_NAMES
from unitree_lerobot.r1a7_hardware_gateway import REAL_ROBOT_CONFIRMATION
from unitree_lerobot.r1a7_inference_task_state import (
    DEFAULT_EXPECTED_BARCODE,
    DEFAULT_TASK_AFTER,
    DEFAULT_TASK_BEFORE,
    InferenceTaskState,
)
from unitree_lerobot.r1a7_live_action_conditioning import (
    ActionChunkBlender,
    build_start_pose_command,
    interpolate_start_pose,
    load_task_start_reference,
    prepare_safe_real_action,
    start_pose_errors,
)
from unitree_lerobot.r1a7_live_shadow import (
    PI05Session,
    StateBridgeClient,
    _read_cameras,
    assemble_live_observation,
)


class RealCommandClient:
    def __init__(self, endpoint: str, timeout_ms: int = 1000):
        self._context = zmq.Context.instance()
        self._endpoint = endpoint
        self._timeout_ms = timeout_ms
        self._socket = None
        self._connect()

    def _connect(self) -> None:
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.connect(self._endpoint)

    def _reset(self) -> None:
        self._socket.close(linger=0)
        self._connect()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            self._socket.send_json(payload)
            reply = self._socket.recv_json()
        except zmq.Again as exc:
            self._reset()
            raise RuntimeError("real command bridge timed out") from exc
        if reply.get("status") != "ok":
            raise RuntimeError(f"real command bridge rejected request: {reply}")
        return reply

    def command(self, action, *, policy_command: bool = True) -> dict[str, Any]:
        return self._request(
            {
                "request": "command",
                "action": action.tolist(),
                "policy_command": policy_command,
            }
        )

    def hold(self) -> dict[str, Any]:
        return self._request({"request": "hold"})

    def disarm(self) -> None:
        try:
            self._request({"request": "disarm"})
        except Exception:
            pass

    def close(self) -> None:
        self._socket.close(linger=0)

class ManualCycleConsole:
    """Read explicit operator commands without blocking the control loop."""

    def __init__(self):
        self._commands: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._worker = threading.Thread(target=self._read, daemon=True)
        self._worker.start()

    def _read(self) -> None:
        while True:
            line = sys.stdin.readline()
            if not line:
                return
            command = line.strip().upper()
            if command in {"P", "N", "R", "D"}:
                self._commands.put(command)
            elif command:
                print(
                    "Operator commands: P + Enter to pause during scan or place; N + Enter "
                    "to return to the start pose and begin a fresh scan; R + Enter "
                    "to clear old action chunks and restart; D + Enter to confirm "
                    "the scanned product is placed and start the completion countdown.",
                    flush=True,
                )

    def poll(self) -> str | None:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None


def return_to_scan_start_pose(
    command_client, state_client, reference, args, log, stopped,
):
    """Return from an operator-cleared place pose before re-arming the scanner."""
    held = command_client.hold()
    starting, clipping = prepare_safe_real_action(
        held["held_state"], max_arm_overshoot=args.max_arm_limit_overshoot
    )
    target = build_start_pose_command(starting, reference)
    log.write(json.dumps({
        "event": "manual_cycle_return_started",
        "start_action": starting.tolist(),
        "target": target.tolist(),
        "clipping": clipping,
        "wall_time_ns": time.time_ns(),
    }, ensure_ascii=False) + "\n")
    # The first non-policy command switches the bridge to its initialization
    # watchdog before any potentially slow state read.
    command_client.command(starting, policy_command=False)
    steps = max(1, int(round(args.initialization_duration / args.interval)))
    for index in range(1, steps + 1):
        if stopped.is_set():
            return None
        command_client.command(
            interpolate_start_pose(starting, target, index / steps),
            policy_command=False,
        )
        stopped.wait(args.interval)

    deadline = time.monotonic() + args.initialization_timeout
    while not stopped.is_set():
        measured = state_client.read()["state"]
        arm_error, gripper_error = start_pose_errors(measured, target)
        if (
            arm_error <= args.initialization_arm_tolerance
            and gripper_error <= args.initialization_gripper_tolerance
        ):
            log.write(json.dumps({
                "event": "manual_cycle_return_complete",
                "measured_state": list(measured),
                "arm_error": arm_error,
                "gripper_error": gripper_error,
                "wall_time_ns": time.time_ns(),
            }, ensure_ascii=False) + "\n")
            return np.asarray(measured, dtype=np.float64)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "manual cycle return did not converge: "
                f"arm_error={arm_error:.4f}, gripper_error={gripper_error:.4f}"
            )
        command_client.command(target, policy_command=False)
        stopped.wait(args.interval)
    return None

class DiagnosticFrames:
    """Save exact policy RGB inputs off the real-time command thread."""

    def __init__(self, root: Path, run_id: str):
        self.directory = root.expanduser().resolve() / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self._queue: queue.Queue[tuple[str, dict[str, Any], dict[str, np.ndarray]] | None] = queue.Queue(maxsize=3)
        self._worker = threading.Thread(target=self._write, daemon=True)
        self._worker.start()

    @staticmethod
    def snapshot(state, images, input_meta, task, step):
        return (
            {
                "step": step,
                "task": task,
                "state": np.asarray(state).tolist(),
                "input": input_meta,
                "wall_time_ns": time.time_ns(),
            },
            {key: images[key].copy() for key in CAMERA_KEYS},
        )

    def save(self, name: str, capture) -> None:
        metadata, images = capture
        self._queue.put_nowait((name, metadata, images))

    def _write(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                name, metadata, images = item
                for key in CAMERA_KEYS:
                    rgb = images[key].transpose(1, 2, 0)
                    path = self.directory / f"{name}_{key.rsplit('.', 1)[-1]}.png"
                    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                        raise OSError(f"could not save diagnostic image: {path}")
                (self.directory / f"{name}.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as exc:
                print(f"Diagnostic frame save failed: {exc}", flush=True)
            finally:
                self._queue.task_done()

    def close(self) -> None:
        self._queue.put(None)
        self._queue.join()
        self._worker.join()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--image-host", default="127.0.0.1")
    parser.add_argument("--bridge-endpoint", default="tcp://127.0.0.1:55582")
    parser.add_argument("--task-event-bind", default="tcp://0.0.0.0:55580")
    parser.add_argument("--expected-barcode", default=DEFAULT_EXPECTED_BARCODE)
    parser.add_argument("--task-before", default=DEFAULT_TASK_BEFORE)
    parser.add_argument("--task-after", default=DEFAULT_TASK_AFTER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-commands", type=int, default=0)
    parser.add_argument("--chunk-blend-steps", type=int, default=3)
    parser.add_argument("--async-prefetch-actions", type=int, default=5)
    parser.add_argument("--initialization-duration", type=float, default=3.0)
    parser.add_argument("--initialization-timeout", type=float, default=12.0)
    parser.add_argument("--initialization-arm-tolerance", type=float, default=0.08)
    parser.add_argument("--initialization-gripper-tolerance", type=float, default=0.20)
    parser.add_argument("--max-arm-limit-overshoot", type=float, default=0.05)
    parser.add_argument("--interval", type=float, default=1 / 15)
    parser.add_argument("--max-state-age-ms", type=float, default=250.0)
    parser.add_argument("--max-camera-age-ms", type=float, default=250.0)
    parser.add_argument("--max-camera-skew-ms", type=float, default=150.0)
    parser.add_argument("--readiness-timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-frames-dir", type=Path)
    parser.add_argument("--manual-cycles", action="store_true")
    parser.add_argument("--completion-wait-seconds", type=float, default=0.0)
    parser.add_argument("--confirm-real-robot", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.confirm_real_robot != REAL_ROBOT_CONFIRMATION:
        raise PermissionError("incorrect real-robot confirmation")
    if os.environ.get("R1A7_REAL_INFERENCE") != "YES":
        raise PermissionError("set R1A7_REAL_INFERENCE=YES for real inference")
    if args.max_commands < 0:
        raise ValueError("max-commands must be non-negative")
    if args.manual_cycles and (args.max_commands != 0 or not sys.stdin.isatty()):
        raise ValueError("manual-cycles requires --max-commands 0 and an interactive terminal")
    if args.completion_wait_seconds < 0:
        raise ValueError("completion-wait-seconds must be non-negative")
    if args.completion_wait_seconds and not args.manual_cycles:
        raise ValueError("completion-wait-seconds requires manual-cycles")
    if args.chunk_blend_steps < 0:
        raise ValueError("chunk-blend-steps must be non-negative")
    if args.async_prefetch_actions <= 0:
        raise ValueError("async-prefetch-actions must be positive")
    for name in (
        "initialization_duration",
        "initialization_timeout",
        "initialization_arm_tolerance",
        "initialization_gripper_tolerance",
        "max_arm_limit_overshoot",
    ):
        if not getattr(args, name) > 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")

    dataset_root = args.dataset_root.expanduser().resolve()
    start_reference, start_reference_count = load_task_start_reference(
        dataset_root, args.task_before
    )

    state_client = None
    command_client = None
    image_client = None
    event_server = None
    chunk_producer = None
    diagnostic = None
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda _s, _f: stopped.set())
    signal.signal(signal.SIGTERM, lambda _s, _f: stopped.set())
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        state_client = StateBridgeClient(args.bridge_endpoint, 1000)
        command_client = RealCommandClient(args.bridge_endpoint)
        image_client = ImageClient(host=args.image_host, request_bgr=True)
        task_state = InferenceTaskState(
            expected_barcode=args.expected_barcode,
            task_before=args.task_before,
            task_after=args.task_after,
        )
        if args.diagnostic_frames_dir is not None:
            diagnostic = DiagnosticFrames(args.diagnostic_frames_dir, task_state.run_id)
        event_server = TaskEventServer(
            args.task_event_bind, task_state.handle_item_accepted
        )
        deadline = time.monotonic() + args.readiness_timeout
        while True:
            try:
                assemble_live_observation(
                    state_client.read(),
                    _read_cameras(image_client),
                    now_monotonic_ns=time.monotonic_ns(),
                    max_state_age_ms=args.max_state_age_ms,
                    max_camera_age_ms=args.max_camera_age_ms,
                    max_camera_skew_ms=args.max_camera_skew_ms,
                )
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)

        print("Real inputs ready; loading PI0.5 checkpoint while robot holds...", flush=True)
        session = PI05Session(
            dataset_root=dataset_root,
            policy_path=args.policy_path.expanduser().resolve(),
            tokenizer_path=args.tokenizer_path.expanduser().resolve(),
            device_name=args.device,
            seed=args.seed,
        )
        if args.async_prefetch_actions >= session.n_action_steps:
            raise ValueError(
                "async-prefetch-actions must be smaller than policy n_action_steps "
                f"({session.n_action_steps})"
            )
        if args.async_prefetch_actions + session.n_action_steps > session.chunk_size:
            raise ValueError(
                "async-prefetch-actions plus policy n_action_steps exceeds the "
                f"candidate chunk size ({session.chunk_size})"
            )
        chunk_producer = AsyncActionChunkProducer(session)
        chunk_blender = ActionChunkBlender(args.chunk_blend_steps)
        commands_sent = 0
        step = 0
        with output.open("a", encoding="utf-8", buffering=1) as log:
            initial_state = state_client.read()["state"]
            initial_action, initial_clipping = prepare_safe_real_action(
                initial_state,
                max_arm_overshoot=args.max_arm_limit_overshoot,
            )
            initialization_target = build_start_pose_command(
                initial_action, start_reference
            )
            init_record = {
                "event": "initialization_started",
                "reference_episode_count": start_reference_count,
                "initial_state": list(initial_state),
                "initial_action": initial_action.tolist(),
                "initial_clipping": initial_clipping,
                "reference_state": start_reference.tolist(),
                "command_target": initialization_target.tolist(),
                "duration": args.initialization_duration,
                "wall_time_ns": time.time_ns(),
            }
            print(
                f"Aligning to median scan start pose from {start_reference_count} episodes...",
                flush=True,
            )
            log.write(json.dumps(init_record, ensure_ascii=False) + "\n")
            initialization_steps = max(
                1, int(round(args.initialization_duration / args.interval))
            )
            for index in range(1, initialization_steps + 1):
                if stopped.is_set():
                    break
                command_client.command(
                    interpolate_start_pose(
                        initial_action,
                        initialization_target,
                        index / initialization_steps,
                    ),
                    policy_command=False,
                )
                stopped.wait(args.interval)
            if stopped.is_set():
                return 130

            convergence_deadline = time.monotonic() + args.initialization_timeout
            next_progress_report = time.monotonic()
            while True:
                measured = state_client.read()["state"]
                arm_error, gripper_error = start_pose_errors(
                    measured, initialization_target
                )
                joint_errors = [
                    abs(float(actual) - float(expected))
                    for actual, expected in zip(
                        measured, initialization_target, strict=True
                    )
                ]
                worst_arm_index = max(range(14), key=joint_errors.__getitem__)
                if (
                    arm_error <= args.initialization_arm_tolerance
                    and gripper_error <= args.initialization_gripper_tolerance
                ):
                    break
                if time.monotonic() >= convergence_deadline:
                    failure = {
                        "event": "initialization_failed",
                        "measured_state": list(measured),
                        "reference_state": start_reference.tolist(),
                        "command_target": initialization_target.tolist(),
                        "joint_errors": dict(
                            zip(JOINT_NAMES, joint_errors, strict=True)
                        ),
                        "worst_arm_joint": JOINT_NAMES[worst_arm_index],
                        "arm_error": arm_error,
                        "gripper_error": gripper_error,
                        "hardware_commands_sent": True,
                        "wall_time_ns": time.time_ns(),
                    }
                    log.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    raise RuntimeError(
                        "start-pose initialization did not converge: "
                        f"{JOINT_NAMES[worst_arm_index]} "
                        f"arm_error={arm_error:.4f} rad, "
                        f"gripper_error={gripper_error:.4f}"
                    )
                if time.monotonic() >= next_progress_report:
                    print(
                        "Initialization settling: "
                        f"{JOINT_NAMES[worst_arm_index]} error={arm_error:.4f} rad, "
                        f"gripper_error={gripper_error:.4f}",
                        flush=True,
                    )
                    next_progress_report = time.monotonic() + 1.0
                command_client.command(
                    initialization_target,
                    policy_command=False,
                )
                stopped.wait(args.interval)

            chunk_blender.prime(measured)
            event_server.start()
            last_generation = task_state.snapshot().generation
            aligned = {
                "event": "initialization_complete",
                "measured_state": list(measured),
                "arm_error": arm_error,
                "gripper_error": gripper_error,
                "hardware_commands_sent": True,
                "wall_time_ns": time.time_ns(),
            }
            log.write(json.dumps(aligned, ensure_ascii=False) + "\n")
            ready = {
                "event": "real_inference_ready",
                "run_id": task_state.run_id,
                "profile": "official",
                "policy_dimension": len(JOINT_NAMES),
                "waist_mode": "hardware_hold_only",
                "chunk_blend_steps": args.chunk_blend_steps,
                "policy_candidate_chunk_size": session.chunk_size,
                "policy_executed_action_steps": session.n_action_steps,
                "async_prefetch_actions": args.async_prefetch_actions,
                "start_reference_episode_count": start_reference_count,
                "hardware_commands_sent": True,
                "wall_time_ns": time.time_ns(),
            }
            print(json.dumps(ready, ensure_ascii=False), flush=True)
            log.write(json.dumps(ready, ensure_ascii=False) + "\n")
            active_actions = deque()
            active_chunk_size = 0
            active_chunk_index = 0
            active_chunk_inference_ms = 0.0
            next_chunk: ActionChunkResult | None = None
            next_chunk_input = None
            pending_chunk_input = None
            active_chunk_input = None
            first_input_saved = False
            first_close_saved = False
            requested_generation: int | None = None
            operator = ManualCycleConsole() if args.manual_cycles else None
            paused_for_next_cycle = False
            paused_at = None
            completion_deadline = None
            next_completion_report_at = None
            cycle_number = 1
            diagnostic_prefix = ""
            if operator is not None:
                print(
                    "Manual cycles enabled: press R + Enter only with empty grippers and a "
                    "clear arm path; it returns to scan start and begins a fresh scan. "
                    "Press P + Enter to hold during scan or place, then press N or R "
                    "to return when the path is clear. "
                    + (
                        f"After placing the product, press D + Enter to hold for "
                        f"{args.completion_wait_seconds:g}s before automatic return. "
                        if args.completion_wait_seconds else ""
                    )
                    + "Ctrl-C stops.",
                    flush=True,
                )
            while not stopped.is_set() and (
                args.max_commands == 0 or commands_sent < args.max_commands
            ):
                loop_start = time.monotonic()
                close_captures = None
                snapshot = task_state.snapshot(last_consumed_generation=last_generation)
                operator_command = None if operator is None else operator.poll()
                automatic_completion_restart = False
                if completion_deadline is not None:
                    if operator_command == "P":
                        completion_deadline = None
                        next_completion_report_at = None
                        paused_for_next_cycle = True
                        paused_at = time.monotonic()
                        operator_command = None
                        log.write(json.dumps({
                            "event": "completion_auto_return_cancelled",
                            "reason": "operator_paused",
                            "run_id": snapshot.run_id,
                            "command_count": commands_sent,
                            "wall_time_ns": time.time_ns(),
                        }, ensure_ascii=False) + "\n")
                        print(
                            "PAUSED: automatic return cancelled. Robot holding; "
                            "press R when empty grippers and the return path are clear.",
                            flush=True,
                        )
                    elif operator_command == "R":
                        completion_deadline = None
                        next_completion_report_at = None
                        log.write(json.dumps({
                            "event": "completion_wait_skipped",
                            "reason": "operator_restart",
                            "run_id": snapshot.run_id,
                            "command_count": commands_sent,
                            "wall_time_ns": time.time_ns(),
                        }, ensure_ascii=False) + "\n")
                    elif time.monotonic() >= completion_deadline:
                        completion_deadline = None
                        next_completion_report_at = None
                        automatic_completion_restart = True
                        operator_command = "R"
                        log.write(json.dumps({
                            "event": "completion_wait_elapsed",
                            "run_id": snapshot.run_id,
                            "command_count": commands_sent,
                            "wall_time_ns": time.time_ns(),
                        }, ensure_ascii=False) + "\n")
                        print(
                            "Completion wait elapsed. Automatically returning to "
                            "scan start; keep the arm path clear.",
                            flush=True,
                        )
                    else:
                        now = time.monotonic()
                        if now >= next_completion_report_at:
                            remaining = max(0.0, completion_deadline - now)
                            log.write(json.dumps({
                                "event": "completion_wait_progress",
                                "run_id": snapshot.run_id,
                                "remaining_seconds": round(remaining, 1),
                                "command_count": commands_sent,
                                "wall_time_ns": time.time_ns(),
                            }, ensure_ascii=False) + "\n")
                            print(
                                f"COMPLETION HOLD: {remaining:.0f}s until automatic "
                                "return; press P to cancel.",
                                flush=True,
                            )
                            next_completion_report_at = now + 5.0
                        command_client.hold()
                        stopped.wait(max(0.0, args.interval - (time.monotonic() - loop_start)))
                        continue
                if operator_command == "R":
                    if not automatic_completion_restart:
                        print(
                            "R reset requested: keep people clear and ensure both grippers are "
                            "empty and the arm path is clear before return begins.",
                            flush=True,
                        )
                    hold_reply = command_client.hold()
                    transition = task_state.pause_for_restart()
                    last_generation = transition["generation"]
                    active_actions.clear()
                    next_chunk = None
                    next_chunk_input = None
                    pending_chunk_input = None
                    active_chunk_input = None
                    requested_generation = None
                    log.write(json.dumps({
                        "event": (
                            "automatic_policy_restart_started"
                            if automatic_completion_restart else "manual_policy_restart_started"
                        ),
                        **transition,
                        "command_count": commands_sent,
                        "hold_reply": hold_reply,
                        "discarded_active_actions": True,
                        "discarded_prefetched_actions": True,
                        "wall_time_ns": time.time_ns(),
                    }, ensure_ascii=False) + "\n")
                    print("Returning to scan start pose; keep clear of the robot.", flush=True)
                    aligned_state = return_to_scan_start_pose(
                        command_client, state_client, start_reference,
                        args, log, stopped,
                    )
                    if aligned_state is None:
                        break
                    transition = task_state.start_restarted_cycle()
                    last_generation = transition["generation"]
                    chunk_blender.prime(aligned_state)
                    paused_for_next_cycle = False
                    paused_at = None
                    cycle_number += 1
                    diagnostic_prefix = f"cycle_{cycle_number:04d}_"
                    first_input_saved = False
                    first_close_saved = False
                    log.write(json.dumps({
                        "event": (
                            "automatic_policy_restart_completed"
                            if automatic_completion_restart else "manual_policy_restart_completed"
                        ),
                        "cycle_number": cycle_number,
                        **transition,
                        "command_count": commands_sent,
                        "hardware_commands_sent": True,
                        "wall_time_ns": time.time_ns(),
                    }, ensure_ascii=False) + "\n")
                    print(
                        f"INFERENCE RESTARTED: cycle={cycle_number} run_id={transition['run_id']} phase=scan; "
                        "old action chunks discarded.",
                        flush=True,
                    )
                    continue
                elif operator_command == "P":
                    if paused_for_next_cycle:
                        print("P ignored: already paused.", flush=True)
                    else:
                        hold_reply = command_client.hold()
                        paused = task_state.pause_for_next_cycle()
                        paused_for_next_cycle = True
                        paused_at = time.monotonic()
                        active_actions.clear()
                        next_chunk = None
                        next_chunk_input = None
                        pending_chunk_input = None
                        active_chunk_input = None
                        requested_generation = None
                        log.write(json.dumps({
                            "event": "manual_cycle_paused",
                            "run_id": paused["run_id"],
                            "phase": paused["phase"],
                            "command_count": commands_sent,
                            "hold_reply": hold_reply,
                            "wall_time_ns": time.time_ns(),
                        }, ensure_ascii=False) + "\n")
                        print(
                            "PAUSED: robot holding. Move product to blue tray; ensure both "
                            "grippers are empty and all people are clear. Then type R or N.",
                            flush=True,
                        )
                elif operator_command == "N":
                    if not paused_for_next_cycle:
                        print("N ignored: first type P to pause.", flush=True)
                    elif time.monotonic() - paused_at < 1.0:
                        print("N ignored: wait for PAUSED, clear the workspace, then press n again.", flush=True)
                    else:
                        print("Returning to scan start pose; keep clear of the robot.", flush=True)
                        aligned_state = return_to_scan_start_pose(
                            command_client, state_client, start_reference,
                            args, log, stopped,
                        )
                        if aligned_state is None:
                            break
                        transition = task_state.start_next_cycle()
                        last_generation = transition["generation"]
                        chunk_blender.prime(aligned_state)
                        active_actions.clear()
                        next_chunk = None
                        next_chunk_input = None
                        pending_chunk_input = None
                        active_chunk_input = None
                        requested_generation = None
                        paused_for_next_cycle = False
                        paused_at = None
                        cycle_number += 1
                        diagnostic_prefix = f"cycle_{cycle_number:04d}_"
                        first_input_saved = False
                        first_close_saved = False
                        log.write(json.dumps({
                            "event": "manual_cycle_ready",
                            "cycle_number": cycle_number,
                            **transition,
                            "command_count": commands_sent,
                            "hardware_commands_sent": True,
                            "wall_time_ns": time.time_ns(),
                        }, ensure_ascii=False) + "\n")
                        print(
                            f"NEXT CYCLE READY: cycle={cycle_number} "
                            f"run_id={transition['run_id']} phase=scan",
                            flush=True,
                        )
                        continue
                elif operator_command == "D":
                    if not args.completion_wait_seconds:
                        print("D ignored: completion countdown is disabled.", flush=True)
                    elif paused_for_next_cycle or snapshot.phase != "place":
                        print(
                            "D ignored: confirm completion only after an accepted scan "
                            "and while the place phase is active.",
                            flush=True,
                        )
                    else:
                        hold_reply = command_client.hold()
                        try:
                            transition = task_state.confirm_completion()
                        except RuntimeError as exc:
                            print(f"D ignored: {exc}", flush=True)
                        else:
                            last_generation = transition["generation"]
                            active_actions.clear()
                            next_chunk = None
                            next_chunk_input = None
                            pending_chunk_input = None
                            active_chunk_input = None
                            requested_generation = None
                            completion_deadline = time.monotonic() + args.completion_wait_seconds
                            next_completion_report_at = time.monotonic()
                            log.write(json.dumps({
                                "event": "task_completion_confirmed",
                                **transition,
                                "cycle_number": cycle_number,
                                "command_count": commands_sent,
                                "wait_seconds": args.completion_wait_seconds,
                                "hold_reply": hold_reply,
                                "discarded_active_actions": True,
                                "discarded_prefetched_actions": True,
                                "wall_time_ns": time.time_ns(),
                            }, ensure_ascii=False) + "\n")
                            print(
                                f"TASK COMPLETE CONFIRMED: holding for "
                                f"{args.completion_wait_seconds:g}s, then returning "
                                "to scan start automatically. Press P to cancel.",
                                flush=True,
                            )
                            stopped.wait(max(0.0, args.interval - (time.monotonic() - loop_start)))
                            continue
                if paused_for_next_cycle:
                    command_client.hold()
                    stopped.wait(max(0.0, args.interval - (time.monotonic() - loop_start)))
                    continue
                if snapshot.policy_reset_required:
                    active_actions.clear()
                    next_chunk = None
                    next_chunk_input = None
                    pending_chunk_input = None
                    active_chunk_input = None
                    requested_generation = None
                    last_generation = snapshot.generation
                try:
                    state, images, input_meta = assemble_live_observation(
                        state_client.read(),
                        _read_cameras(image_client),
                        now_monotonic_ns=time.monotonic_ns(),
                        max_state_age_ms=args.max_state_age_ms,
                        max_camera_age_ms=args.max_camera_age_ms,
                        max_camera_skew_ms=args.max_camera_skew_ms,
                    )
                    policy_state = state

                    completed = chunk_producer.poll()
                    if completed is not None:
                        if completed.generation == requested_generation:
                            requested_generation = None
                        if completed.generation == snapshot.generation:
                            if completed.error is not None:
                                raise RuntimeError(
                                    "asynchronous policy inference failed"
                                ) from completed.error
                            next_chunk = completed
                            next_chunk_input = pending_chunk_input
                            pending_chunk_input = None

                    if not active_actions:
                        if next_chunk is None:
                            if requested_generation != snapshot.generation:
                                chunk_producer.submit(
                                    generation=snapshot.generation,
                                    task=snapshot.task,
                                    lead_actions=0,
                                    state=policy_state,
                                    images=images,
                                )
                                if diagnostic is not None:
                                    pending_chunk_input = DiagnosticFrames.snapshot(
                                        policy_state, images, input_meta, snapshot.task, step
                                    )
                                    if not first_input_saved:
                                        diagnostic.save(f"{diagnostic_prefix}first_policy_input", pending_chunk_input)
                                        first_input_saved = True
                                requested_generation = snapshot.generation
                            hold_reply = command_client.hold()
                            wait_record = {
                                "event": "real_async_prefetch_wait",
                                "step": step,
                                "task_state": asdict(snapshot),
                                "input": input_meta,
                                "hold_reply": hold_reply,
                                "hardware_commands_sent": hold_reply[
                                    "hardware_commands_sent"
                                ],
                                "wall_time_ns": time.time_ns(),
                            }
                            log.write(
                                json.dumps(wait_record, ensure_ascii=False) + "\n"
                            )
                            print(
                                f"step={step} HOLD: waiting for "
                                "asynchronous action chunk",
                                flush=True,
                            )
                            step += 1
                            stopped.wait(
                                max(
                                    0.0,
                                    args.interval - (time.monotonic() - loop_start),
                                )
                            )
                            continue
                        if next_chunk.actions is None:
                            raise RuntimeError(
                                "asynchronous policy result contains no actions"
                            )
                        chunk_offset = next_chunk.lead_actions
                        active_actions = deque(
                            select_executable_window(
                                next_chunk.actions,
                                lead_actions=chunk_offset,
                                n_action_steps=session.n_action_steps,
                            )
                        )
                        active_chunk_size = len(active_actions)
                        active_chunk_index = 0
                        active_chunk_inference_ms = next_chunk.inference_ms
                        active_chunk_offset = chunk_offset
                        active_chunk_input = next_chunk_input
                        next_chunk = None
                        next_chunk_input = None

                    # Submit before popping: this observation is followed by exactly
                    # len(active_actions) queued commands before the new window starts.
                    if (
                        len(active_actions) <= args.async_prefetch_actions
                        and next_chunk is None
                        and requested_generation != snapshot.generation
                    ):
                        chunk_producer.submit(
                            generation=snapshot.generation,
                            task=snapshot.task,
                            lead_actions=len(active_actions),
                            state=policy_state,
                            images=images,
                        )
                        if diagnostic is not None:
                            pending_chunk_input = DiagnosticFrames.snapshot(
                                policy_state, images, input_meta, snapshot.task, step
                            )
                        requested_generation = snapshot.generation

                    raw_action = active_actions.popleft()
                    new_chunk = active_chunk_index == 0
                    unblended_action, clipping = prepare_safe_real_action(
                        raw_action,
                        max_arm_overshoot=args.max_arm_limit_overshoot,
                    )
                    inference_ms = active_chunk_inference_ms if new_chunk else 0.0
                    if task_state.snapshot().generation != snapshot.generation:
                        raise RuntimeError(
                            "task changed during policy inference; discarded old-task action"
                        )
                    action, chunk_blending = chunk_blender.apply(
                        unblended_action,
                        new_chunk=new_chunk,
                    )
                    if (
                        diagnostic is not None
                        and not first_close_saved
                        and min(action[14], action[15]) < 4.0
                    ):
                        close_captures = (
                            active_chunk_input,
                            DiagnosticFrames.snapshot(
                                state, images, input_meta, snapshot.task, step
                            ),
                        )
                    active_chunk_index += 1
                except Exception as exc:
                    hold_reply = command_client.hold()
                    record = {
                        "event": "real_prediction_rejected",
                        "step": step,
                        "task_state": asdict(snapshot),
                        "reason": str(exc),
                        "hold_reply": hold_reply,
                        "hardware_commands_sent": hold_reply["hardware_commands_sent"],
                        "wall_time_ns": time.time_ns(),
                    }
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"step={step} HOLD: {exc}", flush=True)
                    step += 1
                    stopped.wait(max(0.0, args.interval - (time.monotonic() - loop_start)))
                    continue

                if stopped.is_set():
                    break
                try:
                    reply = command_client.command(action)
                except RuntimeError:
                    if stopped.is_set():
                        break
                    raise
                commands_sent += 1
                if close_captures is not None:
                    source_capture, live_capture = close_captures
                    for capture in (source_capture, live_capture):
                        if capture is not None:
                            capture[0]["command_count"] = commands_sent
                            capture[0]["requested_action"] = action.tolist()
                    if source_capture is not None:
                        diagnostic.save(f"{diagnostic_prefix}first_close_policy_input", source_capture)
                    diagnostic.save(f"{diagnostic_prefix}first_close_live_observation", live_capture)
                    first_close_saved = True
                executed_action = reply.get("executed_action")
                if not isinstance(executed_action, list) or len(executed_action) != len(
                    JOINT_NAMES
                ):
                    raise RuntimeError(
                        "real command bridge did not return its 16D executed action"
                    )
                record = {
                    "event": "real_command",
                    "step": step,
                    "command_count": commands_sent,
                    "task_state": asdict(snapshot),
                    "input": input_meta,
                    "state": state.tolist(),
                    "policy_state": policy_state.tolist(),
                    "raw_action": raw_action.tolist(),
                    "unblended_policy_action": unblended_action.tolist(),
                    "policy_action": action.tolist(),
                    "action": executed_action,
                    "clipping": clipping,
                    "chunk_blending": chunk_blending,
                    "inference_ms": inference_ms,
                    "async_chunk": {
                        "index": active_chunk_index - 1,
                        "source_index": active_chunk_offset
                        + active_chunk_index
                        - 1,
                        "size": active_chunk_size,
                        "remaining": len(active_actions),
                        "prefetch_threshold": args.async_prefetch_actions,
                        "prefetch_pending": requested_generation
                        == snapshot.generation,
                    },
                    "bridge_reply": reply,
                    "hardware_commands_sent": True,
                    "wall_time_ns": time.time_ns(),
                }
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"command={commands_sent} phase={snapshot.phase} "
                    f"clipped={len(clipping)} "
                    f"chunk_blend={chunk_blending['alpha']:.2f} "
                    f"inference_ms={inference_ms:.1f}",
                    flush=True,
                )
                step += 1
                stopped.wait(max(0.0, args.interval - (time.monotonic() - loop_start)))
    finally:
        if command_client is not None:
            command_client.disarm()
        if diagnostic is not None:
            diagnostic.close()
        if chunk_producer is not None:
            chunk_producer.close()
        if event_server is not None:
            event_server.stop()
        if image_client is not None:
            image_client.close()
        if state_client is not None:
            state_client.close()
        if command_client is not None:
            command_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
