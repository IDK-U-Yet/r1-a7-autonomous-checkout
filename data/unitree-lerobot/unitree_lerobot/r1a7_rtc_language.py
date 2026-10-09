# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Independent 15 Hz guided RTC experiment; explicit real-hardware authorization."""

import argparse
from dataclasses import asdict
import json
import os
import queue
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

from teleimager.image_client import ImageClient
from teleop.utils.task_events import TaskEventServer
from unitree_lerobot.r1a7_hardware_gateway import REAL_ROBOT_CONFIRMATION
from unitree_lerobot.r1a7_inference_task_state import InferenceTaskState
from unitree_lerobot.r1a7_live_action_conditioning import (
    load_task_start_reference, prepare_safe_real_action,
)
from unitree_lerobot.r1a7_live_real import (
    RealCommandClient, return_to_scan_start_pose,
)
from unitree_lerobot.r1a7_live_shadow import (
    StateBridgeClient, _read_cameras, assemble_live_observation,
)
from unitree_lerobot.r1a7_rtc_async import RTCController
from unitree_lerobot.r1a7_rtc_session import RTCSession

DEFAULT_TASK_BEFORE = "Bring the product from its current location to the scanner in the scanning area, present its barcode, and wait until the scan is accepted."
DEFAULT_TASK_AFTER = "Place the scanned product in the black tray on the right."


def read_training_tasks(dataset_root):
    import pyarrow.parquet as pq
    rows = pq.read_table(dataset_root / "meta/tasks.parquet").to_pylist()
    return {r.get("task", r.get("__index_level_0__")) for r in rows
            if isinstance(r.get("task", r.get("__index_level_0__")), str)}


def resolve_task_contract(args, training_tasks):
    if not args.task_before.strip() or not args.task_after.strip():
        raise ValueError("Task prompts must not be empty")
    matched = args.task_before in training_tasks and args.task_after in training_tasks
    if not matched and not args.allow_task_override:
        raise ValueError("Task text differs from the training dataset; use --allow-task-override for an explicit prompt experiment")
    # Prompt experiments never select an unknown return trajectory.
    reference_task = args.reference_task_before or (
        DEFAULT_TASK_BEFORE if args.allow_task_override else args.task_before)
    if reference_task not in training_tasks:
        raise ValueError("Reference task must exist in the training dataset")
    return {"allow_task_override": args.allow_task_override,
            "prompts_match_training": matched,
            "task_before": args.task_before, "task_after": args.task_after,
            "reference_task_before": reference_task,
            "training_tasks": sorted(training_tasks)}


class DispatchGate:
    """Serialize accepted task events with dispatch; late actions cannot race ACK."""
    def __init__(self):
        self.lock = threading.RLock()
        self.revision = 0
        self.paused = False
        self.scan_accept_after_ns = 0

    def invalidate(self, *, paused=None):
        with self.lock:
            self.revision += 1
            if paused is not None:
                self.paused = paused
                if not paused:
                    self.scan_accept_after_ns = time.time_ns()

    def dispatch(self, revision, callback):
        with self.lock:
            if self.paused or revision != self.revision:
                return None
            return callback()


class RTCConsole:
    def __init__(self):
        self.commands = queue.SimpleQueue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in sys.stdin:
            key = line.strip().upper()
            if key in {"P", "N", "R", "Q"}:
                self.commands.put(key)

    def poll(self):
        commands = []
        while True:
            try:
                commands.append(self.commands.get_nowait())
            except queue.Empty:
                return commands


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "policy-path", "tokenizer-path", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--bridge-endpoint", default="tcp://127.0.0.1:55582")
    parser.add_argument("--image-host", default="127.0.0.1")
    parser.add_argument("--task-event-bind", default="tcp://127.0.0.1:55580")
    parser.add_argument("--task-before", default=DEFAULT_TASK_BEFORE)
    parser.add_argument("--task-after", default=DEFAULT_TASK_AFTER)
    parser.add_argument("--allow-task-override", action="store_true")
    parser.add_argument("--reference-task-before", default=None)
    parser.add_argument("--expected-barcode", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--execution-steps", type=int, default=10)
    parser.add_argument("--prefetch-steps", type=int, default=7)
    parser.add_argument("--initial-delay-steps", type=int, default=7)
    parser.add_argument("--max-guidance-weight", type=float, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--rtc-mask-mode", choices=("configured", "overlap"), default="configured")
    parser.add_argument("--max-commands", type=int, default=30)
    parser.add_argument("--confirm-real-robot", required=True)
    parser.set_defaults(
        interval=1 / 15, initialization_duration=3.0, initialization_timeout=12.0,
        initialization_arm_tolerance=.08, initialization_gripper_tolerance=.2,
        max_arm_limit_overshoot=.05,
    )
    return parser


def read_observation(state_client, images, record):
    """Read state before frames so a slow RPC cannot age an earlier snapshot."""
    started = time.monotonic()
    reply = state_client.read()
    state_ms = (time.monotonic() - started) * 1000
    frame_started = time.monotonic()
    frames = _read_cameras(images)
    timing = {"state_read_ms": state_ms,
              "camera_read_ms": (time.monotonic() - frame_started) * 1000,
              "observation_read_ms": (time.monotonic() - started) * 1000}
    try:
        state, rgb, meta = assemble_live_observation(
            reply, frames, now_monotonic_ns=time.monotonic_ns(),
            max_state_age_ms=250, max_camera_age_ms=250, max_camera_skew_ms=150,
        )
    except RuntimeError as exc:
        record({"event": "rtc_observation_error", "reason": str(exc), **timing})
        raise
    meta["timing"] = timing
    return state, rgb, meta


def wait_fresh_observation(state_client, images, commands, stopped, record, timeout=5.):
    """Hold while acquiring valid cameras; never suppress feedback faults."""
    deadline = time.monotonic() + timeout
    while not stopped.is_set():
        commands.hold()
        try:
            observation = read_observation(state_client, images, record)
            if time.monotonic() > deadline:
                raise RuntimeError("camera synchronization wait timed out")
            return observation
        except RuntimeError as exc:
            camera_error = str(exc).startswith(("camera frame unavailable:",
                "stale camera frame ", "cross-camera skew too large:"))
            if not camera_error or time.monotonic() >= deadline:
                raise
            stopped.wait(.05)
    return None


def read_rtc_observation(state_client, images, commands, controller, stopped, record):
    """Reacquire valid cameras after skew/stale frames; never retry hardware faults."""
    try:
        return read_observation(state_client, images, record)
    except RuntimeError as exc:
        if not str(exc).startswith(("cross-camera skew too large:", "stale camera frame ")):
            raise
        controller.invalidate()
        started = time.monotonic()
        # No commands from the old plan may execute while cameras resynchronize.
        commands.hold()
        record({"event": "rtc_camera_sync_wait", "reason": str(exc), "timeout_ms": 500})
        observation = wait_fresh_observation(
            state_client, images, commands, stopped, record, timeout=.5,
        )
        if observation is not None:
            record({"event": "rtc_camera_sync_restored", "elapsed_ms":
                    (time.monotonic()-started)*1000, "old_actions_discarded": True})
        return observation


def main():
    args = build_parser().parse_args()
    if args.num_inference_steps <= 0:
        raise ValueError("num-inference-steps must be positive")
    if (args.confirm_real_robot != REAL_ROBOT_CONFIRMATION
            or os.environ.get("R1A7_REAL_INFERENCE") != "YES"):
        raise PermissionError("explicit R1A7 real-robot authorization required")
    if not sys.stdin.isatty():
        raise ValueError("RTC real experiment requires a terminal for P/R/Q controls")
    if args.max_commands < 0:
        raise ValueError("max-commands must be non-negative")
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    gate = DispatchGate()
    console = RTCConsole()
    task_contract = resolve_task_contract(args, read_training_tasks(args.dataset_root))
    reference, _ = load_task_start_reference(args.dataset_root, task_contract["reference_task_before"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_client = command_client = images = events = controller = None
    with args.output.open("x", encoding="utf-8", buffering=1) as log:
        log_lock = threading.Lock()

        def record(event):
            with log_lock:
                log.write(json.dumps({**event, "monotonic_ns": time.monotonic_ns(),
                                      "wall_time_ns": time.time_ns()}, ensure_ascii=False) + "\n")

        record({"event": "rtc_task_contract", **task_contract})
        tasks = InferenceTaskState(
            expected_barcode=args.expected_barcode, task_before=args.task_before,
            task_after=args.task_after, scan_armed_after_ns=time.time_ns(),
        )
        tasks.pause_for_next_cycle()  # scans cannot switch task during initialization

        def accept(event):
            with gate.lock:
                if gate.paused:
                    return {"status": "error", "message": "RTC is paused"}
                if gate.scan_accept_after_ns and (
                    not isinstance(event.get("source_wall_time_ns"), int)
                    or event["source_wall_time_ns"] < gate.scan_accept_after_ns
                ):
                    return {"status": "error", "message": "scan predates RTC resume"}
                old_generation = tasks.snapshot().generation
                reply = tasks.handle_item_accepted(event)
                if tasks.snapshot().generation != old_generation:
                    gate.invalidate()
                    record({"event": "rtc_item_accepted", "source_event": event})
                return reply

        try:
            command_client = RealCommandClient(args.bridge_endpoint)
            state_client = StateBridgeClient(args.bridge_endpoint, 1000)
            images = ImageClient(host=args.image_host, request_bgr=True)
            # Loading weights does not start a policy stream; the existing bridge
            # owns hardware feedback protections and pre-policy holding.
            session = RTCSession(
                dataset_root=args.dataset_root, policy_path=args.policy_path,
                tokenizer_path=args.tokenizer_path, device_name=args.device,
                seed=args.seed, execution_horizon=args.execution_steps,
                max_guidance_weight=args.max_guidance_weight,
                mask_mode=args.rtc_mask_mode,
                num_inference_steps=args.num_inference_steps,
            )
            record({"event": "rtc_initializing", "phase": "camera_ready"})
            observation = wait_fresh_observation(state_client, images, command_client, stopped, record)
            if observation is None:
                return
            state, rgb, _ = observation
            record({"event": "rtc_initializing", "phase": "inference_warmup"})
            warm_started = time.monotonic()
            # Exercise both model paths and postprocessing without dispatching
            # any predicted targets. Cold CUDA work must precede READY.
            warm = session.predict_normalized(state, rgb, args.task_before)
            warm = session.predict_normalized(state, rgb, args.task_before,
                prefix=warm[args.execution_steps:].copy(), delay_steps=args.initial_delay_steps)
            for action in warm:
                session.to_hardware_action(action)
            session.reset()
            record({"event": "rtc_warmup_complete", "elapsed_ms":
                    (time.monotonic()-warm_started)*1000, "policy_commands_sent": False})
            if stopped.is_set():
                return
            controller = RTCController(
                session, execution_steps=args.execution_steps,
                prefetch_steps=args.prefetch_steps, initial_delay_steps=args.initial_delay_steps,
            )
            if return_to_scan_start_pose(command_client, state_client, reference, args, log, stopped) is None:
                return
            if wait_fresh_observation(state_client, images, command_client, stopped, record) is None:
                return
            tasks.start_restarted_cycle()
            events = TaskEventServer(args.task_event_bind, accept)
            events.start()
            record({"event": "rtc_ready", "frequency": 15, "checkpoint": str(args.policy_path),
                    "dataset_root": str(args.dataset_root),
                    "task_contract": task_contract,
                    "execution_steps": args.execution_steps, "prefetch_steps": args.prefetch_steps,
                    "mask_mode": args.rtc_mask_mode,
                    "chunk_size": session.chunk_size,
                    "initial_delay_steps": args.initial_delay_steps,
                    "max_guidance_weight": args.max_guidance_weight,
                    "seed": args.seed,
                    "num_inference_steps": args.num_inference_steps,
                    "attention_schedule": "EXP"})
            print("RTC READY: P+Enter pause; N+Enter resume current task; R+Enter return/restart (empty grippers); Ctrl-C exits.", flush=True)
            revision = gate.revision
            commands = 0
            while not stopped.is_set():
                started = time.monotonic()
                for key in console.poll():
                    if key == "Q":
                        stopped.set()
                    elif key == "P":
                        gate.invalidate(paused=True)
                        controller.invalidate()
                        record({"event": "rtc_paused"})
                    elif key == "N" and gate.paused:
                        gate.invalidate(paused=False)
                        controller.invalidate()
                        record({"event": "rtc_resumed"})
                    elif key == "R":
                        gate.invalidate(paused=True)
                        controller.invalidate()
                        with gate.lock:
                            tasks.pause_for_next_cycle()
                        if return_to_scan_start_pose(command_client, state_client, reference, args, log, stopped) is None:
                            break
                        with gate.lock:
                            tasks.start_restarted_cycle()
                            gate.invalidate(paused=False)
                        record({"event": "rtc_restarted"})
                if stopped.is_set():
                    break
                with gate.lock:
                    current_revision = gate.revision
                    snapshot = tasks.snapshot()
                if revision != current_revision:
                    controller.invalidate()
                    revision = current_revision
                if gate.paused:
                    command_client.hold()
                    stopped.wait(max(0, args.interval - (time.monotonic()-started)))
                    continue
                observation = read_rtc_observation(
                    state_client, images, command_client, controller, stopped, record)
                if observation is None:
                    break
                state, rgb, meta = observation
                result = controller.poll()
                if result is not None:
                    record({**result, "generation": controller.timeline.generation})
                controller.request_if_needed(state, rgb, snapshot.task)
                normalized = controller.pop()
                if normalized is None:
                    # A depleted horizon breaks continuity. Invalidate a late
                    # result before any future bootstrap; do not replay old targets.
                    if controller.timeline.step:
                        controller.invalidate()
                    held = command_client.hold()
                    record({"event": "rtc_queue_wait", "hardware_commands_sent": held["hardware_commands_sent"]})
                else:
                    raw = controller.last_hardware_action
                    try:
                        safe, clipping = prepare_safe_real_action(raw)
                    except ValueError as exc:
                        controller.invalidate()
                        command_client.hold()
                        record({"event": "rtc_action_rejected", "reason": str(exc)})
                    else:
                        executed = gate.dispatch(revision, lambda: command_client.command(safe))
                        if executed is None:
                            controller.invalidate()
                            command_client.hold()
                        else:
                            commands += 1
                            actual = np.asarray(executed["executed_action"])
                            # The worker already projects targets in model space.
                            # Float-roundoff clamping at dispatch is not a new plan.
                            unexpected_clip = any(abs(c["delta"]) > 1e-6 for c in clipping)
                            if unexpected_clip or not np.allclose(actual, safe, rtol=0, atol=1e-6):
                                controller.invalidate()
                            record({"event": "rtc_command", "task": asdict(snapshot),
                                    "raw_action": raw.tolist(), "executed_action": actual.tolist(),
                                    "clipping": clipping, "input": meta,
                                    "hardware_commands_sent": executed["hardware_commands_sent"]})
                            if args.max_commands and commands >= args.max_commands:
                                stopped.set()
                cycle_ms = (time.monotonic()-started)*1000
                if cycle_ms > args.interval*1000:
                    record({"event": "rtc_cycle_overrun", "cycle_ms": cycle_ms,
                            **meta["timing"]})
                stopped.wait(max(0, args.interval-(time.monotonic()-started)))
        except Exception as exc:
            gate.invalidate(paused=True)
            if controller is not None:
                controller.invalidate()
            record({"event": "rtc_fault", "reason": str(exc)})
            raise
        finally:
            # Disarm before waiting on GPU work, not after worker shutdown.
            if command_client is not None:
                command_client.disarm()
            if events is not None:
                events.stop()
            if controller is not None:
                controller.close()
            if images is not None:
                images.close()
            if state_client is not None:
                state_client.close()
            if command_client is not None:
                command_client.close()


if __name__ == "__main__":
    main()
