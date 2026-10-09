# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Multi-trial PI0.5 rollout with explicit PICO intervention and restart."""
from __future__ import annotations
from collections import deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import numpy as np
import zmq
from teleimager.image_client import ImageClient
from teleop.utils.task_events import TaskEventServer
from unitree_lerobot.r1a7_async_policy import AsyncActionChunkProducer, select_executable_window
from unitree_lerobot.r1a7_dagger_recording import DaggerRecorder, collection_path, collection_name
from unitree_lerobot.r1a7_hardware_gateway import REAL_ROBOT_CONFIRMATION
from unitree_lerobot.r1a7_inference_task_state import InferenceTaskState
from unitree_lerobot.r1a7_live_action_conditioning import ActionChunkBlender, load_task_start_reference, prepare_safe_real_action
from unitree_lerobot.r1a7_live_real import RealCommandClient, build_parser, return_to_scan_start_pose
from unitree_lerobot.r1a7_live_shadow import PI05Session, StateBridgeClient, _read_cameras, assemble_live_observation


class ReturnInterrupted(RuntimeError):
    """Healthy operator B interruption; never resume motion automatically."""
    def __init__(self, reply):
        self.reply = reply
        super().__init__(f"return interrupted by operator B: {reply}")


def operator_b_hold(reply):
    healthy = (reply.get("fault_locked") is False or
               (reply.get("status") == "ok" and "fault_reason" in reply
                and reply["fault_reason"] is None and not reply.get("fault_locked")))
    return (healthy and not reply.get("fault_reason")
            and reply.get("control_source") == "hold"
            and reply.get("takeover_reason") == "operator B")


class DaggerClient(RealCommandClient):
    def __init__(self, *args, **kwargs):
        self.owner_epoch = 0
        self.strict_rejections = True
        super().__init__(*args, **kwargs)

    def _request(self, payload):
        payload = {**payload, "owner_epoch": self.owner_epoch}
        try:
            self._socket.send_json(payload)
            reply = self._socket.recv_json()
        except zmq.Again as exc:
            self._reset()
            raise RuntimeError("DAgger bridge timed out") from exc
        if reply.get("status") not in {"ok", "rejected"}:
            raise RuntimeError(f"DAgger bridge failed: {reply}")
        if self.strict_rejections and reply.get("status") == "rejected":
            if operator_b_hold(reply):
                raise ReturnInterrupted(reply)
            raise RuntimeError(f"startup interrupted; restart launcher: {reply}")
        return reply


def return_for_restart(commands, state_client, reference, args, audit, stopped):
    """True only after return completed with policy ownership still active."""
    commands.strict_rejections = True
    try:
        return_to_scan_start_pose(commands, state_client, reference, args, audit, stopped)
        if stopped.is_set():
            return False
        ready = commands._request({"request": "dagger_start_ready"})
        if ready.get("control_source") != "policy":
            if operator_b_hold(ready):
                raise ReturnInterrupted(ready)
            raise RuntimeError(f"return completed without policy ownership: {ready}")
        return True
    except ReturnInterrupted as exc:
        commands.strict_rejections = False
        commands.owner_epoch = exc.reply["owner_epoch"]
        # Mark as between trials: A cannot move the robot while restart awaits
        # explicit R. No recorder or scanner arming is created for this attempt.
        held = commands._request({"request": "dagger_between_trials"})
        if held.get("status") != "ok" or held.get("fault_reason") or held.get("fault_locked"):
            raise RuntimeError(f"return interruption HOLD failed: {held}")
        commands.owner_epoch = held["owner_epoch"]
        audit.write(json.dumps({"event": "dagger_return_interrupted", "reason": "operator B",
                                "owner_epoch": commands.owner_epoch}) + "\n")
        print("Return interrupted by B; remain in HOLD. Clear workspace, then press R "
              "or release and press RIGHT GRIP to retry.", flush=True)
        return False
    finally:
        commands.strict_rejections = False


class Console:
    def __init__(self):
        self.commands = queue.SimpleQueue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in sys.stdin:
            command = line.strip().upper()
            if command in {"G", "T", "P", "R", "S", "Q"}:
                self.commands.put(command)

    def poll(self):
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None


def build_dagger_parser():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--dagger-record-dir", required=True, type=Path)
    parser.add_argument("--collection-name", help="Optional override; defaults to dataset-root folder name")
    from .r1a7_dagger import takeover_blend_seconds
    parser.add_argument("--takeover-blend-seconds", type=takeover_blend_seconds, default=3.0)
    return parser


def main(argv=None):
    parser = build_dagger_parser()
    args = parser.parse_args(argv)
    args.collection_name = collection_name(args.dataset_root, args.collection_name)
    record_root = collection_path(args.dagger_record_dir, args.collection_name)
    if args.confirm_real_robot != REAL_ROBOT_CONFIRMATION or os.environ.get("R1A7_REAL_INFERENCE") != "YES":
        raise PermissionError("explicit real robot authorization required")
    if not sys.stdin.isatty():
        raise RuntimeError("DAgger requires an interactive terminal for completion/rejection")
    if args.manual_cycles or args.completion_wait_seconds:
        raise ValueError("DAgger uses explicit P/R cycles, not automatic/manual inference resets")
    if args.interval != 1 / 15:
        raise ValueError("DAgger recording/control interval is fixed to 15 Hz")
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    state_client = StateBridgeClient(args.bridge_endpoint, 1000)
    commands = DaggerClient(args.bridge_endpoint)
    image_client = event_server = producer = recorder = None
    completed = False
    try:
        image_client = ImageClient(host=args.image_host, request_bgr=True)
        task = InferenceTaskState(expected_barcode=args.expected_barcode,
                                  task_before=args.task_before, task_after=args.task_after)
        events = queue.SimpleQueue()
        task_lock = threading.RLock()
        accepting_events = False

        def accepted(event):
            with task_lock:
                if not accepting_events:
                    return {"status": "rejected", "reason": "no active DAgger trial"}
                reply = task.handle_item_accepted(event)
                events.put({"incoming": event, "reply": reply, "task": asdict(task.snapshot())})
                return reply

        reply = state_client.read()
        if "owner_epoch" not in reply:
            raise RuntimeError("bridge was not started with --dagger")
        images = _read_cameras(image_client)
        deadline = time.monotonic() + args.readiness_timeout
        while True:
            try:
                assemble_live_observation(state_client.read(), images,
                    now_monotonic_ns=time.monotonic_ns(), max_state_age_ms=args.max_state_age_ms,
                    max_camera_age_ms=args.max_camera_age_ms, max_camera_skew_ms=args.max_camera_skew_ms)
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                stopped.wait(0.1)
                images = _read_cameras(image_client)
        print("Loading PI0.5; PICO website is ready in standby. No separate teleop process.", flush=True)
        session = PI05Session(dataset_root=args.dataset_root, policy_path=args.policy_path,
                              tokenizer_path=args.tokenizer_path, device_name=args.device, seed=args.seed)
        reference, _count = load_task_start_reference(args.dataset_root, args.task_before)
        if not 0 < args.async_prefetch_actions < session.n_action_steps or args.async_prefetch_actions + session.n_action_steps > session.chunk_size:
            raise ValueError("invalid prefetch/action horizon")
        console = Console()
        print("Connect PICO pass-through now. Clear the workspace, then G + Enter or release "
              "and press the RIGHT inner grip button to "
              "move to the scan start and begin recording/model rollout.", flush=True)
        previous_squeeze = None
        start_baseline = state_client.read().get("start_press_sequence", 0)
        while not stopped.is_set():
            ready = state_client.read()
            squeeze = bool(ready.get("right_squeeze", False))
            # Require a fresh release after model loading; a held button cannot start motion.
            grip_start = ready.get("xr_ready") and previous_squeeze is False and squeeze
            grip_start = grip_start or (ready.get("xr_ready") and
                                       ready.get("start_press_sequence", 0) > start_baseline)
            previous_squeeze = squeeze if ready.get("xr_ready") else None
            if console.poll() == "G" or grip_start:
                if ready.get("xr_ready"):
                    print("DAgger start confirmed by " + ("RIGHT GRIP" if grip_start else "keyboard G"), flush=True)
                    break
                print("PICO motion stream is not fresh; enter pass-through and retry G.", flush=True)
            stopped.wait(0.05)
        if stopped.is_set():
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", buffering=1) as audit:
            # Enable XR/host loss protection before the bounded start movement.
            commands._request({"request": "dagger_heartbeat"})
            return_to_scan_start_pose(commands, state_client, reference, args, audit, stopped)
            if stopped.is_set():
                return 0
            commands._request({"request": "dagger_start_ready"})
            commands.strict_rejections = False
            source_metadata = {"collection_name": args.collection_name,
                               "dataset_root": str(args.dataset_root)}
            recorder = DaggerRecorder(record_root, args.policy_path, source_metadata=source_metadata)
            task = InferenceTaskState(expected_barcode=args.expected_barcode,
                                      task_before=args.task_before, task_after=args.task_after,
                                      run_id=recorder.trial_id, scan_armed_after_ns=time.time_ns())
            accepting_events = True
            event_server = TaskEventServer(args.task_event_bind, accepted)
            event_server.start()
            producer = AsyncActionChunkProducer(session)
            actions = deque()
            blender = ActionChunkBlender(args.chunk_blend_steps)
            generation = None
            inflight = False
            sent = 0
            cycle = 1
            waiting = False
            restart_baseline = state_client.read().get("restart_press_sequence", 0)

            def finish_trial(save, *, delete=False):
                nonlocal recorder, waiting, accepting_events, inflight, restart_baseline
                # Explicit Q/R discards are not dataset candidates. Preserve
                # interrupted/faulted finalization separately in cleanup.
                delete = delete or not save
                commands._request({"request": "dagger_between_trials"})
                with task_lock:
                    accepting_events = False
                    task.pause_for_restart()
                    while not events.empty():
                        recorder.event("scanner_event", **events.get_nowait())
                recorder.event("operator_completion", completed=save)
                finished_recorder = recorder
                recorder = None
                finish_errors = []
                finished = threading.Event()
                cancel_delete = threading.Event()
                accepted_id = task.snapshot().accepted_event_id
                def flush_trial():
                    try:
                        finished_recorder.finish(completed=save, accepted_event_id=accepted_id)
                        if delete and not stopped.is_set() and not cancel_delete.is_set():
                            finished_recorder.discard()
                    except Exception as exc:
                        finish_errors.append(exc)
                    finally:
                        finished.set()
                threading.Thread(target=flush_trial, daemon=True).start()
                # Storage may be slow; keep the healthy HOLD bridge heartbeating.
                # This client stays exclusively on the coordinator thread.
                try:
                    while not finished.wait(0.05):
                        if stopped.is_set():
                            raise RuntimeError("interrupted while finalizing trial; shutdown disarms hardware")
                        commands._request({"request": "dagger_heartbeat"})
                except BaseException:
                    cancel_delete.set()  # Preserve data if shutdown precedes deletion.
                    raise
                if finish_errors:
                    raise RuntimeError(f"trial finalization failed at {finished_recorder.path}") from finish_errors[0]
                # Ignore presses during finalization; only a new press after the
                # ready/HOLD prompt may start a subsequent cycle.
                restart_baseline = state_client.read().get("restart_press_sequence", 0)
                result_label = "deleted" if delete else "saved" if save else "rejected audit retained"
                print(f"DAgger {result_label}: {finished_recorder.path}; HOLD. "
                      "Empty grippers and clear workspace before R + Enter.", flush=True)
                waiting = True
                actions.clear()
                inflight = False

            print("DAgger: PICO A = takeover, B = HOLD; P/T+Enter = HOLD then A align; "
                  "Left X / S+Enter = completed trial; Left Y / Q+Enter = delete current trial. "
                  "R+Enter or fresh RIGHT GRIP in HOLD = clear grippers/workspace, return to start, new trial. "
                  "After takeover, match each trigger to the held grip to enable it.", flush=True)
            while not stopped.is_set():
                started = time.monotonic()
                if recorder is not None:
                    recorder.check()
                owner = commands._request({"request": "dagger_heartbeat"})
                operator = console.poll()
                restart_sequence = owner.get("restart_press_sequence", 0)
                grip_restart = restart_sequence > restart_baseline
                restart_baseline = restart_sequence  # Consume even if no longer eligible.
                if (operator is None and grip_restart and owner.get("xr_ready")
                        and (waiting or owner["control_source"] == "hold")):
                    operator = "R"
                pico_finish = owner.get("operator_finish")
                if operator == "Q" or pico_finish == "Q":
                    operator = "Q"
                elif pico_finish == "S":
                    operator = "S"
                if operator in {"T", "P"} and not waiting:
                    owner = commands._request({"request": "takeover_hold"})
                if operator in {"S", "Q"} and not waiting:
                    commands._request({"request": "takeover_hold"})
                    if operator == "S" and task.snapshot().accepted_event_id is None:
                        recorder.event("completion_refused", reason="scan not accepted")
                        commands._request({"request": "dagger_ack_finish"})
                        print("Not saved: scan not accepted. HOLD; press A to continue this trial.", flush=True)
                        stopped.wait(args.interval)
                        continue
                    finish_trial(operator == "S", delete=operator == "Q")
                if operator == "R":
                    if not waiting and owner["control_source"] != "hold":
                        print("Press P to HOLD before R. Restart not executed.", flush=True)
                        stopped.wait(args.interval)
                        continue
                    if not owner.get("xr_ready"):
                        print("Restart refused: PICO is not fresh. Remain in HOLD.", flush=True)
                        stopped.wait(args.interval)
                        continue
                    if not waiting:
                        finish_trial(False)
                    restarted = commands._request({"request": "dagger_prepare_restart"})
                    if restarted.get("status") == "rejected":
                        print(f"Restart refused: {restarted}", flush=True)
                        stopped.wait(args.interval)
                        continue
                    commands.owner_epoch = restarted["owner_epoch"]
                    commands.strict_rejections = True
                    print("R confirmed: returning to start. Work area and grippers must be clear.", flush=True)
                    returned = return_for_restart(commands, state_client, reference, args, audit, stopped)
                    if stopped.is_set():
                        break
                    if not returned:
                        actions.clear()
                        inflight = False
                        producer.poll()
                        restart_baseline = state_client.read().get("restart_press_sequence", 0)
                        stopped.wait(args.interval)
                        continue
                    restart_baseline = state_client.read().get("restart_press_sequence", 0)
                    commands.strict_rejections = False
                    recorder = DaggerRecorder(record_root, args.policy_path, source_metadata=source_metadata)
                    with task_lock:
                        task = InferenceTaskState(expected_barcode=args.expected_barcode,
                            task_before=args.task_before, task_after=args.task_after,
                            run_id=recorder.trial_id, scan_armed_after_ns=time.time_ns())
                        accepting_events = True
                    cycle += 1
                    sent = 0
                    generation = None
                    waiting = False
                    owner = commands._request({"request": "dagger_heartbeat"})
                if waiting:
                    producer.poll()
                    stopped.wait(args.interval)
                    continue
                snapshot = task.snapshot()
                current_generation = cycle * 1_000_000_000 + snapshot.generation * 1_000_000 + owner["owner_epoch"]
                if current_generation != generation:
                    actions.clear()
                    blender = ActionChunkBlender(args.chunk_blend_steps)
                    if owner.get("commanded_action"):
                        blender.prime(owner["commanded_action"])
                    inflight = False
                    generation = current_generation
                    recorder.event("control_boundary", owner=owner["control_source"],
                                   epoch=owner["owner_epoch"], task=asdict(snapshot),
                                   reason=owner.get("takeover_reason"))
                commands.owner_epoch = owner["owner_epoch"]
                state_reply = state_client.read()
                if state_reply.get("fault_reason"):
                    raise RuntimeError(state_reply["fault_reason"])
                try:
                    state, rgb, image_meta = assemble_live_observation(
                        state_reply, _read_cameras(image_client),
                        now_monotonic_ns=time.monotonic_ns(), max_state_age_ms=args.max_state_age_ms,
                        max_camera_age_ms=args.max_camera_age_ms, max_camera_skew_ms=args.max_camera_skew_ms)
                except RuntimeError as exc:
                    print(f"DAgger observation HOLD: {exc}", flush=True)
                    recorder.gap(exc)
                    commands._request({"request": "takeover_hold"})
                    stopped.wait(args.interval)
                    continue
                # Record accepted targets plus actual limiter gripper targets, not
                # arbitrary trigger values. HOLD/model prefixes are audit only.
                stamp = time.monotonic_ns()
                command_stamp = state_reply.get("command_monotonic_ns", 0)
                if command_stamp and 0 <= stamp - command_stamp <= 120_000_000:
                    recorder.add(state, rgb, state_reply["commanded_action"], {
                        "active_task": snapshot.task, "task_phase": snapshot.phase,
                        "control_source": state_reply["command_source"],
                        "owner_epoch": state_reply["command_owner_epoch"], "sample_valid": True,
                        "active_owner": state_reply["control_source"],
                        "takeover_reason": state_reply.get("takeover_reason"),
                        "record_monotonic_ns": stamp, "command_monotonic_ns": command_stamp,
                        "camera_metadata": image_meta, "state_timestamps": state_reply["sample_monotonic_ns"],
                        "requested_action": state_reply["requested_action"],
                        "accepted_event_id": snapshot.accepted_event_id,
                        "trigger_pickup": state_reply["trigger_pickup"],
                        "takeover_blend": state_reply.get("takeover_blend"),
                        "expert_action_valid": (state_reply.get("takeover_blend") or {}).get(
                            "expert_action_valid", True),
                    })
                else:
                    recorder.gap("no recent commanded target")
                while not events.empty():
                    recorder.event("scanner_event", **events.get_nowait())
                if state_reply["control_source"] != "policy":
                    actions.clear()
                    producer.poll()  # Drain; in-flight computation cannot command hardware.
                    stopped.wait(max(0, args.interval - (time.monotonic() - started)))
                    continue
                result = producer.poll()
                if result is not None and result.generation == generation:
                    inflight = False
                    if result.error:
                        raise result.error
                    window = select_executable_window(result.actions, lead_actions=result.lead_actions,
                                                      n_action_steps=session.n_action_steps)
                    actions.extend((row, index == 0) for index, row in enumerate(window))
                if not inflight and len(actions) <= args.async_prefetch_actions:
                    producer.submit(generation=generation, task=snapshot.task, lead_actions=len(actions),
                                    state=state, images=rgb)
                    inflight = True
                if actions and task.snapshot().generation == snapshot.generation:
                    raw, new_chunk = actions.popleft()
                    # No prediction may bypass the ordinary hard-position contract.
                    try:
                        safe, clipping = prepare_safe_real_action(
                            raw, max_arm_overshoot=args.max_arm_limit_overshoot,
                            clip_arm_overshoot=True)
                        action, blend_meta = blender.apply(safe, new_chunk=new_chunk)
                    except ValueError as exc:
                        # Malformed/non-finite predictions still fail closed: hold
                        # the current pose/grip for one cycle, then keep running.
                        # Do not catch hardware-command failures here.
                        hold_reply = commands.hold()
                        rejection = {"event": "dagger_prediction_rejected", "reason": str(exc),
                                     "raw_action": raw.tolist(), "task": asdict(snapshot),
                                     "hold_status": hold_reply.get("status")}
                        audit.write(json.dumps(rejection) + "\n")
                        recorder.event("prediction_rejected", reason=str(exc), task=asdict(snapshot))
                        print(f"DAgger HOLD one cycle; skipped prediction: {exc}", flush=True)
                        stopped.wait(max(0, args.interval - (time.monotonic() - started)))
                        continue
                    # Linearize scan acceptance and action commit: a task switch
                    # may occur while conditioning the prediction above.
                    with task_lock:
                        if task.snapshot().generation != snapshot.generation:
                            actions.clear()
                            continue
                        response = commands.command(action)
                    if response.get("status") == "ok":
                        sent += 1
                        audit.write(json.dumps({"event": "dagger_model_command", "command": sent,
                                                "raw_action": raw.tolist(), "action": action.tolist(),
                                                "clipping": clipping, "owner_epoch": commands.owner_epoch,
                                                "task": asdict(snapshot)}) + "\n")
                        if sent % 15 == 0:
                            print(f"command={sent} phase={snapshot.phase} owner=policy", flush=True)
                else:
                    commands.hold()  # Preserves grip; rejected harmlessly after A edge.
                if args.max_commands and sent >= args.max_commands:
                    recorder.event("command_cap_reached")
                    break
                stopped.wait(max(0, args.interval - (time.monotonic() - started)))
    finally:
        # Stop actuators BEFORE flushing files or joining model work.
        cleanup_failed = []
        def clean(name, callback):
            try:
                callback()
            except Exception as exc:
                cleanup_failed.append((name, exc))
                print(f"DAgger cleanup failed ({name}): {exc}", file=sys.stderr, flush=True)
        clean("disarm", commands.disarm)
        if producer:
            clean("policy worker", producer.close)
        if event_server:
            clean("scanner server", event_server.stop)
        if image_client:
            clean("image client", image_client.close)
        clean("state client", state_client.close)
        clean("command client", commands.close)
        if recorder:
            clean("recorder", lambda: recorder.finish(completed=completed and not cleanup_failed,
                  accepted_event_id=task.snapshot().accepted_event_id))
            print(f"DAgger audit directory: {recorder.path}", flush=True)
        if cleanup_failed and sys.exc_info()[0] is None:
            raise RuntimeError(f"DAgger shutdown failed: {cleanup_failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
