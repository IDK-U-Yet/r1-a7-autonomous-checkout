# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Run PI0.5 on live R1-A7 inputs without publishing robot commands."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zmq

from unitree_lerobot.r1a7_contract import (
    ACTION_LOWER_LIMITS,
    ACTION_UPPER_LIMITS,
    CAMERA_KEYS,
    CAMERA_SHAPES,
    JOINT_NAMES,
    ROBOT_TYPE,
    validate_action_vector,
    validate_state_vector,
)
from unitree_lerobot.r1a7_inference_task_state import (
    DEFAULT_EXPECTED_BARCODE,
    DEFAULT_TASK_AFTER,
    DEFAULT_TASK_BEFORE,
    InferenceTaskState,
)
from teleop.utils.task_events import TaskEventServer


CAMERA_GETTERS = {
    "observation.images.cam_global": "get_head_frame",
    "observation.images.cam_left_wrist": "get_left_wrist_frame",
    "observation.images.cam_right_wrist": "get_right_wrist_frame",
    "observation.images.cam_scanner": "get_scanner_frame",
}
GRIPPER_INDICES = (14, 15)


def prepare_safe_shadow_action(
    values: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    """Clip only grippers; keep arm hard-limit validation strict."""

    raw_action = np.asarray(values, dtype=np.float64)
    if raw_action.shape != (len(JOINT_NAMES),):
        # Reuse the contract's exact shape/finite error reporting.
        validate_action_vector(raw_action)
    safe_action = raw_action.copy()
    clipping: list[dict[str, float | str]] = []
    for index in GRIPPER_INDICES:
        raw_value = float(raw_action[index])
        safe_value = float(
            np.clip(raw_value, ACTION_LOWER_LIMITS[index], ACTION_UPPER_LIMITS[index])
        )
        if safe_value != raw_value:
            clipping.append(
                {
                    "joint": JOINT_NAMES[index],
                    "raw": raw_value,
                    "safe": safe_value,
                    "delta": safe_value - raw_value,
                }
            )
            safe_action[index] = safe_value
    return validate_action_vector(safe_action), clipping


class StateBridgeClient:
    def __init__(self, endpoint: str, timeout_ms: int):
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

    def read(self) -> Mapping[str, Any]:
        try:
            self._socket.send_json({"request": "state"})
            reply = self._socket.recv_json()
        except zmq.Again as exc:
            self._socket.close(linger=0)
            self._connect()
            raise RuntimeError("read-only state bridge timed out") from exc
        if reply.get("status") != "ok":
            raise RuntimeError(f"read-only state bridge is not ready: {reply}")
        return reply

    def close(self) -> None:
        self._socket.close(linger=0)


def assemble_live_observation(
    state_reply: Mapping[str, Any],
    camera_frames: Mapping[str, Any],
    *,
    now_monotonic_ns: int,
    max_state_age_ms: float,
    max_camera_age_ms: float,
    max_camera_skew_ms: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Validate one synchronized-enough live sample and convert BGR to RGB CHW."""

    state = validate_state_vector(state_reply.get("state", ()))
    state_times = state_reply.get("sample_monotonic_ns", {})
    try:
        state_ages = {
            name: (now_monotonic_ns - int(timestamp)) / 1e6
            for name, timestamp in state_times.items()
        }
    except (TypeError, ValueError) as exc:
        raise RuntimeError("state bridge returned invalid timestamps") from exc
    if set(state_ages) != {"lowstate", "left_gripper", "right_gripper"}:
        raise RuntimeError("state bridge did not return all three source timestamps")
    if min(state_ages.values()) < 0 or max(state_ages.values()) > max_state_age_ms:
        raise RuntimeError(f"stale robot state: ages_ms={state_ages}")

    images: dict[str, np.ndarray] = {}
    camera_times: dict[str, int] = {}
    camera_meta: dict[str, Any] = {}
    for key in CAMERA_KEYS:
        frame = camera_frames.get(key)
        if frame is None or frame.bgr is None:
            raise RuntimeError(f"camera frame unavailable: {key}")
        bgr = np.asarray(frame.bgr)
        if bgr.shape != CAMERA_SHAPES[key]:
            raise RuntimeError(
                f"camera shape mismatch for {key}: {bgr.shape}, expected {CAMERA_SHAPES[key]}"
            )
        timestamp = frame.capture_monotonic_ns or frame.receive_monotonic_ns
        if timestamp is None:
            raise RuntimeError(f"camera timestamp unavailable: {key}")
        timestamp = int(timestamp)
        age_ms = (now_monotonic_ns - timestamp) / 1e6
        if age_ms < 0 or age_ms > max_camera_age_ms:
            raise RuntimeError(f"stale camera frame {key}: age_ms={age_ms:.3f}")
        camera_times[key] = timestamp
        camera_meta[key] = {
            "frame_sequence": frame.frame_sequence,
            "age_ms": age_ms,
            "fps": float(frame.fps),
        }
        images[key] = np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1))
    skew_ms = (max(camera_times.values()) - min(camera_times.values())) / 1e6
    if skew_ms > max_camera_skew_ms:
        raise RuntimeError(f"cross-camera skew too large: {skew_ms:.3f} ms")
    return state, images, {
        "state_age_ms": state_ages,
        "cameras": camera_meta,
        "cross_camera_skew_ms": skew_ms,
    }


class PI05Session:
    """Loaded-once PI0.5 policy and processors for live inputs."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        policy_path: Path,
        tokenizer_path: Path,
        device_name: str,
        seed: int,
    ):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        self.torch = torch
        self.device = torch.device(device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        config = PreTrainedConfig.from_pretrained(str(policy_path))
        config.device = str(self.device)
        # Training checkpoints may persist gradient checkpointing. Inference
        # needs no backward graph, and disabling it also reduces GPU pressure.
        config.gradient_checkpointing = False
        config.pretrained_path = str(policy_path)
        config.tokenizer_name = str(tokenizer_path)
        dataset = LeRobotDataset(
            repo_id=dataset_root.name, root=dataset_root, video_backend="pyav"
        )
        self.policy = make_policy(cfg=config, ds_meta=dataset.meta)
        self.policy.eval()
        if config.type == "pi05":
            importlib.import_module("lerobot.policies.pi05.processor_pi05")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(policy_path),
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "rename_observations_processor": {"rename_map": {}},
                "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)},
            },
        )
        self.use_amp = bool(config.use_amp)
        self.chunk_size = int(config.chunk_size)
        self.n_action_steps = int(config.n_action_steps)
        torch.manual_seed(seed)
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.last_prediction_started_new_chunk = False

    def predict(
        self, state: np.ndarray, images: Mapping[str, np.ndarray], task: str
    ) -> np.ndarray:
        torch = self.torch
        observation: dict[str, Any] = {
            "observation.state": torch.from_numpy(state.astype(np.float32))
            .unsqueeze(0)
            .to(self.device),
            "task": task,
            "robot_type": ROBOT_TYPE,
        }
        for key in CAMERA_KEYS:
            observation[key] = (
                torch.from_numpy(images[key].astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(self.device)
            )
        amp_context = (
            torch.autocast(device_type="cuda")
            if self.device.type == "cuda" and self.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), amp_context:
            batch = self.preprocessor(observation)
            action_queue = getattr(self.policy, "_action_queue", None)
            self.last_prediction_started_new_chunk = (
                action_queue is not None and len(action_queue) == 0
            )
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)
        return action.squeeze(0).detach().to("cpu", dtype=torch.float64).numpy()

    def predict_chunk(
        self, state: np.ndarray, images: Mapping[str, np.ndarray], task: str
    ) -> np.ndarray:
        """Predict and postprocess the full action horizon for timed consumption."""

        torch = self.torch
        observation: dict[str, Any] = {
            "observation.state": torch.from_numpy(state.astype(np.float32))
            .unsqueeze(0)
            .to(self.device),
            "task": task,
            "robot_type": ROBOT_TYPE,
        }
        for key in CAMERA_KEYS:
            observation[key] = (
                torch.from_numpy(images[key].astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(self.device)
            )
        amp_context = (
            torch.autocast(device_type="cuda")
            if self.device.type == "cuda" and self.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), amp_context:
            batch = self.preprocessor(observation)
            normalized = self.policy.predict_action_chunk(batch)
            actions = [
                self.postprocessor(normalized[:, index])
                .squeeze(0)
                .detach()
                .to("cpu", dtype=torch.float64)
                .numpy()
                for index in range(normalized.shape[1])
            ]
        return np.stack(actions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--image-host", default="127.0.0.1")
    parser.add_argument("--state-endpoint", default="tcp://127.0.0.1:55582")
    parser.add_argument("--task-event-bind", default="tcp://0.0.0.0:55580")
    parser.add_argument("--task-event-token", default=os.environ.get("R1_TASK_EVENT_TOKEN"))
    parser.add_argument("--expected-barcode", default=DEFAULT_EXPECTED_BARCODE)
    parser.add_argument("--task-before", default=DEFAULT_TASK_BEFORE)
    parser.add_argument("--task-after", default=DEFAULT_TASK_AFTER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1 / 15)
    parser.add_argument("--state-timeout-ms", type=int, default=1000)
    parser.add_argument("--max-state-age-ms", type=float, default=250.0)
    parser.add_argument("--max-camera-age-ms", type=float, default=250.0)
    parser.add_argument("--max-camera-skew-ms", type=float, default=100.0)
    parser.add_argument("--readiness-timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_cameras(client) -> dict[str, Any]:
    return {
        key: getattr(client, CAMERA_GETTERS[key])()
        for key in CAMERA_KEYS
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_steps < 0 or args.interval <= 0:
        raise ValueError("max-steps must be non-negative and interval must be positive")
    from teleimager.image_client import ImageClient

    state_client = StateBridgeClient(args.state_endpoint, args.state_timeout_ms)
    image_client = ImageClient(host=args.image_host, request_bgr=True)
    task_state = InferenceTaskState(
        expected_barcode=args.expected_barcode,
        task_before=args.task_before,
        task_after=args.task_after,
    )
    event_server = TaskEventServer(
        args.task_event_bind,
        task_state.handle_item_accepted,
        token=args.task_event_token,
    )
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda _s, _f: stopped.set())
    signal.signal(signal.SIGTERM, lambda _s, _f: stopped.set())
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        deadline = time.monotonic() + args.readiness_timeout
        while True:
            try:
                state_reply = state_client.read()
                camera_frames = _read_cameras(image_client)
                assemble_live_observation(
                    state_reply,
                    camera_frames,
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

        print("Live inputs ready; loading PI0.5 checkpoint...", flush=True)
        session = PI05Session(
            dataset_root=args.dataset_root.expanduser().resolve(),
            policy_path=args.policy_path.expanduser().resolve(),
            tokenizer_path=args.tokenizer_path.expanduser().resolve(),
            device_name=args.device,
            seed=args.seed,
        )
        event_server.start()
        last_generation = task_state.snapshot().generation
        step = 0
        with output.open("a", encoding="utf-8", buffering=1) as log:
            ready = {
                "event": "ready",
                "run_id": task_state.run_id,
                "task_event_bind": args.task_event_bind,
                "hardware_commands_sent": False,
                "wall_time_ns": time.time_ns(),
            }
            print(json.dumps(ready, ensure_ascii=False), flush=True)
            log.write(json.dumps(ready, ensure_ascii=False) + "\n")
            while not stopped.is_set() and (args.max_steps == 0 or step < args.max_steps):
                loop_start = time.monotonic()
                snapshot = task_state.snapshot(
                    last_consumed_generation=last_generation
                )
                if snapshot.policy_reset_required:
                    session.reset()
                    last_generation = snapshot.generation
                try:
                    state_reply = state_client.read()
                    state, images, input_meta = assemble_live_observation(
                        state_reply,
                        _read_cameras(image_client),
                        now_monotonic_ns=time.monotonic_ns(),
                        max_state_age_ms=args.max_state_age_ms,
                        max_camera_age_ms=args.max_camera_age_ms,
                        max_camera_skew_ms=args.max_camera_skew_ms,
                    )
                    inference_start = time.monotonic_ns()
                    raw_action = session.predict(state, images, snapshot.task)
                    action, clipping = prepare_safe_shadow_action(raw_action)
                    inference_ms = (time.monotonic_ns() - inference_start) / 1e6
                    completed_snapshot = task_state.snapshot()
                    if completed_snapshot.generation != snapshot.generation:
                        raise RuntimeError(
                            "task changed during policy inference; discarded old-task action"
                        )
                    record = {
                        "event": "prediction_clipped" if clipping else "prediction",
                        "step": step,
                        "wall_time_ns": time.time_ns(),
                        "task_state": asdict(snapshot),
                        "input": input_meta,
                        "state": state.tolist(),
                        "raw_action": raw_action.tolist(),
                        "action": action.tolist(),
                        "clipping": clipping,
                        "inference_ms": inference_ms,
                        "hardware_commands_sent": False,
                    }
                    status = (
                        " CLIPPED="
                        + ",".join(
                            f"{item['joint']}:{item['raw']:.4f}->{item['safe']:.4f}"
                            for item in clipping
                        )
                        if clipping
                        else ""
                    )
                    print(
                        f"step={step} phase={snapshot.phase}{status} "
                        f"inference_ms={inference_ms:.1f}",
                        flush=True,
                    )
                except Exception as exc:
                    record = {
                        "event": "prediction_rejected",
                        "step": step,
                        "wall_time_ns": time.time_ns(),
                        "task_state": asdict(snapshot),
                        "reason": str(exc),
                        "hardware_commands_sent": False,
                    }
                    print(f"step={step} REJECTED: {exc}", flush=True)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                step += 1
                stopped.wait(max(0.0, args.interval - (time.monotonic() - loop_start)))
    finally:
        event_server.stop()
        image_client.close()
        state_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
