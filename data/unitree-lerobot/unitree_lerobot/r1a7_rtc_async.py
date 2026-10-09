# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""RTC worker and controller-owned timeline, independent of hardware publishers."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time

import numpy as np

from unitree_lerobot.r1a7_rtc_queue import RTCQueue


@dataclass(frozen=True)
class RTCResult:
    generation: int
    observed_step: int
    actions: np.ndarray
    inference_ms: float
    guidance_delay_steps: int
    hardware_actions: np.ndarray
    prefix_overlap_steps: int = 0
    target_projection: dict | None = None
    mask_endpoint: int | None = None
    model_latency_ms: float | None = None
    reserved_delay_steps: int = 0


class RTCController:
    """One worker; no Torch calls on the control thread while a job is running.

    Call request before consuming this cycle's action. Step counts represent
    successful target dispatches, not elapsed HOLD ticks. Invalidate after any
    rejected/unexpectedly modified dispatch, task switch, pause, or fault.
    """

    def __init__(self, session, *, interval=1 / 15, execution_steps=10,
                 prefetch_steps=5, initial_delay_steps=5):
        if (not np.isfinite(interval) or interval <= 0
                or not 0 < prefetch_steps < execution_steps < session.chunk_size
                or not 0 <= initial_delay_steps < session.chunk_size):
            raise ValueError("invalid RTC interval/execution/prefetch configuration")
        self.session = session
        self.interval = interval
        self.execution_steps = execution_steps
        self.prefetch_steps = prefetch_steps
        self.delay_steps = initial_delay_steps
        self.timeline = RTCQueue()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="r1a7-rtc")
        self._future = None
        self._required_merge_step = 0
        self._requested_generation = 0
        self._reset_generation = None
        self._since_merge = 0
        self.closed = False
        self._hardware_actions = np.empty((0, 16))
        self.last_hardware_action = None

    def invalidate(self):
        self._since_merge = 0
        self._hardware_actions = np.empty((0, 16))
        self.last_hardware_action = None
        return self.timeline.invalidate()

    def request_if_needed(self, state, images, task):
        if self.closed:
            raise RuntimeError("RTC controller is closed")
        if self._future is not None:
            return False
        if len(self.timeline) and self._since_merge < self.execution_steps - self.prefetch_steps:
            return False
        step, prefix = self.timeline.snapshot()
        generation = self.timeline.generation
        # Commit no more steps than are available in the old horizon.
        # Preserve the committed old window even when inference finishes early.
        committed = max(0, self.execution_steps - self._since_merge) if len(prefix) else 0
        delay = min(max(self.delay_steps, committed), len(prefix)) if len(prefix) else 0
        self._required_merge_step = step + committed
        self._requested_generation = generation
        immutable_state = np.asarray(state).copy()
        immutable_images = {k: np.asarray(v).copy() for k, v in images.items()}

        def work():
            if generation != self._reset_generation:
                self.session.reset()
                self._reset_generation = generation
            started = time.monotonic_ns()
            actions = self.session.predict_normalized(
                immutable_state, immutable_images, task,
                prefix=prefix if len(prefix) else None, delay_steps=delay,
            )
            hardware = np.stack([self.session.to_hardware_action(a) for a in actions])
            if hardware.shape != actions.shape or not np.isfinite(hardware).all():
                raise ValueError("RTC postprocessor returned invalid actions")
            return RTCResult(generation, step, actions,
                             (time.monotonic_ns() - started) / 1e6, delay, hardware,
                             len(prefix), getattr(self.session, "last_projection_summary", None),
                             getattr(self.session, "last_mask_endpoint", None),
                             getattr(self.session, "last_model_latency_ms", None), committed)

        self._future = self._pool.submit(work)
        return True

    def poll(self):
        if self._future is None or not self._future.done():
            return None
        if (self._requested_generation == self.timeline.generation
                and self.timeline.step < self._required_merge_step and len(self.timeline)):
            return None
        future, self._future = self._future, None
        result = future.result()  # genuine worker failures must reach caller
        if result.generation != self.timeline.generation:
            return {"event": "rtc_result_discarded", "reason": "generation_changed"}
        consumed = self.timeline.step - result.observed_step
        accepted = self.timeline.merge(
            result.actions, generation=result.generation, observed_step=result.observed_step,
        )
        # Bootstrap has no committed prefix and includes first-use CUDA costs;
        # it must not inflate the next chunk's hard guidance prefix.
        if result.guidance_delay_steps:
            self.delay_steps = max(1, math.ceil(result.inference_ms / (self.interval * 1000)))
        if self.delay_steps >= self.session.chunk_size:
            self.invalidate()
            raise RuntimeError("RTC inference delay exceeds the entire model horizon")
        if accepted:
            self._since_merge = 0
            self._hardware_actions = result.hardware_actions[consumed:].copy()
        return {"event": "rtc_chunk_merged" if accepted else "rtc_result_discarded",
                "inference_ms": result.inference_ms, "consumed_delay_steps": consumed,
                "chunk_processing_ms": result.inference_ms,
                "model_latency_ms": result.model_latency_ms,
                "ceil_model_delay_steps": (math.ceil(result.model_latency_ms / (self.interval * 1000))
                                           if result.model_latency_ms is not None else None),
                "ceil_processing_delay_steps": math.ceil(result.inference_ms / (self.interval * 1000)),
                "conditioned_delay_steps": result.guidance_delay_steps,
                "reserved_delay_steps": result.reserved_delay_steps,
                "guidance_delay_steps": result.guidance_delay_steps,
                "prefix_overlap_steps": result.prefix_overlap_steps,
                "target_projection": result.target_projection,
                "mask_endpoint": result.mask_endpoint,
                "queue_length": len(self.timeline)}

    def pop(self):
        action = self.timeline.pop()
        if action is not None:
            self._since_merge += 1
            self.last_hardware_action = self._hardware_actions[0].copy()
            self._hardware_actions = self._hardware_actions[1:]
        else:
            self.last_hardware_action = None
        return action

    def close(self):
        self.closed = True
        self.invalidate()
        self._pool.shutdown(wait=True, cancel_futures=True)
