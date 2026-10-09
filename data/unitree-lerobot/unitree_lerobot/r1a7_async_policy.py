# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Single-worker asynchronous action-chunk inference for live R1-A7 control."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


class ChunkSession(Protocol):
    def reset(self) -> None: ...

    def predict_chunk(
        self, state: np.ndarray, images: Mapping[str, np.ndarray], task: str
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ActionChunkResult:
    generation: int
    task: str
    lead_actions: int
    actions: np.ndarray | None
    inference_ms: float
    error: Exception | None = None


@dataclass(frozen=True)
class _ChunkRequest:
    generation: int
    task: str
    lead_actions: int
    state: np.ndarray
    images: dict[str, np.ndarray]


def select_executable_window(
    actions: np.ndarray, *, lead_actions: int, n_action_steps: int
) -> np.ndarray:
    """Select the time-aligned executable window from a full policy horizon."""

    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 16:
        raise ValueError(f"policy chunk must have shape (N, 16), got {array.shape}")
    if lead_actions < 0:
        raise ValueError("lead_actions must be non-negative")
    if n_action_steps <= 0:
        raise ValueError("n_action_steps must be positive")
    end = lead_actions + n_action_steps
    if end > len(array):
        raise ValueError(
            "policy chunk is shorter than its scheduled execution window: "
            f"need actions[{lead_actions}:{end}], got {len(array)} candidates"
        )
    return array[lead_actions:end].copy()


class AsyncActionChunkProducer:
    """Own a policy session in one worker and retain only the newest request."""

    def __init__(self, session: ChunkSession):
        self._session = session
        self._condition = threading.Condition()
        self._pending: _ChunkRequest | None = None
        self._result: ActionChunkResult | None = None
        self._inflight = False
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="r1a7-pi05-chunk-producer",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        *,
        generation: int,
        task: str,
        lead_actions: int,
        state: np.ndarray,
        images: Mapping[str, np.ndarray],
    ) -> bool:
        """Queue an immutable observation unless this generation is already queued."""

        with self._condition:
            if self._closed:
                raise RuntimeError("asynchronous action-chunk producer is closed")
            if lead_actions < 0:
                raise ValueError("lead_actions must be non-negative")
            if self._pending is not None and self._pending.generation == generation:
                return False
            self._pending = _ChunkRequest(
                generation=int(generation),
                task=str(task),
                lead_actions=int(lead_actions),
                state=np.asarray(state).copy(),
                images={key: np.asarray(value).copy() for key, value in images.items()},
            )
            self._condition.notify_all()
            return True

    def poll(self) -> ActionChunkResult | None:
        with self._condition:
            result = self._result
            self._result = None
            return result

    def wait(self, timeout: float | None = None) -> ActionChunkResult | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._result is None and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            result = self._result
            self._result = None
            return result

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        self._worker.join(timeout=5.0)

    def _run(self) -> None:
        active_generation: int | None = None
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                request = self._pending
                self._pending = None
                self._inflight = True

            started_ns = time.monotonic_ns()
            try:
                if request.generation != active_generation:
                    self._session.reset()
                    active_generation = request.generation
                actions = np.asarray(
                    self._session.predict_chunk(
                        request.state, request.images, request.task
                    ),
                    dtype=np.float64,
                )
                if actions.ndim != 2 or actions.shape[1] != 16 or len(actions) == 0:
                    raise ValueError(
                        f"policy chunk must have shape (N, 16), got {actions.shape}"
                    )
                if not np.all(np.isfinite(actions)):
                    raise ValueError("policy chunk contains non-finite actions")
                result = ActionChunkResult(
                    generation=request.generation,
                    task=request.task,
                    lead_actions=request.lead_actions,
                    actions=actions,
                    inference_ms=(time.monotonic_ns() - started_ns) / 1e6,
                )
            except Exception as exc:
                result = ActionChunkResult(
                    generation=request.generation,
                    task=request.task,
                    lead_actions=request.lead_actions,
                    actions=None,
                    inference_ms=(time.monotonic_ns() - started_ns) / 1e6,
                    error=exc,
                )

            with self._condition:
                self._inflight = False
                # A task switch may have queued a newer generation while inference ran.
                if self._pending is None or self._pending.generation == request.generation:
                    self._result = result
                self._condition.notify_all()
