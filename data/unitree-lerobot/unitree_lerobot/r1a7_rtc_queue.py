# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Single-controller RTC action timeline; never shared directly with a worker.

Actions here are normalized model actions, not hardware commands. A worker gets
an immutable snapshot, and returns a chunk anchored to that snapshot's step.
The caller must invalidate on task changes, pauses, rejected targets or unexpected
bridge modifications. Legal arm/gripper projection occurs before caching this queue.
"""

from __future__ import annotations

import numpy as np


class RTCQueue:
    def __init__(self):
        self.generation = 0
        self.step = 0
        self._actions = np.empty((0, 16), dtype=np.float32)

    def snapshot(self) -> tuple[int, np.ndarray]:
        return self.step, self._actions.copy()

    def invalidate(self) -> int:
        self.generation += 1
        self.step = 0
        self._actions = np.empty((0, 16), dtype=np.float32)
        return self.generation

    def merge(self, actions, *, generation: int, observed_step: int) -> bool:
        if generation != self.generation:
            return False
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 16 or not np.isfinite(actions).all():
            raise ValueError("RTC result must be finite (N, 16) model actions")
        elapsed = self.step - observed_step
        if elapsed < 0:
            raise ValueError("RTC result is anchored in the future")
        if elapsed >= len(actions):
            return False
        self._actions = actions[elapsed:].copy()
        return True

    def pop(self) -> np.ndarray | None:
        if not len(self._actions):
            return None
        action = self._actions[0].copy()
        self._actions = self._actions[1:]
        self.step += 1
        return action

    def __len__(self) -> int:
        return len(self._actions)
