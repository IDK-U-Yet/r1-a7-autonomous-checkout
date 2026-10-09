# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
from __future__ import annotations

import time

import numpy as np

from unitree_lerobot.r1a7_async_policy import (
    AsyncActionChunkProducer,
    select_executable_window,
)


class FakeSession:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def predict_chunk(self, state, images, task):
        time.sleep(0.01)
        return np.tile(np.asarray(state, dtype=np.float64), (3, 1))


def test_select_executable_window_uses_time_aligned_candidate_indices():
    candidates = np.repeat(np.arange(50, dtype=np.float64)[:, None], 16, axis=1)

    selected = select_executable_window(
        candidates, lead_actions=5, n_action_steps=10
    )

    assert selected.shape == (10, 16)
    np.testing.assert_array_equal(selected[:, 0], np.arange(5, 15))


def test_select_executable_window_rejects_short_horizon():
    candidates = np.zeros((12, 16), dtype=np.float64)

    with np.testing.assert_raises_regex(ValueError, "shorter"):
        select_executable_window(candidates, lead_actions=5, n_action_steps=10)


def test_async_producer_returns_valid_chunk_and_resets_only_on_generation_change():
    session = FakeSession()
    producer = AsyncActionChunkProducer(session)
    try:
        state = np.arange(16, dtype=np.float64)
        assert producer.submit(
            generation=0, task="scan", lead_actions=0, state=state, images={}
        )
        result = producer.wait(1.0)
        assert result is not None and result.error is None
        np.testing.assert_array_equal(result.actions, np.tile(state, (3, 1)))
        assert session.reset_count == 1

        assert producer.submit(
            generation=0, task="scan", lead_actions=2, state=state, images={}
        )
        assert producer.wait(1.0).error is None
        assert session.reset_count == 1

        assert producer.submit(
            generation=1, task="place", lead_actions=0, state=state, images={}
        )
        changed = producer.wait(1.0)
        assert changed.generation == 1
        assert session.reset_count == 2
    finally:
        producer.close()


def test_new_generation_supersedes_stale_inflight_result():
    session = FakeSession()
    producer = AsyncActionChunkProducer(session)
    try:
        state = np.zeros(16)
        producer.submit(
            generation=0, task="scan", lead_actions=2, state=state, images={}
        )
        producer.submit(
            generation=1, task="place", lead_actions=0, state=state + 1, images={}
        )
        result = producer.wait(1.0)
        assert result is not None
        assert result.generation == 1
        np.testing.assert_array_equal(result.actions, np.ones((3, 16)))
    finally:
        producer.close()
