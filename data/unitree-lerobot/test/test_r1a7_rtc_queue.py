# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import numpy as np
import pytest

from unitree_lerobot.r1a7_rtc_queue import RTCQueue


def chunk(n=50):
    return np.repeat(np.arange(n)[:, None], 16, axis=1).astype(float)


def test_result_aligned_to_consumed_steps_not_prefetch_constant():
    q = RTCQueue()
    assert q.merge(chunk(), generation=0, observed_step=0)
    for _ in range(3):
        q.pop()
    assert q.merge(chunk() + 100, generation=0, observed_step=1)
    assert q.pop()[0] == 102


def test_pause_invalidates_inflight_and_returns_hold():
    q = RTCQueue()
    q.merge(chunk(), generation=0, observed_step=0)
    q.invalidate()
    assert not q.merge(chunk(), generation=0, observed_step=0)
    assert q.pop() is None


def test_snapshot_is_independent_and_empty_queue_does_not_advance():
    q = RTCQueue()
    assert q.pop() is None
    assert q.step == 0
    q.merge(chunk(), generation=0, observed_step=0)
    step, prefix = q.snapshot()
    prefix[:] = -1
    assert step == 0 and q.pop()[0] == 0


def test_expired_horizon_cannot_execute_old_actions():
    q = RTCQueue()
    q.merge(chunk(), generation=0, observed_step=0)
    for _ in range(5):
        q.pop()
    assert not q.merge(chunk(3), generation=0, observed_step=0)


def test_bad_result_is_rejected():
    q = RTCQueue()
    with pytest.raises(ValueError):
        q.merge(np.full((50, 16), np.nan), generation=0, observed_step=0)
