# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import threading
import time
import numpy as np

from unitree_lerobot.r1a7_rtc_async import RTCController


class FakeSession:
    chunk_size = 50

    def __init__(self):
        self.calls = []
        self.release = threading.Event()
        self.release.set()

    def reset(self):
        pass

    def to_hardware_action(self, action):
        return action + 10

    def predict_normalized(self, state, images, task, **kwargs):
        self.calls.append((state, task, kwargs))
        assert self.release.wait(2)
        return np.ones((50, 16))


def wait_result(controller):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        result = controller.poll()
        if result is not None:
            return result
        time.sleep(.001)
    raise AssertionError("worker timed out")


def test_inflight_task_switch_discards_old_result():
    session = FakeSession()
    session.release.clear()
    controller = RTCController(session)
    try:
        controller.request_if_needed(np.zeros(16), {}, "scan")
        controller.invalidate()
        session.release.set()
        assert wait_result(controller)["event"] == "rtc_result_discarded"
        assert controller.pop() is None
        assert controller.request_if_needed(np.zeros(16), {}, "place")
        assert wait_result(controller)["event"] == "rtc_chunk_merged"
        controller.pop()
        np.testing.assert_allclose(controller.last_hardware_action, np.ones(16)*11)
    finally:
        session.release.set()
        controller.close()


def test_prefetch_receives_remaining_normalized_prefix():
    session = FakeSession()
    controller = RTCController(session)
    try:
        controller.request_if_needed(np.zeros(16), {}, "scan")
        wait_result(controller)
        for _ in range(5):
            controller.pop()
        assert controller.request_if_needed(np.zeros(16), {}, "scan")
        for _ in range(5):
            controller.pop()
        wait_result(controller)
        assert session.calls[-1][2]["prefix"].shape == (45, 16)
        assert session.calls[-1][2]["delay_steps"] >= 5
    finally:
        controller.close()


def test_fast_result_waits_for_committed_old_window():
    session = FakeSession()
    controller = RTCController(session)
    try:
        controller.request_if_needed(np.zeros(16), {}, "scan")
        wait_result(controller)
        for _ in range(5):
            controller.pop()
        controller.request_if_needed(np.zeros(16), {}, "scan")
        assert controller._future.result(timeout=2) is not None
        assert controller.poll() is None
        for _ in range(5):
            controller.pop()
        merged = controller.poll()
        assert merged["consumed_delay_steps"] == 5
        assert merged["reserved_delay_steps"] == 5
        assert merged["conditioned_delay_steps"] == merged["guidance_delay_steps"]
        assert merged["chunk_processing_ms"] == merged["inference_ms"]
        assert merged["model_latency_ms"] is None  # mock does not measure a model
        assert merged["guidance_delay_steps"] >= 5
    finally:
        controller.close()


def test_bootstrap_does_not_replace_initial_guidance_delay():
    controller = RTCController(FakeSession(), initial_delay_steps=7)
    try:
        controller.request_if_needed(np.zeros(16), {}, "scan")
        wait_result(controller)
        assert controller.delay_steps == 7
    finally:
        controller.close()
