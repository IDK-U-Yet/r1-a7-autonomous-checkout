# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from unitree_lerobot.r1a7_contract import CAMERA_KEYS, CAMERA_SHAPES
from unitree_lerobot.r1a7_live_shadow import (
    StateBridgeClient,
    assemble_live_observation,
    prepare_safe_shadow_action,
)


def _frame(shape, timestamp, value=0):
    return SimpleNamespace(
        bgr=np.full(shape, value, dtype=np.uint8),
        capture_monotonic_ns=timestamp,
        receive_monotonic_ns=timestamp,
        frame_sequence=3,
        fps=15.0,
    )


def _inputs(now=1_000_000_000):
    state = {
        "state": np.arange(16, dtype=float).tolist(),
        "sample_monotonic_ns": {
            "lowstate": now - 10_000_000,
            "left_gripper": now - 11_000_000,
            "right_gripper": now - 12_000_000,
        },
    }
    frames = {
        key: _frame(CAMERA_SHAPES[key], now - index * 5_000_000, index + 1)
        for index, key in enumerate(CAMERA_KEYS)
    }
    return state, frames


def test_assemble_live_observation_preserves_state_and_converts_bgr_to_rgb_chw():
    now = 1_000_000_000
    state_reply, frames = _inputs(now)
    frames[CAMERA_KEYS[0]].bgr[0, 0] = [1, 2, 3]

    state, images, metadata = assemble_live_observation(
        state_reply,
        frames,
        now_monotonic_ns=now,
        max_state_age_ms=250,
        max_camera_age_ms=250,
        max_camera_skew_ms=100,
    )

    np.testing.assert_array_equal(state, np.arange(16, dtype=float))
    assert images[CAMERA_KEYS[0]].shape == (3, 360, 640)
    np.testing.assert_array_equal(images[CAMERA_KEYS[0]][:, 0, 0], [3, 2, 1])
    assert metadata["cross_camera_skew_ms"] == 15.0


def test_assemble_live_observation_rejects_stale_or_wrong_shape():
    now = 1_000_000_000
    state_reply, frames = _inputs(now)
    frames[CAMERA_KEYS[0]].capture_monotonic_ns = now - 300_000_000
    with pytest.raises(RuntimeError, match="stale camera"):
        assemble_live_observation(
            state_reply,
            frames,
            now_monotonic_ns=now,
            max_state_age_ms=250,
            max_camera_age_ms=250,
            max_camera_skew_ms=100,
        )

    _, frames = _inputs(now)
    frames[CAMERA_KEYS[-1]].bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        assemble_live_observation(
            state_reply,
            frames,
            now_monotonic_ns=now,
            max_state_age_ms=250,
            max_camera_age_ms=250,
            max_camera_skew_ms=100,
        )


def test_live_shadow_module_has_no_unitree_command_publisher():
    source = inspect.getsource(__import__("unitree_lerobot.r1a7_live_shadow", fromlist=["*"]))
    assert "ChannelPublisher" not in source
    assert "rt/lowcmd" not in source
    assert "rt/dex1/left/cmd" not in source
    assert "rt/dex1/right/cmd" not in source


def test_safe_shadow_action_clips_grippers_and_reports_raw_values():
    raw = np.zeros(16, dtype=np.float64)
    raw[14:] = [5.4198751, -0.02]

    safe, clipping = prepare_safe_shadow_action(raw)

    np.testing.assert_array_equal(safe[14:], [5.4, 0.0])
    assert clipping == [
        {
            "joint": "kLeftGripper",
            "raw": 5.4198751,
            "safe": 5.4,
            "delta": 5.4 - 5.4198751,
        },
        {
            "joint": "kRightGripper",
            "raw": -0.02,
            "safe": 0.0,
            "delta": 0.02,
        },
    ]


def test_safe_shadow_action_still_rejects_arm_limit_violation():
    raw = np.zeros(16, dtype=np.float64)
    raw[0] = 2.2
    raw[14] = 5.42

    with pytest.raises(ValueError, match="kLeftShoulderPitch"):
        prepare_safe_shadow_action(raw)


def test_state_bridge_timeout_recreates_req_socket(monkeypatch):
    class Socket:
        def setsockopt(self, *_args):
            pass

        def connect(self, *_args):
            pass

        def send_json(self, *_args):
            pass

        def recv_json(self):
            import zmq

            raise zmq.Again()

        def close(self, **_kwargs):
            pass

    sockets = []

    class Context:
        def socket(self, *_args):
            socket = Socket()
            sockets.append(socket)
            return socket

    monkeypatch.setattr("zmq.Context.instance", lambda: Context())
    client = StateBridgeClient("inproc://missing", 1)
    with pytest.raises(RuntimeError, match="timed out"):
        client.read()
    assert len(sockets) == 2
