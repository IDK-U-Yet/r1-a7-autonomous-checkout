# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from unitree_lerobot.r1a7_contract import CAMERA_KEYS
from unitree_lerobot.r1a7_rtc_session import RTCSession


def make_session():
    session = RTCSession.__new__(RTCSession)
    session.torch = torch
    session.device = torch.device("cpu")
    session.use_amp = False
    session.chunk_size = 50
    session.mask_mode = "configured"
    session.mask_endpoint = 10
    session.num_inference_steps = 10
    session._normalized_bounds = np.stack([np.full(16, -2), np.full(16, 2)])
    captured = {}

    def predict(batch, **kwargs):
        assert not torch.is_inference_mode_enabled()
        captured.update(kwargs)
        return torch.ones(1, 50, 16)

    session.policy = SimpleNamespace(predict_action_chunk=predict)
    session.preprocessor = lambda observation: observation
    session.postprocessor = lambda action: action*2+3
    return session, captured


def observation():
    return np.zeros(16), {key: np.zeros((3, 4, 4), dtype=np.uint8) for key in CAMERA_KEYS}


def test_normalized_prefix_not_denormalized_or_reencoded():
    session, captured = make_session()
    state, images = observation()
    prefix = np.full((45,16), .25, dtype=np.float32)
    with torch.inference_mode():
        result = session.predict_normalized(state, images, "scan", prefix=prefix, delay_steps=5)
    torch.testing.assert_close(captured["prev_chunk_left_over"], torch.full((1,45,16), .25))
    assert captured["inference_delay"] == 5
    assert captured["execution_horizon"] == 10
    assert captured["num_steps"] == 10
    assert session.last_model_latency_ms >= 0
    np.testing.assert_allclose(session.to_hardware_action(result[0]), np.ones(16)*5)
    np.testing.assert_allclose(prefix, np.full((45,16), .25))


def test_prefix_shorter_than_committed_delay_is_rejected():
    session, _ = make_session()
    state, images = observation()
    with pytest.raises(ValueError, match="committed delay"):
        session.predict_normalized(state, images, "scan", prefix=np.zeros((2,16)), delay_steps=5)


@pytest.mark.parametrize("steps", [7, 5])
def test_sampling_steps_override_reaches_policy_without_config_mutation(steps):
    session, captured = make_session()
    session.num_inference_steps = steps
    session.policy.config = SimpleNamespace(num_inference_steps=10)
    state, images = observation()
    session.predict_normalized(state, images, "scan", prefix=np.zeros((42, 16)), delay_steps=7)
    assert captured["num_steps"] == steps
    assert captured["inference_delay"] == 7
    assert session.policy.config.num_inference_steps == 10


@pytest.mark.parametrize("steps", [0, -1, 7.5, True])
def test_invalid_sampling_steps_rejected_before_loading_model(steps):
    with pytest.raises(ValueError, match="positive integer"):
        RTCSession(num_inference_steps=steps)


def test_target_projection_keeps_legal_predictions_and_source_unchanged():
    session, _ = make_session()
    source = np.full((50, 16), 3., dtype=np.float32)
    source[:, 15] = -3
    source[:, :5] = .5
    projected = session.project_targets(source)
    np.testing.assert_array_equal(projected[:, :5], source[:, :5])
    np.testing.assert_array_equal(projected[:, 5:14], np.full((50,9), 2))
    np.testing.assert_array_equal(projected[:, 14], np.full(50, 2))
    np.testing.assert_array_equal(projected[:, 15], np.full(50, -2))
    assert source[0, 14] == 3
    assert session.last_projection_summary["joint_counts"]["kLeftWristPitch"] == 50


def test_nonfinite_predictions_are_not_hidden_by_projection():
    session, _ = make_session()
    source = np.zeros((50,16)); source[0,5] = np.inf
    with pytest.raises(ValueError, match="finite"):
        session.project_targets(source)


@pytest.mark.parametrize("length", [47, 40, 36])
def test_overlap_mask_tracks_actual_remaining_prefix(length):
    session, captured = make_session()
    session.mask_mode = "overlap"
    state, images = observation()
    session.predict_normalized(state, images, "scan", prefix=np.zeros((length,16)), delay_steps=7)
    assert captured["execution_horizon"] == length
    assert session.last_mask_endpoint == length


def test_mask_does_not_constrain_bootstrap_without_prefix():
    session, captured = make_session()
    state, images = observation()
    session.predict_normalized(state, images, "scan")
    assert captured["execution_horizon"] is None
    assert session.last_mask_endpoint is None
