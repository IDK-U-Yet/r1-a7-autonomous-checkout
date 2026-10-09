# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import copy

import numpy as np
import pytest

from unitree_lerobot.r1a7_contract import (
    CAMERA_KEYS,
    CAMERA_SHAPES,
    JOINT_NAMES,
    ROBOT_TYPE,
    R1A7ContractError,
    decode_action,
    encode_action,
    encode_observation_state,
    validate_action_vector,
    validate_dataset_info,
)


def _valid_info():
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [16],
            "names": [list(JOINT_NAMES)],
        },
        "action": {
            "dtype": "float32",
            "shape": [16],
            "names": [list(JOINT_NAMES)],
        },
    }
    for key in CAMERA_KEYS:
        features[key] = {
            "dtype": "video",
            "shape": list(CAMERA_SHAPES[key]),
            "names": ["height", "width", "channel"],
        }
    return {
        "codebase_version": "v3.0",
        "robot_type": ROBOT_TYPE,
        "features": features,
    }


def test_action_codec_preserves_r1a7_order():
    arms = np.arange(14, dtype=np.float64) / 100
    action = encode_action(arms, [1.25, 4.75])
    decoded = decode_action(action)

    np.testing.assert_array_equal(action[:14], arms)
    np.testing.assert_array_equal(action[14:], [1.25, 4.75])
    np.testing.assert_array_equal(decoded.left_arm, arms[:7])
    np.testing.assert_array_equal(decoded.right_arm, arms[7:])
    assert tuple(decoded.as_named_dict()) == JOINT_NAMES


def test_observation_state_uses_arm_gripper_order():
    state = encode_observation_state(np.zeros(14), [2.0, 3.0])
    np.testing.assert_array_equal(state[:14], np.zeros(14))
    np.testing.assert_array_equal(state[14:], [2.0, 3.0])


@pytest.mark.parametrize("bad", [np.zeros(15), np.zeros((1, 16))])
def test_action_rejects_wrong_shape(bad):
    with pytest.raises(R1A7ContractError, match="shape"):
        validate_action_vector(bad)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_action_rejects_non_finite_values(value):
    action = np.zeros(16)
    action[14] = value
    with pytest.raises(R1A7ContractError, match="non-finite"):
        validate_action_vector(action)


def test_action_rejects_out_of_range_grippers():
    action = np.zeros(16)
    action[14] = 5.5
    action[15] = -0.1
    with pytest.raises(R1A7ContractError, match="kLeftGripper.*kRightGripper"):
        validate_action_vector(action)


def test_dataset_metadata_strictly_validates_names_and_cameras():
    info = _valid_info()
    validate_dataset_info(info)

    wrong_order = copy.deepcopy(info)
    names = wrong_order["features"]["action"]["names"][0]
    names[14], names[15] = names[15], names[14]
    with pytest.raises(R1A7ContractError, match="wrong joint order"):
        validate_dataset_info(wrong_order)

    missing_camera = copy.deepcopy(info)
    del missing_camera["features"]["observation.images.cam_scanner"]
    with pytest.raises(R1A7ContractError, match="cam_scanner"):
        validate_dataset_info(missing_camera)


def test_model_contract_excludes_waist():
    assert len(JOINT_NAMES) == 16
    assert "kWaistYaw" not in JOINT_NAMES
