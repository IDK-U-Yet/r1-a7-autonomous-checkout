# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
from unitree_lerobot.utils.constants import (
    G1_DEX1_CONFIG,
    R1_A7_DEX1_4CAM_CONFIG,
    R1_A7_DEX1_CONFIG,
)


def test_r1_a7_dex1_model_uses_dual_arms_and_grippers_only():
    assert len(R1_A7_DEX1_CONFIG.motors) == 16
    assert R1_A7_DEX1_CONFIG.motors[14:] == [
        "kLeftGripper",
        "kRightGripper",
    ]
    assert R1_A7_DEX1_CONFIG.json_state_data_name == [
        "left_arm.qpos",
        "right_arm.qpos",
        "left_ee.qpos",
        "right_ee.qpos",
    ]
    assert R1_A7_DEX1_CONFIG.json_action_data_name == (
        R1_A7_DEX1_CONFIG.json_state_data_name
    )
    assert all("Head" not in name for name in R1_A7_DEX1_CONFIG.motors)
    assert "kWaistYaw" not in R1_A7_DEX1_CONFIG.motors


def test_g1_dex1_configuration_was_not_changed():
    assert len(G1_DEX1_CONFIG.motors) == 16
    assert "waist.qpos" not in G1_DEX1_CONFIG.json_state_data_name


def test_r1_a7_four_camera_configuration_adds_scanner_view():
    assert R1_A7_DEX1_4CAM_CONFIG.motors == R1_A7_DEX1_CONFIG.motors
    assert R1_A7_DEX1_4CAM_CONFIG.cameras == [
        "cam_global",
        "cam_left_wrist",
        "cam_right_wrist",
        "cam_scanner",
    ]
    assert R1_A7_DEX1_4CAM_CONFIG.camera_to_image_key["color_3"] == "cam_scanner"
    assert R1_A7_DEX1_4CAM_CONFIG.camera_shapes == {
        "cam_global": (360, 640, 3),
        "cam_left_wrist": (480, 640, 3),
        "cam_right_wrist": (480, 640, 3),
        "cam_scanner": (480, 640, 3),
    }
