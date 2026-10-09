# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import dataclasses


@dataclasses.dataclass(frozen=True)
class RobotConfig:
    motors: list[str]
    cameras: list[str]
    camera_to_image_key: dict[str, str]
    json_state_data_name: list[str]
    json_action_data_name: list[str]
    camera_shapes: dict[str, tuple[int, int, int]] = dataclasses.field(default_factory=dict)


Z1_CONFIG = RobotConfig(
    motors=[
        "kLeftWaist",
        "kLeftShoulder",
        "kLeftElbow",
        "kLeftForearmRoll",
        "kLeftWristAngle",
        "kLeftWristRotate",
        "kLeftGripper",
        "kRightWaist",
        "kRightShoulder",
        "kRightElbow",
        "kRightForearmRoll",
        "kRightWristAngle",
        "kRightWristRotate",
        "kRightGripper",
    ],
    cameras=[
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key={"color_0": "cam_high", "color_1": "cam_left_wrist", "color_2": "cam_right_wrist"},
    json_state_data_name=["left_arm.qpos", "right_arm.qpos"],
    json_action_data_name=["left_arm.qpos", "right_arm.qpos"],
)


Z1_SINGLE_CONFIG = RobotConfig(
    motors=[
        "kWaist",
        "kShoulder",
        "kElbow",
        "kForearmRoll",
        "kWristAngle",
        "kWristRotate",
        "kGripper",
    ],
    cameras=[
        "cam_high",
        "cam_wrist",
    ],
    camera_to_image_key={"color_0": "cam_high", "color_1": "cam_wrist"},
    json_state_data_name=["left_arm.qpos", "right_arm.qpos"],
    json_action_data_name=["left_arm.qpos", "right_arm.qpos"],
)


G1_DEX1_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftGripper",
        "kRightGripper",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key={
        "color_0": "cam_left_high",
        "color_1": "cam_right_high",
        "color_2": "cam_left_wrist",
        "color_3": "cam_right_wrist",
    },
    json_state_data_name=["left_arm.qpos", "right_arm.qpos", "left_ee.qpos", "right_ee.qpos"],
    json_action_data_name=["left_arm.qpos", "right_arm.qpos", "left_ee.qpos", "right_ee.qpos"],
)


G1_DEX1_CONFIG_SIM = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftGripper",
        "kRightGripper",
    ],
    cameras=[
        "cam_left_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key={
        "color_0": "cam_left_high",
        "color_1": "cam_left_wrist",
        "color_2": "cam_right_wrist",
    },
    json_state_data_name=["left_arm.qpos", "right_arm.qpos", "left_ee.qpos", "right_ee.qpos"],
    json_action_data_name=["left_arm.qpos", "right_arm.qpos", "left_ee.qpos", "right_ee.qpos"],
)


R1_A7_DEX1_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftGripper",
        "kRightGripper",
    ],
    cameras=[
        "cam_global",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key={
        "color_0": "cam_global",
        "color_1": "cam_left_wrist",
        "color_2": "cam_right_wrist",
    },
    json_state_data_name=[
        "left_arm.qpos",
        "right_arm.qpos",
        "left_ee.qpos",
        "right_ee.qpos",
    ],
    json_action_data_name=[
        "left_arm.qpos",
        "right_arm.qpos",
        "left_ee.qpos",
        "right_ee.qpos",
    ],
    camera_shapes={
        "cam_global": (1080, 1920, 3),
        "cam_left_wrist": (480, 640, 3),
        "cam_right_wrist": (480, 640, 3),
    },
)


R1_A7_DEX1_4CAM_CONFIG = RobotConfig(
    motors=list(R1_A7_DEX1_CONFIG.motors),
    cameras=[
        "cam_global",
        "cam_left_wrist",
        "cam_right_wrist",
        "cam_scanner",
    ],
    camera_to_image_key={
        "color_0": "cam_global",
        "color_1": "cam_left_wrist",
        "color_2": "cam_right_wrist",
        "color_3": "cam_scanner",
    },
    json_state_data_name=list(R1_A7_DEX1_CONFIG.json_state_data_name),
    json_action_data_name=list(R1_A7_DEX1_CONFIG.json_action_data_name),
    camera_shapes={
        "cam_global": (360, 640, 3),
        "cam_left_wrist": (480, 640, 3),
        "cam_right_wrist": (480, 640, 3),
        "cam_scanner": (480, 640, 3),
    },
)


G1_DEX3_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftHandThumb0",
        "kLeftHandThumb1",
        "kLeftHandThumb2",
        "kLeftHandMiddle0",
        "kLeftHandMiddle1",
        "kLeftHandIndex0",
        "kLeftHandIndex1",
        "kRightHandThumb0",
        "kRightHandThumb1",
        "kRightHandThumb2",
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key={
        "color_0": "cam_left_high",
        "color_1": "cam_right_high",
        "color_2": "cam_left_wrist",
        "color_3": "cam_right_wrist",
    },
    json_state_data_name=["left_arm.qpos", "right_arm.qpos", "left_ee.qpos", "right_ee.qpos"],
    json_action_data_name=["left_arm.qpos", "right_arm.qpos", "left_ee.qpos", "right_ee.qpos"],
)


G1_BRAINCO_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftHandThumb",
        "kLeftHandThumbAux",
        "kLeftHandIndex",
        "kLeftHandMiddle",
        "kLeftHandRing",
        "kLeftHandPinky",
        "kRightHandThumb",
        "kRightHandThumbAux",
        "kRightHandIndex",
        "kRightHandMiddle",
        "kRightHandRing",
        "kRightHandPinky",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key={
        "color_0": "cam_left_high",
        "color_1": "cam_right_high",
        "color_2": "cam_left_wrist",
        "color_3": "cam_right_wrist",
    },
    json_state_data_name=["left_arm.qpos", "right_arm.qpos", "left_ee.qpos", "right_ee.qpos"],
    json_action_data_name=["left_arm.qpos", "right_arm.qpos", "left_ee.qpos", "right_ee.qpos"],
)


G1_INSPIRE_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftHandPinky",
        "kLeftHandRing",
        "kLeftHandMiddle",
        "kLeftHandIndex",
        "kLeftHandThumbBend",
        "kLeftHandThumbRotation",
        "kRightHandPinky",
        "kRightHandRing",
        "kRightHandMiddle",
        "kRightHandIndex",
        "kRightHandThumbBend",
        "kRightHandThumbRotation",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key={
        "color_0": "cam_left_high",
        "color_1": "cam_right_high",
        "color_2": "cam_left_wrist",
        "color_3": "cam_right_wrist",
    },
    json_state_data_name=["left_arm.qpos", "right_arm.qpos", "left_ee.qpos", "right_ee.qpos"],
    json_action_data_name=["left_arm.qpos", "right_arm.qpos", "left_ee.qpos", "right_ee.qpos"],
)


MOVEIBLE_LIFT_G1_DEX1_USEWAIST_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kWaistYaw",
        "kWaistPitch",
        "kHighLift",
        "kMoveX",
        "kMoveYaw",
        "kLeftGripper",
        "kRightGripper",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key={
        "color_0": "cam_left_high",
        "color_1": "cam_right_high",
        "color_2": "cam_left_wrist",
        "color_3": "cam_right_wrist",
    },
    json_state_data_name=[
        "left_arm.qpos",
        "right_arm.qpos",
        "waist.qpos",
        "torso.height",
        "chassis.qvel",
        "left_ee.qpos",
        "right_ee.qpos",
    ],
    json_action_data_name=[
        "left_arm.qpos",
        "right_arm.qpos",
        "waist.qpos",
        "torso.qvel",
        "chassis.qvel",
        "left_ee.qpos",
        "right_ee.qpos",
    ],
)


MOVEIBLE_LIFT_G1_DEX1_NOUSEWAIST_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kHighLift",
        "kMoveX",
        "kMoveYaw",
        "kLeftGripper",
        "kRightGripper",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key={
        "color_0": "cam_left_high",
        "color_1": "cam_right_high",
        "color_2": "cam_left_wrist",
        "color_3": "cam_right_wrist",
    },
    json_state_data_name=[
        "left_arm.qpos",
        "right_arm.qpos",
        "torso.height",
        "chassis.qvel",
        "left_ee.qpos",
        "right_ee.qpos",
    ],
    json_action_data_name=[
        "left_arm.qpos",
        "right_arm.qpos",
        "torso.qvel",
        "chassis.qvel",
        "left_ee.qpos",
        "right_ee.qpos",
    ],
)


LIFT_G1_DEX1_USEWAIST_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kWaistYaw",
        "kWaistRoll",
        "kHighLift",
        "kLeftGripper",
        "kRightGripper",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key={
        "color_0": "cam_left_high",
        "color_1": "cam_right_high",
        "color_2": "cam_left_wrist",
        "color_3": "cam_right_wrist",
    },
    json_state_data_name=[
        "left_arm.qpos",
        "right_arm.qpos",
        "waist.qpos",
        "torso.height",
        "left_ee.qpos",
        "right_ee.qpos",
    ],
    json_action_data_name=[
        "left_arm.qpos",
        "right_arm.qpos",
        "waist.qpos",
        "torso.qvel",
        "left_ee.qpos",
        "right_ee.qpos",
    ],
)


LIFT_G1_DEX1_NOUSEWAIST_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kHighLift",
        "kLeftGripper",
        "kRightGripper",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key={
        "color_0": "cam_left_high",
        "color_1": "cam_right_high",
        "color_2": "cam_left_wrist",
        "color_3": "cam_right_wrist",
    },
    json_state_data_name=["left_arm.qpos", "right_arm.qpos", "torso.height", "left_ee.qpos", "right_ee.qpos"],
    json_action_data_name=["left_arm.qpos", "right_arm.qpos", "torso.qvel", "left_ee.qpos", "right_ee.qpos"],
)
UNIARML1_CONFIG = RobotConfig(
    motors=[
        "kShoulderPan",
        "kShoulderLift",
        "kElbowFlex",
        "kWristFlex",
        "kWristRoll",
        "kGripper",
    ],
    cameras=[
        "cam_top",
        "cam_wrist",
    ],
    camera_to_image_key={
        "head": "cam_top",
        "wrist": "cam_wrist",
    },
    json_state_data_name=["arm.qpos", "gripper.qpos"],
    json_action_data_name=["arm.qpos", "gripper.qpos"],
)

ROBOT_CONFIGS = {
    "Unitree_Z1_Single": Z1_SINGLE_CONFIG,
    "Unitree_Z1_Dual": Z1_CONFIG,
    "Unitree_G1_Dex1": G1_DEX1_CONFIG,
    "Unitree_G1_Dex1_Sim": G1_DEX1_CONFIG_SIM,
    "Unitree_R1_A7_Dex1": R1_A7_DEX1_CONFIG,
    "Unitree_R1_A7_Dex1_4Cam": R1_A7_DEX1_4CAM_CONFIG,
    "Unitree_G1_Dex3": G1_DEX3_CONFIG,
    "Unitree_G1_Brainco": G1_BRAINCO_CONFIG,
    "Unitree_G1_Inspire": G1_INSPIRE_CONFIG,
    "Unitree_G1_MoveibleLift_Dex1_UseWaist": MOVEIBLE_LIFT_G1_DEX1_USEWAIST_CONFIG,
    "Unitree_G1_MoveibleLift_Dex1_NoUseWaist": MOVEIBLE_LIFT_G1_DEX1_NOUSEWAIST_CONFIG,
    "Unitree_G1_Lift_Dex1_UseWaist": LIFT_G1_DEX1_USEWAIST_CONFIG,
    "Unitree_G1_Lift_Dex1_NoUseWaist": LIFT_G1_DEX1_NOUSEWAIST_CONFIG,
    "Unitree_Arm_L1": UNIARML1_CONFIG,
}
