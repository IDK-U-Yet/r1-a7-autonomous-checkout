# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import numpy as np
import logging_mp
try:
    logging_mp.basicConfig(level=logging_mp.INFO)
except RuntimeError:
    # Test runners and embedded launchers may configure logging before import.
    pass
logger_mp = logging_mp.getLogger(__name__)

import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
repo_dir = current_dir.parent
system_root = Path(
    os.environ.get("R1A7_ROOT", repo_dir.parents[1])
).expanduser().resolve()
sdk2_python_dir = Path(
    os.environ.get(
        "UNITREE_SDK2_PYTHON_PATH",
        system_root / "vendor" / "unitree-sdk2-python",
    )
).expanduser().resolve()
for local_path in (
    current_dir / "televuer" / "src",
    current_dir / "teleimager" / "src",
    current_dir / "robot_control" / "dex-retargeting" / "src",
    sdk2_python_dir,
    repo_dir,
):
    local_path = str(local_path)
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_29_Arm_Internal_Dex1_Controller, G1_23_ArmController, H1_2_ArmController, H1_ArmController, H2_ArmController, R1_A5_ArmController, R1_A7_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK, R1_A5_ArmIK, R1_A7_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.collection_batch import CollectionBatch, CollectionBatchError
from teleop.utils.ipc import IPC_Server
from teleop.utils.task_events import TaskEventServer
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from teleop.utils.network import resolve_dds_interface
from teleop.robot_control.safety import (
    ActivationPoseEnvelope,
    ControllerButtonEdges,
    TeleopState,
    XRWatchdog,
    apply_enabled_arms,
)
from teleop.utils.diagnostics import DiagnosticsLogger
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
RECORD_REJECT  = False  # Abort the active episode or reject the last saved episode
HOME_REQUESTED = False
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.


def dex1_service_environment(binary, environ=None):
    env = dict(os.environ if environ is None else environ)
    library_dir = Path(binary).resolve().parent.parent / "lib"
    if not library_dir.is_dir():
        return env

    existing_paths = [
        path
        for path in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if path
    ]
    library_path = str(library_dir)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [library_path, *(path for path in existing_paths if path != library_path)]
    )
    return env


def start_dex1_service(binary, network_interface):
    binary = os.path.abspath(binary)
    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"Dex1 service binary is not executable: {binary}")
    process_env = dex1_service_environment(binary)
    direct_command = [binary, "--network", network_interface]
    serial_ports = [f"/dev/ttyUSB{index}" for index in range(4)]
    existing_ports = [port for port in serial_ports if os.path.exists(port)]
    has_serial_access = bool(existing_ports) and all(
        os.access(port, os.R_OK | os.W_OK) for port in existing_ports
    )
    if has_serial_access:
        command = direct_command
    else:
        command_text = "exec " + " ".join(
            shlex.quote(value) for value in direct_command
        )
        command = ["sg", "dialout", "-c", command_text]
    process = subprocess.Popen(command, start_new_session=True, env=process_env)
    time.sleep(0.4)
    if process.poll() is not None:
        raise RuntimeError(
            f"Dex1 service exited during startup with code {process.returncode}"
        )
    return process


def stop_dex1_service(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=1.0)


def camera_frame_metadata(image, record_monotonic_ns):
    if image is None:
        return None
    capture_monotonic_ns = image.capture_monotonic_ns
    receive_monotonic_ns = image.receive_monotonic_ns
    return {
        "frame_sequence": image.frame_sequence,
        "capture_monotonic_ns": capture_monotonic_ns,
        "capture_wall_time_ns": image.capture_wall_time_ns,
        "receive_monotonic_ns": receive_monotonic_ns,
        "sample_age_ms": (
            max(0, record_monotonic_ns - capture_monotonic_ns) / 1_000_000.0
            if capture_monotonic_ns is not None else None
        ),
        "transport_age_ms": (
            max(0, receive_monotonic_ns - capture_monotonic_ns) / 1_000_000.0
            if capture_monotonic_ns is not None and receive_monotonic_ns is not None
            else None
        ),
    }

def on_press(key):
    global STOP, START, RECORD_TOGGLE, RECORD_REJECT, HOME_REQUESTED
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 'h':
        START = False
        HOME_REQUESTED = True
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    elif key == 'y':
        RECORD_REJECT = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }


class PeriodicRateGate:
    """Select loop iterations at a fixed rate without slowing the control loop."""

    def __init__(self, frequency, polling_frequency=None):
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency must be positive and finite")
        self.period_ns = int(round(1_000_000_000 / frequency))
        self.tolerance_ns = 0
        if polling_frequency is not None:
            if not np.isfinite(polling_frequency) or polling_frequency <= 0:
                raise ValueError("polling_frequency must be positive and finite")
            self.tolerance_ns = int(round(500_000_000 / polling_frequency))
        self.next_sample_ns = None

    def reset(self):
        self.next_sample_ns = None

    def due(self, now_ns):
        if self.next_sample_ns is None:
            self.next_sample_ns = now_ns + self.period_ns
            return True
        if now_ns + self.tolerance_ns < self.next_sample_ns:
            return False
        if now_ns < self.next_sample_ns:
            self.next_sample_ns += self.period_ns
        else:
            elapsed_periods = (now_ns - self.next_sample_ns) // self.period_ns + 1
            self.next_sample_ns += elapsed_periods * self.period_ns
        return True

def build_arg_parser():
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type=float, default=30.0, help='robot control-loop frequency')
    parser.add_argument(
        '--record-frequency',
        type=float,
        help='dataset sampling frequency; defaults to the control-loop frequency',
    )
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2', 'R1_A5', 'R1_A7'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex1_internal', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    # network parameters
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default='enx207bd51abe1f', help='Network interface for DDS communication; renamed USB NICs are recovered by 192.168.123.222.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--no-camera', action='store_true', help='Disable all image clients and force pass-through display')
    parser.add_argument('--enabled-arms', choices=['left', 'right', 'both'], default='left', help='Arm sides allowed to track XR targets')
    parser.add_argument('--confirm-real-robot', action='store_true', help='Confirm clear area and physical emergency stop before real R1 control')
    parser.add_argument('--diagnostics-path', default='./logs/r1_a7_teleop.jsonl', help='JSONL safety diagnostics path')
    parser.add_argument('--dex1-service-mode', choices=['auto', 'external'], default='auto', help='Start the local Dex1 serial-to-DDS service or use an existing service')
    parser.add_argument(
        '--dex1-service-bin',
        default=str(system_root / 'robot' / 'dex1-service' / 'bin' / 'dex1_1_gripper_server'),
        help='Path to the Dex1 serial-to-DDS service binary',
    )
    parser.add_argument('--dex1-kp', type=float, default=5.0, help='Dex1 position stiffness (default: 5.0, matching the Unitree Dex1 test program)')
    parser.add_argument('--dex1-kd', type=float, default=0.05, help='Dex1 velocity damping (default: 0.05)')
    parser.add_argument('--dex1-command-step', type=float, default=0.36, help='Maximum Dex1 target error relative to measured position in rad (default: 0.36)')
    parser.add_argument('--dex1-soft-torque-limit', type=float, help='Calibrated absolute Dex1 tau_est soft limit; must be paired with the hard limit')
    parser.add_argument('--dex1-hard-torque-limit', type=float, help='Calibrated absolute Dex1 tau_est hard-release limit; must be greater than the soft limit')
    parser.add_argument('--dex1-contact-samples', type=int, default=3, help='Consecutive samples required to enter or leave the Dex1 soft limit (default: 3)')
    parser.add_argument('--dex1-torque-hysteresis-ratio', type=float, default=0.8, help='Soft-limit release threshold as a fraction of the soft limit (default: 0.8)')
    parser.add_argument('--dex1-release-step', type=float, default=0.018, help='Dex1 opening step used for hard-limit release in rad (default: 0.018)')
    parser.add_argument('--dex1-state-timeout', type=float, default=0.1, help='Maximum Dex1 state age before closing is blocked in seconds (default: 0.1)')
    parser.add_argument('--r1-control-profile', choices=['tuned', 'official'], default='tuned', help='R1-A7 motion limits: tuned defaults or fixed upstream parameters')
    parser.add_argument('--r1-arm-velocity-limit', type=float, help='Tuned profile per-joint target velocity limit in rad/s (default: 0.8)')
    parser.add_argument('--r1-joint-margin-deg', type=float, help='Tuned profile margin inside official joint limits in degrees (default: 5)')
    parser.add_argument('--r1-workspace-horizontal-limit', type=float, help='Tuned profile X/Y displacement limit from activation in metres (default: 0.15)')
    parser.add_argument('--r1-workspace-down-limit', type=float, help='Tuned profile downward displacement limit from activation in metres (default: 0.15)')
    parser.add_argument('--r1-workspace-up-limit', type=float, help='Tuned profile upward displacement limit from activation in metres (default: 0.30)')
    parser.add_argument('--r1-workspace-rotation-limit-deg', type=float, help='Tuned profile rotation limit from activation in degrees (default: 35)')
    parser.add_argument('--enable-r1-waist-control', action='store_true', help='Opt in to right-thumbstick R1-A7 waist yaw control; disabled by default')
    parser.add_argument('--r1-waist-velocity-limit', type=float, help='Right-thumbstick waist yaw speed in rad/s (default: 1.0, URDF hard maximum: 18.7)')
    parser.add_argument('--r1-waist-thumbstick-deadzone', type=float, help='Right-thumbstick horizontal deadzone for waist yaw (default: 0.10)')
    parser.add_argument('--confirm-official-parameters', action='store_true', help='Acknowledge that the official R1-A7 profile removes the added workspace/margin limits and uses 30 rad/s')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument(
        '--joint-test-record',
        action='store_true',
        help=(
            'Record an R1-A7 joint-only precision-test episode. This implies '
            '--record and --no-camera; omit --ee to exclude the grippers.'
        ),
    )
    parser.add_argument(
        '--task-dir',
        type=str,
        default=str(system_root / 'runtime' / 'datasets' / 'teleoperation'),
        help='path to save data',
    )
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')
    parser.add_argument('--task-event-bind', help='ZMQ endpoint for item_accepted events, for example tcp://0.0.0.0:55580')
    parser.add_argument('--task-event-token', default=os.environ.get('R1_TASK_EVENT_TOKEN'), help='Optional shared token (or R1_TASK_EVENT_TOKEN) for task events')
    parser.add_argument(
        '--task-before-accept',
        default=(
            'Pick up a product from the blue tray on the left, present it to the '
            'scanner in the scanning area, and wait until the scan is accepted.'
        ),
    )
    parser.add_argument(
        '--task-after-accept',
        default='Place the scanned product in the black tray on the right.',
    )
    parser.add_argument(
        '--collection-plan',
        type=Path,
        help='JSON plan for persistent multi-product batch collection and resume',
    )
    parser.add_argument(
        '--collection-reset-seconds',
        type=float,
        help='Override the collection plan reset interval; X/s may start early',
    )
    return parser


def validate_args(parser, args):
    if not np.isfinite(args.frequency) or args.frequency <= 0:
        parser.error("--frequency must be positive and finite.")
    if args.record_frequency is not None:
        if not np.isfinite(args.record_frequency) or args.record_frequency <= 0:
            parser.error("--record-frequency must be positive and finite.")
        if args.record_frequency > args.frequency:
            parser.error("--record-frequency cannot exceed --frequency.")
    if args.task_event_bind and not args.record:
        parser.error("--task-event-bind requires --record.")
    if args.task_event_bind:
        if not args.task_before_accept.strip() or not args.task_after_accept.strip():
            parser.error("task labels used by --task-event-bind cannot be empty.")
        if args.task_before_accept.strip() == args.task_after_accept.strip():
            parser.error("before/after task labels must be different.")
    if args.collection_plan:
        if not args.record or not args.task_event_bind:
            parser.error("--collection-plan requires --record and --task-event-bind.")
        if args.collection_reset_seconds is not None and (
            not np.isfinite(args.collection_reset_seconds)
            or args.collection_reset_seconds < 0
        ):
            parser.error("--collection-reset-seconds must be finite and non-negative.")
    if args.joint_test_record:
        if args.arm != "R1_A7":
            parser.error("--joint-test-record requires --arm R1_A7.")
        args.record = True
        args.no_camera = True
    if args.ee == "dex1_internal" and args.motion:
        parser.error("--ee dex1_internal does not currently support --motion.")
    dex1_positive_parameters = {
        "dex1_kp": args.dex1_kp,
        "dex1_command_step": args.dex1_command_step,
        "dex1_release_step": args.dex1_release_step,
        "dex1_state_timeout": args.dex1_state_timeout,
    }
    for name, value in dex1_positive_parameters.items():
        if not np.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not np.isfinite(args.dex1_kd) or args.dex1_kd < 0.0:
        parser.error("--dex1-kd must be non-negative and finite")
    if args.dex1_contact_samples < 1:
        parser.error("--dex1-contact-samples must be at least 1")
    if (
        not np.isfinite(args.dex1_torque_hysteresis_ratio)
        or not 0.0 <= args.dex1_torque_hysteresis_ratio < 1.0
    ):
        parser.error("--dex1-torque-hysteresis-ratio must be finite and in [0, 1)")
    torque_limits = (
        args.dex1_soft_torque_limit,
        args.dex1_hard_torque_limit,
    )
    if (torque_limits[0] is None) != (torque_limits[1] is None):
        parser.error(
            "--dex1-soft-torque-limit and --dex1-hard-torque-limit must be set together"
        )
    if torque_limits[0] is not None:
        if args.ee != "dex1":
            parser.error("Dex1 torque limits require --ee dex1 (external Dex1 service)")
        if (
            not all(np.isfinite(value) for value in torque_limits)
            or torque_limits[0] <= 0.0
            or torque_limits[1] <= torque_limits[0]
        ):
            parser.error(
                "Dex1 torque limits must be positive and finite with hard > soft"
            )
    if args.arm == "R1_A7" and args.motion:
        parser.error("R1_A7 does not support --motion; use debug low-level control.")
    if args.no_camera and args.record and not args.joint_test_record:
        parser.error(
            "--no-camera can only be combined with recording through "
            "--joint-test-record."
        )
    if args.no_camera:
        args.display_mode = "pass-through"
    if args.arm == "R1_A7" and not args.sim and not args.confirm_real_robot:
        parser.error(
            "real R1_A7 control requires --confirm-real-robot after checking the "
            "clear area, physical emergency stop, and network interface"
        )
    tuned_overrides = (
        args.r1_arm_velocity_limit,
        args.r1_joint_margin_deg,
        args.r1_workspace_horizontal_limit,
        args.r1_workspace_down_limit,
        args.r1_workspace_up_limit,
        args.r1_workspace_rotation_limit_deg,
    )
    if args.arm != "R1_A7" and (
        args.r1_control_profile != "tuned"
        or args.confirm_official_parameters
        or any(value is not None for value in tuned_overrides)
    ):
        parser.error("R1 control profile options require --arm R1_A7.")
    if args.arm != "R1_A7" and (
        args.enable_r1_waist_control
        or args.r1_waist_velocity_limit is not None
        or args.r1_waist_thumbstick_deadzone is not None
    ):
        parser.error("R1 waist control options require --arm R1_A7.")
    if args.arm == "R1_A7":
        if not args.enable_r1_waist_control and (
            args.r1_waist_velocity_limit is not None
            or args.r1_waist_thumbstick_deadzone is not None
        ):
            parser.error(
                "R1 waist velocity/deadzone options require "
                "--enable-r1-waist-control"
            )
        if args.r1_waist_velocity_limit is None:
            args.r1_waist_velocity_limit = 1.0
        if args.r1_waist_thumbstick_deadzone is None:
            args.r1_waist_thumbstick_deadzone = 0.10
        if (
            not np.isfinite(args.r1_waist_velocity_limit)
            or not 0.0 < args.r1_waist_velocity_limit <= 18.7
        ):
            parser.error(
                "--r1-waist-velocity-limit must be positive, finite, and no greater "
                "than the URDF limit 18.7 rad/s"
            )
        if (
            not np.isfinite(args.r1_waist_thumbstick_deadzone)
            or not 0.0 <= args.r1_waist_thumbstick_deadzone < 1.0
        ):
            parser.error(
                "--r1-waist-thumbstick-deadzone must be finite and in [0, 1)"
            )
    if args.arm == "R1_A7" and args.r1_control_profile == "official":
        if any(value is not None for value in tuned_overrides):
            parser.error(
                "the official profile is fixed; R1 limit overrides require "
                "--r1-control-profile tuned"
            )
        if not args.sim and not args.confirm_official_parameters:
            parser.error(
                "real-robot official profile requires --confirm-official-parameters; "
                "it has no added workspace envelope or 5 degree margin and uses the "
                "upstream 30 rad/s target limit"
            )
    if args.arm == "R1_A7" and args.r1_control_profile == "tuned":
        if args.confirm_official_parameters:
            parser.error(
                "--confirm-official-parameters requires --r1-control-profile official"
            )
        defaults = {
            "r1_arm_velocity_limit": 0.8,
            "r1_joint_margin_deg": 5.0,
            "r1_workspace_horizontal_limit": 0.15,
            "r1_workspace_down_limit": 0.15,
            "r1_workspace_up_limit": 0.30,
            "r1_workspace_rotation_limit_deg": 35.0,
        }
        for name, default in defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, default)
            value = getattr(args, name)
            if not np.isfinite(value) or (
                value < 0.0 if name == "r1_joint_margin_deg" else value <= 0.0
            ):
                qualifier = "non-negative" if name == "r1_joint_margin_deg" else "positive"
                parser.error(
                    f"--{name.replace('_', '-')} must be {qualifier} and finite"
                )
    return args


def camera_config_without_images():
    disabled = {"enable_zmq": False}
    return {
        "head_camera": {
            **disabled,
            "enable_webrtc": False,
            "binocular": False,
            "image_shape": (480, 640),
            "webrtc_port": 0,
        },
        "left_wrist_camera": disabled.copy(),
        "right_wrist_camera": disabled.copy(),
        "scanner_camera": disabled.copy(),
    }


def camera_recording_schema(camera_config):
    if camera_config["head_camera"]["binocular"]:
        return None

    schema = {}
    definitions = (
        ("global", "head_camera", "color_0"),
        ("left_wrist", "left_wrist_camera", "color_1"),
        ("right_wrist", "right_wrist_camera", "color_2"),
        ("scanner", "scanner_camera", "color_3"),
    )
    for name, config_key, color_key in definitions:
        config = camera_config.get(config_key, {})
        if not config.get("enable_zmq", False):
            continue
        height, width = config["image_shape"]
        schema[name] = {
            "color_key": color_key,
            "shape": [int(height), int(width), 3],
            "config_key": config_key,
        }
    return schema


if __name__ == '__main__':
    parser = build_arg_parser()

    args = parser.parse_args()
    args = validate_args(parser, args)
    record_frequency = (
        args.frequency if args.record_frequency is None else args.record_frequency
    )
    recording_rate_gate = PeriodicRateGate(record_frequency, args.frequency)
    logged_args = vars(args).copy()
    if logged_args.get("task_event_token"):
        logged_args["task_event_token"] = "<redacted>"
    logger_mp.debug(f"args: {logged_args}")

    collection_batch = None
    if args.collection_plan:
        try:
            collection_batch = CollectionBatch(
                os.path.join(args.task_dir, args.task_name),
                args.collection_plan,
                reset_seconds=args.collection_reset_seconds,
            )
        except (CollectionBatchError, OSError, ValueError) as exc:
            parser.error(f"cannot initialize collection batch: {exc}")
        summary = collection_batch.progress_summary()
        logger_mp.info(
            "Collection batch %s %s (selection=%s): %d/%d accepted, %d rejected.",
            summary["batch_id"],
            "resumed" if collection_batch.resumed else "created",
            summary["selection_mode"],
            summary["accepted"],
            summary["target"],
            summary["rejected"],
        )

    arm_ctrl = None
    img_client = None
    tv_wrapper = None
    diagnostics = None
    gripper_ctrl = None
    dex1_service_process = None
    fault_reason = None
    ipc_server = None
    task_event_server = None
    listen_keyboard_thread = None
    sim_state_subscriber = None
    recorder = None
    batch_last_reset_announcement = None
    exit_code = 0

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            resolved_interface, renamed = resolve_dds_interface(
                args.network_interface
            )
            if renamed:
                logger_mp.warning(
                    "DDS interface %s is unavailable or has the wrong address; using %s for 192.168.123.222.",
                    args.network_interface,
                    resolved_interface,
                )
            args.network_interface = resolved_interface
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        if args.no_camera:
            camera_config = camera_config_without_images()
        else:
            img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
            camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        from televuer import TeleVuerWrapper
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     arm_reference_mode="head_yaw"
                                     )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        elif not args.sim:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            if status != 0:
                raise RuntimeError(
                    f"failed to enter exclusive debug control: status={status}, result={result}"
                )
            logger_mp.info("Exclusive debug control acquired.")

        xr_motion_data_ready = Value('b', False, lock=True)        # [input] whether XR hand/controller motion data has arrived

        if args.ee == "dex1_internal":
            if args.arm != "G1_29":
                raise ValueError("dex1_internal is only supported with --arm G1_29.")
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            if args.ee == "dex1_internal":
                arm_ctrl = G1_29_Arm_Internal_Dex1_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array,
                                                              dual_gripper_action_array, motion_mode=args.motion, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
            else:
                arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        elif args.arm == "H2":
            arm_ik = H2_ArmIK()
            arm_ctrl = H2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "R1_A5":
            arm_ik = R1_A5_ArmIK()
            arm_ctrl = R1_A5_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "R1_A7":
            arm_ik = R1_A7_ArmIK()
            official_profile = args.r1_control_profile == "official"
            arm_ctrl = R1_A7_ArmController(
                motion_mode=args.motion,
                simulation_mode=args.sim,
                arm_velocity_limit=(30.0 if official_profile else args.r1_arm_velocity_limit),
                joint_limit_margin=(0.0 if official_profile else np.deg2rad(args.r1_joint_margin_deg)),
                rate_limit_mode=("official" if official_profile else "independent"),
                clip_feedforward_torque=not official_profile,
                waist_velocity_limit=args.r1_waist_velocity_limit,
                waist_deadzone=args.r1_waist_thumbstick_deadzone,
            )
            if official_profile:
                logger_mp.warning(
                    "R1-A7 OFFICIAL motion parameters selected: no added pose envelope, "
                    "no added joint margin, upstream 30 rad/s rate behavior. XR watchdog "
                    "and explicit tracking activation remain enabled."
                )
            else:
                logger_mp.info(
                    "R1-A7 tuned limits: velocity=%.3f rad/s, joint margin=%.1f deg, "
                    "horizontal=+/-%.3f m, down=%.3f m, up=%.3f m, rotation=%.1f deg.",
                    args.r1_arm_velocity_limit,
                    args.r1_joint_margin_deg,
                    args.r1_workspace_horizontal_limit,
                    args.r1_workspace_down_limit,
                    args.r1_workspace_up_limit,
                    args.r1_workspace_rotation_limit_deg,
                )

        # end-effector
        if args.ee in ("dex3", "inspire_ftp", "inspire_dfx") and args.input_mode == "controller":
            raise ValueError(f"{args.ee} does not support controller input mode.")
        elif args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            if not args.sim and args.dex1_service_mode == "auto":
                dex1_service_process = start_dex1_service(
                    args.dex1_service_bin, args.network_interface
                )
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready,
                                                     enabled_arms=args.enabled_arms, input_mode=args.input_mode,
                                                     kp=args.dex1_kp, kd=args.dex1_kd,
                                                     command_step=args.dex1_command_step,
                                                     soft_torque_limit=args.dex1_soft_torque_limit,
                                                     hard_torque_limit=args.dex1_hard_torque_limit,
                                                     contact_samples=args.dex1_contact_samples,
                                                     torque_hysteresis_ratio=args.dex1_torque_hysteresis_ratio,
                                                     release_step=args.dex1_release_step,
                                                     runtime_state_timeout=args.dex1_state_timeout)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "hand":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_hand
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_hand(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                                dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "controller":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_ctrl
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        else:
            pass
        
        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recording_task_dir = os.path.join(args.task_dir, args.task_name)
            recorder = EpisodeWriter(task_dir = recording_task_dir,
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = record_frequency,
                                     rerun_log = not args.headless,
                                     task_before_accept = (
                                         args.task_before_accept if args.task_event_bind else None
                                     ),
                                     task_after_accept = (
                                         args.task_after_accept if args.task_event_bind else None
                                     ))
            recorder.info["recording_mode"] = (
                "r1_joint_precision" if args.joint_test_record else "teleoperation"
            )
            recorder.info["control_frequency"] = args.frequency
            if args.arm == "R1_A7" and args.ee in ("dex1", "dex1_internal"):
                recorder.info["raw_joint_names"] = [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                    "waist_yaw_joint",
                    "left_gripper",
                    "right_gripper",
                ]
                recorder.info["raw_state_dimension"] = 17
                recorder.info["raw_action_dimension"] = 17
                recorder.info["converted_model_joint_names"] = [
                    *recorder.info["raw_joint_names"][:14],
                    *recorder.info["raw_joint_names"][15:],
                ]
                recorder.info["converted_model_state_dimension"] = 16
                recorder.info["converted_model_action_dimension"] = 16
                recorder.info["conversion_excluded_fields"] = ["waist.qpos"]
                recorder.info["joint_names"]["waist"] = ["waist_yaw_joint"]
            recording_schema = camera_recording_schema(camera_config)
            if recording_schema is not None:
                recorder.info["camera_schema"] = recording_schema
            if args.task_event_bind:
                recorder.info["task_event_contract"] = {
                    "event": "item_accepted",
                    "task_before": args.task_before_accept,
                    "task_after": args.task_after_accept,
                }

                def handle_item_accepted(event):
                    if collection_batch is not None:
                        with collection_batch.transaction():
                            if not RECORD_RUNNING:
                                return {
                                    "status": "error",
                                    "message": "recording is not running",
                                }
                            # The first valid scan owns the active Episode. Later
                            # scanner emissions must not revalidate or rebind it.
                            if recorder.accepted_event() is not None:
                                return recorder.accept_item_event(event)
                            try:
                                collection_batch.validate_barcode(
                                    event.get("barcode", "")
                                )
                            except CollectionBatchError as exc:
                                return {"status": "error", "message": str(exc)}
                            reply = recorder.accept_item_event(event)
                            if (
                                reply.get("status") == "ok"
                                and not reply.get("duplicate", False)
                            ):
                                attempt = collection_batch.mark_item_accepted(
                                    event["event_id"]
                                )
                                if collection_batch.selection_mode == "flexible":
                                    logger_mp.info(
                                        "PRODUCT SELECTED %s | %s | item %d/%d | attempt %d",
                                        attempt["barcode"],
                                        attempt["name"],
                                        attempt["ordinal"],
                                        attempt["target_episodes"],
                                        attempt["attempt_number"],
                                    )
                            return reply
                    if not RECORD_RUNNING:
                        return {
                            "status": "error",
                            "message": "recording is not running",
                        }
                    return recorder.accept_item_event(event)

                task_event_server = TaskEventServer(
                    args.task_event_bind,
                    handle_item_accepted,
                    token=args.task_event_token,
                )
                task_event_server.start()

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press keyboard [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        if args.arm == "R1_A7":
            logger_mp.info("Press [h] for an operator-confirmed controlled home and exit.")
            if args.input_mode == "controller":
                logger_mp.info("Right controller [A]: start/resume; [B]: pause and hold.")
                if args.record:
                    logger_mp.info("Left controller [X] or keyboard [s]: start/save recording.")
                    if collection_batch is not None:
                        logger_mp.info(
                            "Left controller [Y] or keyboard [y]: abort the active episode, "
                            "or reject the last saved episode before the next recording."
                        )
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        if args.arm == "R1_A7":
            watchdog = XRWatchdog(hold_timeout=0.25, fault_timeout=1.0)
            pose_envelope = None
            if args.r1_control_profile == "tuned":
                pose_envelope = ActivationPoseEnvelope(
                    horizontal_limit=args.r1_workspace_horizontal_limit,
                    vertical_down_limit=args.r1_workspace_down_limit,
                    vertical_up_limit=args.r1_workspace_up_limit,
                    rotation_limit_degrees=args.r1_workspace_rotation_limit_deg,
                )
            diagnostics = DiagnosticsLogger(args.diagnostics_path)
            controller_buttons = (
                ControllerButtonEdges()
                if args.input_mode == "controller"
                else None
            )
        else:
            watchdog = None
            pose_envelope = None
            controller_buttons = None
        READY = True                  # now ready to (1) enter START state
        if args.arm != "R1_A7":
            while not START and not STOP: # wait for start or stop signal.
                time.sleep(0.033)
                if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                    head_img = img_client.get_head_frame()
                    if head_img.bgr is not None:
                        tv_wrapper.render_to_xr(head_img.bgr)

        logger_mp.info("---------------------XR input loop ready-------------------------")

        head_img = None
        left_wrist_img = None
        right_wrist_img = None
        scanner_img = None

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img and head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()
            if camera_config.get('scanner_camera', {}).get('enable_zmq', False):
                if args.record:
                    scanner_img = img_client.get_scanner_frame()

            # Y aborts an active attempt; between attempts it rejects the last save.
            if args.record and RECORD_REJECT:
                RECORD_REJECT = False
                if collection_batch is None:
                    logger_mp.warning(
                        "Reject ignored: start with --collection-plan to enable batch retry."
                    )
                else:
                    try:
                        if RECORD_RUNNING:
                            with collection_batch.transaction():
                                collection_batch.mark_aborting()
                                RECORD_RUNNING = False
                            recorder.save_episode()
                            logger_mp.warning(
                                "Current episode abort requested; finalizing it into rejected/."
                            )
                        else:
                            rejected = collection_batch.reject_last_accepted()
                            logger_mp.warning(
                                "Episode %04d rejected; retry %s %d/%d after reset.",
                                rejected["episode_id"],
                                rejected["name"],
                                rejected["ordinal"],
                                rejected["target_episodes"],
                            )
                    except CollectionBatchError as exc:
                        logger_mp.warning("Reject ignored: %s", exc)

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    can_start = True
                    if collection_batch is not None:
                        collection_batch.refresh()
                        slot = collection_batch.next_slot()
                        if collection_batch.status == "complete" or slot is None:
                            logger_mp.info("Collection batch is complete; recording not started.")
                            can_start = False
                        else:
                            recorder.info["collection"] = {
                                "batch_id": collection_batch.plan["batch_id"],
                                "selection_mode": collection_batch.selection_mode,
                            }
                            if collection_batch.selection_mode == "ordered":
                                recorder.info["collection"].update(slot.as_dict())
                    if can_start:
                        if recorder.create_episode():
                            recording_rate_gate.reset()
                            if collection_batch is not None:
                                attempt = collection_batch.start_attempt(
                                    recorder.episode_id, recorder.episode_dir
                                )
                                if collection_batch.selection_mode == "flexible":
                                    logger_mp.info(
                                        "RECORDING any product with remaining quota | batch %d/%d; "
                                        "the first accepted barcode selects this episode's product.",
                                        collection_batch.accepted_count,
                                        collection_batch.total_target,
                                    )
                                else:
                                    logger_mp.info(
                                        "RECORDING %s | %s | item %d/%d | attempt %d | batch %d/%d",
                                        attempt["barcode"],
                                        attempt["name"],
                                        attempt["ordinal"],
                                        attempt["target_episodes"],
                                        attempt["attempt_number"],
                                        collection_batch.accepted_count,
                                        collection_batch.total_target,
                                    )
                            RECORD_RUNNING = True
                        else:
                            logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    try:
                        if collection_batch is not None:
                            with collection_batch.transaction():
                                collection_batch.mark_saving()
                                RECORD_RUNNING = False
                        else:
                            RECORD_RUNNING = False
                        recorder.save_episode()
                        if args.sim:
                            publish_reset_category(1, reset_pose_publisher)
                    except CollectionBatchError as exc:
                        logger_mp.error("Failed to enter batch saving state: %s", exc)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            xr_state = None
            xr_age = np.inf
            if args.arm == "R1_A7":
                if controller_buttons is not None:
                    a_rising, b_rising, x_rising, y_rising = controller_buttons.update(
                        tele_data.right_ctrl_aButton,
                        tele_data.right_ctrl_bButton,
                        tele_data.left_ctrl_aButton,
                        tele_data.left_ctrl_bButton,
                    )
                    # Pause wins if both buttons rise in the same XR event.
                    if b_rising:
                        START = False
                        watchdog.pause()
                        arm_ctrl.set_tracking_active(False)
                        if pose_envelope is not None:
                            pose_envelope.disengage()
                        logger_mp.info(
                            "Right controller B pressed; tracking paused at measured pose."
                        )
                    elif a_rising:
                        if watchdog.state == TeleopState.FAULT:
                            logger_mp.error(
                                "Right controller A ignored: XR fault is latched; exit and restart."
                            )
                        elif not START:
                            START = True
                            logger_mp.info(
                                "Right controller A pressed; tracking activation requested."
                            )
                    if x_rising:
                        if not args.record:
                            logger_mp.warning(
                                "Left controller X ignored: recording is disabled."
                            )
                        elif not START:
                            logger_mp.warning(
                                "Left controller X ignored: start tracking with right controller A first."
                            )
                        else:
                            RECORD_TOGGLE = True
                            logger_mp.info(
                                "Left controller X pressed; recording toggle requested."
                            )
                    if y_rising:
                        if collection_batch is None:
                            logger_mp.warning(
                                "Left controller Y ignored: managed collection is disabled."
                            )
                        else:
                            RECORD_REJECT = True
                            logger_mp.info(
                                "Left controller Y pressed; abort/reject request received."
                            )
                previous_state = watchdog.state
                if START and watchdog.state == TeleopState.WAITING:
                    try:
                        watchdog.activate(tele_data.motion_data_timestamp)
                        if pose_envelope is not None:
                            pose_envelope.engage(
                                (tele_data.left_wrist_pose, tele_data.right_wrist_pose)
                            )
                        arm_ctrl.set_tracking_active(True)
                    except Exception as exc:
                        START = False
                        logger_mp.error(f"R1-A7 activation rejected: {exc}")
                xr_state, xr_age = watchdog.update(tele_data.motion_data_timestamp)
                if xr_state == TeleopState.HOLD:
                    if previous_state != TeleopState.HOLD:
                        logger_mp.warning("XR data is older than 250 ms; holding position.")
                    arm_ctrl.set_tracking_active(False)
                elif xr_state == TeleopState.ACTIVE and previous_state == TeleopState.HOLD:
                    arm_ctrl.set_tracking_active(True)
                    logger_mp.info("XR stream recovered before fault timeout; tracking resumed.")
                elif xr_state == TeleopState.FAULT:
                    fault_reason = watchdog.fault_reason
                    START = False
                    arm_ctrl.set_tracking_active(False)
                    if pose_envelope is not None:
                        pose_envelope.disengage()
            if args.ee in ("dex3", "inspire_ftp", "inspire_dfx", "brainco")  and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "brainco" and args.input_mode == "controller":
                with left_gripper_trigger_in.get_lock():
                    left_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                with left_gripper_squeeze_in.get_lock():
                    left_gripper_squeeze_in.value = tele_data.left_ctrl_squeezeValue
                with right_gripper_trigger_in.get_lock():
                    right_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                with right_gripper_squeeze_in.get_lock():
                    right_gripper_squeeze_in.value = tele_data.right_ctrl_squeezeValue
            elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = bool(
                    tele_data.motion_data_ready
                    and (xr_state in (None, TeleopState.ACTIVE))
                )
            
            # high level control
            if args.input_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            waist_joystick_command = 0.0
            if args.arm == "R1_A7" and args.input_mode == "controller":
                if args.enable_r1_waist_control and xr_state == TeleopState.ACTIVE:
                    waist_joystick_command = -float(
                        tele_data.right_ctrl_thumbstickValue[0]
                    )
                arm_ctrl.ctrl_waist_yaw(waist_joystick_command)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # Solve IK only while tracking is active. HOLD and FAULT retain the
            # measured pose and do not continue executing a stale XR target.
            sol_q = current_lr_arm_q.copy()
            sol_tauff = np.zeros_like(current_lr_arm_q)
            left_target = tele_data.left_wrist_pose
            right_target = tele_data.right_wrist_pose
            envelope_status = None
            if args.arm != "R1_A7" or xr_state == TeleopState.ACTIVE:
                if pose_envelope is not None:
                    left_target, right_target = pose_envelope.apply(
                        (left_target, right_target)
                    )
                    envelope_status = {
                        "left": pose_envelope.last_status[0],
                        "right": pose_envelope.last_status[1],
                    }
                time_ik_start = time.time()
                sol_q, sol_tauff = arm_ik.solve_ik(
                    left_target,
                    right_target,
                    current_lr_arm_q,
                    current_lr_arm_dq,
                )
                if args.arm == "R1_A7":
                    sol_q, sol_tauff = apply_enabled_arms(
                        sol_q,
                        sol_tauff,
                        current_lr_arm_q,
                        args.enabled_arms,
                    )
                logger_mp.debug(f"ik:\t{round(time.time() - time_ik_start, 6)}")
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            if diagnostics is not None:
                gripper_state = []
                gripper_target = []
                gripper_force_control = None
                if args.ee in ("dex1", "dex1_internal"):
                    with dual_gripper_data_lock:
                        gripper_state = list(dual_gripper_state_array[:])
                        gripper_target = list(dual_gripper_action_array[:])
                if args.ee == "dex1" and gripper_ctrl is not None:
                    gripper_force_control = gripper_ctrl.get_force_control_status()
                diagnostics.write(
                    control_profile=args.r1_control_profile,
                    waist_control_enabled=args.enable_r1_waist_control,
                    state=xr_state.value,
                    xr_age=xr_age,
                    raw_left_wrist_pose=tele_data.left_wrist_pose,
                    raw_right_wrist_pose=tele_data.right_wrist_pose,
                    left_trigger_value=tele_data.left_ctrl_triggerValue,
                    right_trigger_value=tele_data.right_ctrl_triggerValue,
                    target_left_wrist_pose=left_target,
                    target_right_wrist_pose=right_target,
                    envelope_status=envelope_status,
                    arm_q=current_lr_arm_q,
                    arm_target_q=sol_q,
                    gripper_q=gripper_state,
                    gripper_target_q=gripper_target,
                    gripper_force_control=gripper_force_control,
                    waist_joystick_command=waist_joystick_command,
                    waist_q=arm_ctrl.get_current_waist_yaw_q(),
                    waist_target_q=arm_ctrl.get_commanded_waist_yaw_q(),
                    fault_reason=fault_reason,
                )

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                if (
                    collection_batch is not None
                    and collection_batch.status == "saving"
                    and READY
                ):
                    accepted, attempt = collection_batch.finish_save(
                        recorder.accepted_event()
                    )
                    if accepted:
                        logger_mp.info(
                            "ACCEPTED episode_%04d | %s %d/%d | batch %d/%d. "
                            "Reset the product; X/s can start early.",
                            attempt["episode_id"],
                            attempt["name"],
                            attempt["ordinal"],
                            attempt["target_episodes"],
                            collection_batch.accepted_count,
                            collection_batch.total_target,
                        )
                    else:
                        logger_mp.error(
                            "REJECTED episode_%04d automatically: %s. Restore the scene and retry.",
                            attempt["episode_id"],
                            attempt["rejection_reason"],
                        )
                    batch_last_reset_announcement = None
                if (
                    collection_batch is not None
                    and collection_batch.status == "aborting"
                    and READY
                ):
                    attempt = collection_batch.finish_abort()
                    if attempt.get("barcode"):
                        logger_mp.warning(
                            "ABORTED episode_%04d -> rejected/ | %s %d/%d. "
                            "Restore the scene, then press X/s to record again.",
                            attempt["episode_id"],
                            attempt["name"],
                            attempt["ordinal"],
                            attempt["target_episodes"],
                        )
                    else:
                        logger_mp.warning(
                            "ABORTED episode_%04d -> rejected/ before a product was selected. "
                            "Restore the scene, then press X/s to record again.",
                            attempt["episode_id"],
                        )
                    batch_last_reset_announcement = None
                if collection_batch is not None:
                    previous_batch_status = collection_batch.status
                    collection_batch.refresh()
                    if collection_batch.status == "resetting":
                        remaining = int(np.ceil(collection_batch.reset_remaining_seconds()))
                        announcement = remaining // 5
                        if announcement != batch_last_reset_announcement:
                            logger_mp.info(
                                "RESET %ds remaining; press X/s when the scene is ready, "
                                "or Y/y to reject the last accepted episode.",
                                remaining,
                            )
                            batch_last_reset_announcement = announcement
                    elif previous_batch_status == "resetting":
                        batch_last_reset_announcement = None
                        if collection_batch.status == "complete":
                            logger_mp.info(
                                "COLLECTION COMPLETE: %d/%d accepted.",
                                collection_batch.accepted_count,
                                collection_batch.total_target,
                            )
                        else:
                            if collection_batch.selection_mode == "flexible":
                                logger_mp.info(
                                    "RESET COMPLETE. Choose any product with remaining quota; "
                                    "press X/s to record."
                                )
                            else:
                                slot = collection_batch.next_slot()
                                logger_mp.info(
                                    "RESET COMPLETE. Ready for %s %d/%d; press X/s to record.",
                                    slot.name,
                                    slot.ordinal,
                                    slot.target_episodes,
                                )
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                elif (args.ee == "brainco" and args.input_mode == "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                if args.arm == "R1_A7":
                    waist_state = [arm_ctrl.get_current_waist_yaw_q()]
                    waist_state_dq = [arm_ctrl.get_current_waist_yaw_dq()]
                    waist_action = [arm_ctrl.get_commanded_waist_yaw_q()]
                    # R1-A7 does not use the controller thumbsticks as a mobile-base action.
                    current_body_action = []
                else:
                    waist_state = []
                    waist_state_dq = []
                    waist_action = []

                # arm state and action (split into left/right halves by the arm's own DOF, so it works for any variant: H1/G1_23/R1_A5 = 4/5 per arm, G1_29/R1_A7 = 7)
                half = len(current_lr_arm_q) // 2
                left_arm_state,  right_arm_state  = current_lr_arm_q[:half], current_lr_arm_q[half:]
                left_arm_state_dq, right_arm_state_dq = current_lr_arm_dq[:half], current_lr_arm_dq[half:]
                left_arm_action, right_arm_action = sol_q[:half], sol_q[half:]
                left_arm_tauff, right_arm_tauff = sol_tauff[:half], sol_tauff[half:]
                record_monotonic_ns = time.monotonic_ns()
                if RECORD_RUNNING and recording_rate_gate.due(record_monotonic_ns):
                    record_wall_time_ns = time.time_ns()
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                        if camera_config.get('scanner_camera', {}).get('enable_zmq', False):
                            if scanner_img is not None:
                                colors[f"color_{4}"] = scanner_img.bgr
                            else:
                                logger_mp.warning("Scanner image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        elif camera_config['head_camera']['enable_zmq']:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                        if camera_config.get('scanner_camera', {}).get('enable_zmq', False):
                            if scanner_img is not None:
                                colors[f"color_{3}"] = scanner_img.bgr
                            else:
                                logger_mp.warning("Scanner image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   left_arm_state_dq.tolist(),
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   right_arm_state_dq.tolist(),
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        },
                        "waist": {
                            "qpos": waist_state,
                            "qvel": waist_state_dq,
                            "torque": [],
                        },
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": left_arm_tauff.tolist(),
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": right_arm_tauff.tolist(),
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        },
                        "waist": {
                            "qpos": waist_action,
                            "qvel": [],
                            "torque": [],
                        },
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    record_metadata = {
                        "record_monotonic_ns": record_monotonic_ns,
                        "record_wall_time_ns": record_wall_time_ns,
                        "xr_age_seconds": float(xr_age) if np.isfinite(xr_age) else None,
                        "teleop_state": xr_state.value if xr_state is not None else None,
                        "recording_mode": (
                            "r1_joint_precision"
                            if args.joint_test_record
                            else "teleoperation"
                        ),
                        "camera_receive_fps": {
                            "global": float(head_img.fps) if head_img is not None else None,
                            "left_wrist": float(left_wrist_img.fps) if left_wrist_img is not None else None,
                            "right_wrist": float(right_wrist_img.fps) if right_wrist_img is not None else None,
                            "scanner": float(scanner_img.fps) if scanner_img is not None else None,
                        },
                        "camera_frames": {
                            "global": camera_frame_metadata(head_img, record_monotonic_ns),
                            "left_wrist": camera_frame_metadata(left_wrist_img, record_monotonic_ns),
                            "right_wrist": camera_frame_metadata(right_wrist_img, record_monotonic_ns),
                            "scanner": camera_frame_metadata(scanner_img, record_monotonic_ns),
                        },
                    }
                    record_metadata.update(
                        recorder.task_context_for_timestamp(record_monotonic_ns)
                    )
                    if collection_batch is not None:
                        active_attempt = collection_batch.state.get("active_attempt")
                        if active_attempt is not None:
                            record_metadata["collection"] = {
                                "batch_id": collection_batch.plan["batch_id"],
                                **active_attempt,
                            }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state, metadata=record_metadata)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, metadata=record_metadata)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception as exc:
        exit_code = 1
        fault_reason = str(exc)
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if arm_ctrl is not None and args.arm == "R1_A7":
                arm_ctrl.shutdown(go_home=HOME_REQUESTED and fault_reason is None)
            elif arm_ctrl is not None:
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to stop arm controller: {e}")

        try:
            if gripper_ctrl is not None:
                gripper_ctrl.shutdown()
                if dex1_service_process is not None:
                    time.sleep(1.1)
        except Exception as e:
            logger_mp.error(f"Failed to stop gripper controller: {e}")

        try:
            stop_dex1_service(dex1_service_process)
        except Exception as e:
            logger_mp.error(f"Failed to stop Dex1 service: {e}")
        
        try:
            if task_event_server is not None:
                task_event_server.stop()
        except Exception as e:
            logger_mp.error(f"Failed to stop task event server: {e}")

        try:
            if ipc_server is not None:
                ipc_server.stop()
            elif listen_keyboard_thread is not None:
                stop_listening()
                listen_keyboard_thread.join(timeout=1.0)
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            if tv_wrapper is not None:
                tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if sim_state_subscriber is not None:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if recorder is not None:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        try:
            if diagnostics is not None:
                diagnostics.close()
        except Exception as e:
            logger_mp.error(f"Failed to close diagnostics log: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        raise SystemExit(exit_code)
