# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
# for dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_                               # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
# for gripper
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import time
import os
import sys
import threading
from multiprocessing import Process, Array, Value, Lock

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from teleop.utils.weighted_moving_filter import WeightedMovingFilter

import logging_mp
logger_mp = logging_mp.getLogger(__name__)


Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


class Dex3_1_Controller:
    def __init__(self, left_hand_array_in, right_hand_array_in, dual_hand_data_lock = None, dual_hand_state_array_out = None,
                       dual_hand_action_array_out = None, fps = 100.0, Unit_Test = False, simulation_mode = False, xr_motion_data_ready_in = None):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_hand_array_in: [input] Left hand skeleton data (required from XR device) to hand_ctrl.control_process

        right_hand_array_in: [input] Right hand skeleton data (required from XR device) to hand_ctrl.control_process

        dual_hand_data_lock: Data synchronization lock for dual_hand_state_array and dual_hand_action_array

        dual_hand_state_array_out: [output] Return left(7), right(7) hand motor state

        dual_hand_action_array_out: [output] Return left(7), right(7) hand motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing

        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """
        logger_mp.info("Initialize Dex3_1_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)
        else:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3_Unit_Test)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Dex3_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            logger_mp.warning("[Dex3_1_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex3_1_Controller] Subscribe dds ok.")

        hand_control_process = Process(target=self.control_process, args=(left_hand_array_in, right_hand_array_in,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array_out, dual_hand_action_array_out, xr_motion_data_ready_in))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Dex3_1_Controller OK!")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                # Update left hand state
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                # Update right hand state
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
            time.sleep(0.002)
    
    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """set current left, right hand motor state target q"""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)
        # logger_mp.debug("hand ctrl publish ok.")
    
    def control_process(self, left_hand_array_in, right_hand_array_in, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array_out = None, dual_hand_action_array_out = None, xr_motion_data_ready_in = None):
        self.running = True

        left_q_target  = np.full(Dex3_Num_Motors, 0)
        right_q_target = np.full(Dex3_Num_Motors, 0)

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2

        # initialize dex3-1's left hand cmd msg
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q    = q
            self.left_msg.motor_cmd[id].dq   = dq
            self.left_msg.motor_cmd[id].tau  = tau
            self.left_msg.motor_cmd[id].kp   = kp
            self.left_msg.motor_cmd[id].kd   = kd

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode  
            self.right_msg.motor_cmd[id].q    = q
            self.right_msg.motor_cmd[id].dq   = dq
            self.right_msg.motor_cmd[id].tau  = tau
            self.right_msg.motor_cmd[id].kp   = kp
            self.right_msg.motor_cmd[id].kd   = kd  

        try:
            while self.running:
                start_time = time.time()
                # get dual hand state
                with left_hand_array_in.get_lock():
                    left_hand_data  = np.array(left_hand_array_in[:]).reshape(25, 3).copy()
                with right_hand_array_in.get_lock():
                    right_hand_data = np.array(right_hand_array_in[:]).reshape(25, 3).copy()
                if xr_motion_data_ready_in is not None:
                    with xr_motion_data_ready_in.get_lock():
                        xr_motion_data_ready = xr_motion_data_ready_in.value
                else:
                    xr_motion_data_ready = True

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if xr_motion_data_ready:
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array_out and dual_hand_action_array_out:
                    with dual_hand_data_lock:
                        dual_hand_state_array_out[:] = state_data
                        dual_hand_action_array_out[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex3_1_Controller has been closed.")

class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


kTopicGripperLeftCommand = "rt/dex1/left/cmd"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightCommand = "rt/dex1/right/cmd"
kTopicGripperRightState = "rt/dex1/right/state"


def map_dex1_input(value, input_mode, mapped_min=0.0, mapped_max=5.40):
    """Map XR input to Dex1 position; smaller targets close the gripper."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Dex1 input must be finite")
    if input_mode == "controller":
        input_min, input_max = 0.0, 10.0
    elif input_mode == "hand":
        input_min, input_max = 5.0, 7.0
    else:
        raise ValueError(f"unsupported Dex1 input mode: {input_mode}")
    return float(np.interp(value, [input_min, input_max], [mapped_min, mapped_max]))


class Dex1TorqueLimiter:
    """Rate-limit Dex1 position commands and cap contact torque with hysteresis."""

    def __init__(
        self,
        command_step=0.36,
        soft_torque_limit=None,
        hard_torque_limit=None,
        contact_samples=3,
        torque_hysteresis_ratio=0.8,
        release_step=0.018,
        state_timeout=0.1,
        position_min=0.0,
        position_max=5.4,
    ):
        self.command_step = float(command_step)
        self.soft_torque_limit = (
            None if soft_torque_limit is None else float(soft_torque_limit)
        )
        self.hard_torque_limit = (
            None if hard_torque_limit is None else float(hard_torque_limit)
        )
        self.contact_samples = int(contact_samples)
        self.torque_hysteresis_ratio = float(torque_hysteresis_ratio)
        self.release_step = float(release_step)
        self.state_timeout = float(state_timeout)
        self.position_min = float(position_min)
        self.position_max = float(position_max)
        self._validate_parameters()

        self.last_command = None
        self.state = "initializing"
        self.soft_limit_active = False
        self.above_soft_samples = 0
        self.below_release_samples = 0

    @property
    def force_limit_enabled(self):
        return self.soft_torque_limit is not None

    def _validate_parameters(self):
        finite_positive = {
            "command_step": self.command_step,
            "release_step": self.release_step,
            "state_timeout": self.state_timeout,
        }
        for name, value in finite_positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"Dex1 {name} must be positive and finite")
        if (
            not np.isfinite(self.position_min)
            or not np.isfinite(self.position_max)
            or self.position_min >= self.position_max
        ):
            raise ValueError("Dex1 position limits must be finite and increasing")
        if self.contact_samples < 1:
            raise ValueError("Dex1 contact_samples must be at least 1")
        if (
            not np.isfinite(self.torque_hysteresis_ratio)
            or not 0.0 <= self.torque_hysteresis_ratio < 1.0
        ):
            raise ValueError("Dex1 torque_hysteresis_ratio must be in [0, 1)")
        if (self.soft_torque_limit is None) != (self.hard_torque_limit is None):
            raise ValueError("Dex1 soft and hard torque limits must be set together")
        if self.force_limit_enabled:
            if (
                not np.isfinite(self.soft_torque_limit)
                or not np.isfinite(self.hard_torque_limit)
                or self.soft_torque_limit <= 0.0
                or self.hard_torque_limit <= self.soft_torque_limit
            ):
                raise ValueError(
                    "Dex1 torque limits must be positive, finite, and hard > soft"
                )

    def _clip_position(self, value):
        return float(np.clip(value, self.position_min, self.position_max))

    def _rate_limit(self, requested, measured):
        return self._clip_position(
            np.clip(
                requested,
                measured - self.command_step,
                measured + self.command_step,
            )
        )

    def _set_command(self, command, state):
        self.last_command = self._clip_position(command)
        self.state = state
        return self.last_command

    def _opening_command(self, requested, measured):
        return min(
            requested,
            max(self.last_command, self._rate_limit(requested, measured)),
        )

    def _reset_contact_state(self):
        self.soft_limit_active = False
        self.above_soft_samples = 0
        self.below_release_samples = 0

    def update(self, requested, measured, torque_est, state_age, motion_allowed=True):
        requested = float(requested)
        measured = float(measured)
        torque_est = float(torque_est)
        state_age = float(state_age)
        if not np.isfinite(requested) or not np.isfinite(measured):
            raise ValueError("Dex1 requested and measured positions must be finite")
        requested = self._clip_position(requested)
        measured = self._clip_position(measured)
        if self.last_command is None:
            self.last_command = measured

        state_fresh = (
            np.isfinite(torque_est)
            and np.isfinite(state_age)
            and 0.0 <= state_age <= self.state_timeout
        )
        if not state_fresh:
            self._reset_contact_state()
            if motion_allowed and requested > self.last_command + 1e-6:
                return self._set_command(
                    self._opening_command(requested, measured), "stale-opening"
                )
            return self._set_command(measured, "stale-hold")

        torque_magnitude = abs(torque_est)
        if self.force_limit_enabled and torque_magnitude >= self.hard_torque_limit:
            self.soft_limit_active = True
            self.above_soft_samples = self.contact_samples
            self.below_release_samples = 0
            return self._set_command(
                max(self.last_command, measured + self.release_step),
                "hard-release",
            )

        if not motion_allowed:
            self._reset_contact_state()
            return self._set_command(measured, "motion-hold")

        opening_requested = requested > self.last_command + 1e-6
        if opening_requested:
            self._reset_contact_state()
            return self._set_command(
                self._opening_command(requested, measured), "opening"
            )

        if not self.force_limit_enabled:
            return self._set_command(
                self._rate_limit(requested, measured), "monitoring"
            )

        release_threshold = (
            self.soft_torque_limit * self.torque_hysteresis_ratio
        )
        if self.soft_limit_active:
            if torque_magnitude <= release_threshold:
                self.below_release_samples += 1
            else:
                self.below_release_samples = 0
            if self.below_release_samples < self.contact_samples:
                return self._set_command(self.last_command, "soft-limit")
            self._reset_contact_state()

        if torque_magnitude >= self.soft_torque_limit:
            self.above_soft_samples += 1
        else:
            self.above_soft_samples = 0
        if self.above_soft_samples >= self.contact_samples:
            self.soft_limit_active = True
            self.below_release_samples = 0
            return self._set_command(self.last_command, "soft-limit")

        return self._set_command(
            self._rate_limit(requested, measured), "closing"
        )

    def status(self, torque_est, state_age):
        return {
            "state": self.state,
            "torque_est": float(torque_est),
            "torque_magnitude": abs(float(torque_est)),
            "state_age": float(state_age),
            "command_q": self.last_command,
            "soft_limit_active": self.soft_limit_active,
            "above_soft_samples": self.above_soft_samples,
        }


class Dex1_1_Gripper_Controller:
    def __init__(self, left_gripper_value_in, right_gripper_value_in, dual_gripper_data_lock = None, dual_gripper_state_out = None, dual_gripper_action_out = None, 
                       filter = True, fps = 200.0, Unit_Test = False, simulation_mode = False, xr_motion_data_ready_in = None,
                       enabled_arms = "both", input_mode = "hand", state_timeout = 5.0,
                       kp = 5.0, kd = 0.05, command_step = 0.36,
                       soft_torque_limit = None, hard_torque_limit = None,
                       contact_samples = 3, torque_hysteresis_ratio = 0.8,
                       release_step = 0.018, runtime_state_timeout = 0.1):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_gripper_value_in: [input] Left ctrl data (required from XR device) to control_thread

        right_gripper_value_in: [input] Right ctrl data (required from XR device) to control_thread

        dual_gripper_data_lock: Data synchronization lock for dual_gripper_state_array and dual_gripper_action_array

        dual_gripper_state_out: [output] Return left(1), right(1) gripper motor state

        dual_gripper_action_out: [output] Return left(1), right(1) gripper motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing

        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """

        logger_mp.info("Initialize Dex1_1_Gripper_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.gripper_sub_ready = False
        self.simulation_mode = simulation_mode
        if enabled_arms not in ("left", "right", "both"):
            raise ValueError(f"invalid enabled_arms value: {enabled_arms}")
        self.enabled_arms = enabled_arms
        if input_mode not in ("hand", "controller"):
            raise ValueError(f"invalid Dex1 input_mode value: {input_mode}")
        self.input_mode = input_mode
        self.running = True
        self.kp = float(kp)
        self.kd = float(kd)
        if not np.isfinite(self.kp) or self.kp <= 0.0:
            raise ValueError("Dex1 kp must be positive and finite")
        if not np.isfinite(self.kd) or self.kd < 0.0:
            raise ValueError("Dex1 kd must be non-negative and finite")
        limiter_parameters = {
            "command_step": command_step,
            "soft_torque_limit": soft_torque_limit,
            "hard_torque_limit": hard_torque_limit,
            "contact_samples": contact_samples,
            "torque_hysteresis_ratio": torque_hysteresis_ratio,
            "release_step": release_step,
            "state_timeout": runtime_state_timeout,
        }
        self.torque_limiters = [
            Dex1TorqueLimiter(**limiter_parameters),
            Dex1TorqueLimiter(**limiter_parameters),
        ]
        self.torque_limiter_lock = threading.Lock()
        self.gripper_state_lock = threading.Lock()
        self.gripper_q = np.zeros(2, dtype=np.float64)
        self.gripper_tau_est = np.zeros(2, dtype=np.float64)
        self.gripper_state_time = np.full(2, -np.inf, dtype=np.float64)
        self.gripper_side_ready = np.zeros(2, dtype=bool)
        logger_mp.info(
            "Dex1 control kp=%.3f kd=%.3f command_step=%.4f force_limit=%s soft=%s hard=%s",
            self.kp,
            self.kd,
            self.torque_limiters[0].command_step,
            self.torque_limiters[0].force_limit_enabled,
            self.torque_limiters[0].soft_torque_limit,
            self.torque_limiters[0].hard_torque_limit,
        )
        
        if filter and not self.simulation_mode:
            # A held controller trigger must keep advancing the filter window. If
            # duplicate vectors are skipped, one gripper can remain at a blended
            # target until the other trigger changes.
            self.smooth_filter = WeightedMovingFilter(
                np.array([0.5, 0.3, 0.2]), 2, skip_duplicates=False
            )
        else:
            self.smooth_filter = None
 
        # initialize handcmd publisher and handstate subscriber
        self.LeftGripperCmb_publisher = ChannelPublisher(kTopicGripperLeftCommand, MotorCmds_)
        self.LeftGripperCmb_publisher.Init()
        self.RightGripperCmb_publisher = ChannelPublisher(kTopicGripperRightCommand, MotorCmds_)
        self.RightGripperCmb_publisher.Init()

        self.LeftGripperState_subscriber = ChannelSubscriber(kTopicGripperLeftState, MotorStates_)
        self.LeftGripperState_subscriber.Init()
        self.RightGripperState_subscriber = ChannelSubscriber(kTopicGripperRightState, MotorStates_)
        self.RightGripperState_subscriber.Init()

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        state_deadline = time.monotonic() + float(state_timeout)
        while not self.gripper_sub_ready and self.running:
            if time.monotonic() >= state_deadline:
                self.running = False
                raise TimeoutError(
                    "timed out waiting for both Dex1 state topics; check the serial "
                    "service and motor IDs 0 (right) and 1 (left)"
                )
            time.sleep(0.01)
        logger_mp.info("[Dex1_1_Gripper_Controller] Subscribe dds ok.")

        self.gripper_control_thread = threading.Thread(target=self.control_thread, args=(left_gripper_value_in, right_gripper_value_in,
                                                                                         dual_gripper_data_lock, dual_gripper_state_out, dual_gripper_action_out, xr_motion_data_ready_in, enabled_arms, input_mode))
        self.gripper_control_thread.daemon = True
        self.gripper_control_thread.start()

        logger_mp.info("Initialize Dex1_1_Gripper_Controller OK!")

    def _subscribe_gripper_state(self):
        while self.running:
            left_gripper_msg  = self.LeftGripperState_subscriber.Read()
            right_gripper_msg  = self.RightGripperState_subscriber.Read()
            now = time.monotonic()
            with self.gripper_state_lock:
                for index, message in enumerate((left_gripper_msg, right_gripper_msg)):
                    if message is None or not message.states:
                        continue
                    position = float(message.states[0].q)
                    torque_est = float(message.states[0].tau_est)
                    if not np.isfinite(position) or not np.isfinite(torque_est):
                        continue
                    self.gripper_q[index] = position
                    self.gripper_tau_est[index] = torque_est
                    self.gripper_state_time[index] = now
                    self.gripper_side_ready[index] = True
                self.gripper_sub_ready = bool(np.all(self.gripper_side_ready))
            time.sleep(0.002)

    def _snapshot_gripper_state(self):
        with self.gripper_state_lock:
            positions = self.gripper_q.copy()
            torques = self.gripper_tau_est.copy()
            timestamps = self.gripper_state_time.copy()
        return positions, torques, time.monotonic() - timestamps

    def get_force_control_status(self):
        positions, torques, state_ages = self._snapshot_gripper_state()
        with self.torque_limiter_lock:
            sides = {
                side: {
                    "q": float(positions[index]),
                    **self.torque_limiters[index].status(
                        torques[index], state_ages[index]
                    ),
                }
                for index, side in enumerate(("left", "right"))
            }
        return {
            "enabled": self.torque_limiters[0].force_limit_enabled,
            "kp": self.kp,
            "kd": self.kd,
            "command_step": self.torque_limiters[0].command_step,
            "soft_torque_limit": self.torque_limiters[0].soft_torque_limit,
            "hard_torque_limit": self.torque_limiters[0].hard_torque_limit,
            "sides": sides,
        }
    
    def ctrl_dual_gripper(self, dual_gripper_action):
        """set current left, right gripper motor cmd target q"""
        self.left_gripper_msg.cmds[0].q  = dual_gripper_action[0]
        self.right_gripper_msg.cmds[0].q = dual_gripper_action[1]

        self.LeftGripperCmb_publisher.Write(self.left_gripper_msg)
        self.RightGripperCmb_publisher.Write(self.right_gripper_msg)
        # logger_mp.debug("gripper ctrl publish ok.")
    
    def control_thread(self, left_gripper_value_in, right_gripper_value_in, dual_hand_data_lock = None,
                             dual_gripper_state_out = None, dual_gripper_action_out = None, xr_motion_data_ready_in = None,
                             enabled_arms = "both", input_mode = "hand"):
        LEFT_MAPPED_MIN  = 0.0           # The minimum initial motor position when the gripper closes at startup.
        RIGHT_MAPPED_MIN = 0.0           # The minimum initial motor position when the gripper closes at startup.
        # The maximum initial motor position when the gripper closes before calibration (with the rail stroke calculated as 0.6 cm/rad * 9 rad = 5.4 cm).
        LEFT_MAPPED_MAX = LEFT_MAPPED_MIN + 5.40 
        RIGHT_MAPPED_MAX = RIGHT_MAPPED_MIN + 5.40
        left_target_action  = (LEFT_MAPPED_MAX - LEFT_MAPPED_MIN) / 2.0
        right_target_action = (RIGHT_MAPPED_MAX - RIGHT_MAPPED_MIN) / 2.0

        dq = 0.0
        tau = 0.0
        # initialize gripper cmd msg
        self.left_gripper_msg  = MotorCmds_()
        self.left_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self.right_gripper_msg = MotorCmds_()
        self.right_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]

        self.left_gripper_msg.cmds[0].dq  = dq
        self.left_gripper_msg.cmds[0].tau = tau
        self.left_gripper_msg.cmds[0].kp  = self.kp
        self.left_gripper_msg.cmds[0].kd  = self.kd

        self.right_gripper_msg.cmds[0].dq  = dq
        self.right_gripper_msg.cmds[0].tau = tau
        self.right_gripper_msg.cmds[0].kp  = self.kp
        self.right_gripper_msg.cmds[0].kd  = self.kd
        previous_limiter_states = [None, None]
        try:
            while self.running:
                start_time = time.time()
                # get dual hand skeletal point state from XR device
                with left_gripper_value_in.get_lock():
                    left_gripper_value  = left_gripper_value_in.value
                with right_gripper_value_in.get_lock():
                    right_gripper_value = right_gripper_value_in.value
                if xr_motion_data_ready_in is not None:
                    with xr_motion_data_ready_in.get_lock():
                        xr_motion_data_ready = xr_motion_data_ready_in.value
                else:
                    xr_motion_data_ready = True
                # get current dual gripper motor state
                dual_gripper_state, dual_gripper_torque, state_ages = self._snapshot_gripper_state()

                if xr_motion_data_ready:
                    left_target_action = map_dex1_input(
                        left_gripper_value, input_mode, LEFT_MAPPED_MIN, LEFT_MAPPED_MAX
                    )
                    right_target_action = map_dex1_input(
                        right_gripper_value, input_mode, RIGHT_MAPPED_MIN, RIGHT_MAPPED_MAX
                    )
                else:
                    left_target_action = dual_gripper_state[0]
                    right_target_action = dual_gripper_state[1]
                if enabled_arms == "left":
                    right_target_action = dual_gripper_state[1]
                elif enabled_arms == "right":
                    left_target_action = dual_gripper_state[0]
                requested_actions = np.array(
                    [left_target_action, right_target_action], dtype=np.float64
                )
                if self.smooth_filter:
                    self.smooth_filter.add_data(requested_actions)
                    requested_actions = self.smooth_filter.filtered_data
                side_motion_allowed = np.array(
                    [enabled_arms != "right", enabled_arms != "left"], dtype=bool
                )
                with self.torque_limiter_lock:
                    dual_gripper_action = np.array(
                        [
                            limiter.update(
                                requested_actions[index],
                                dual_gripper_state[index],
                                dual_gripper_torque[index],
                                state_ages[index],
                                motion_allowed=(
                                    xr_motion_data_ready and side_motion_allowed[index]
                                ),
                            )
                            for index, limiter in enumerate(self.torque_limiters)
                        ]
                    )
                    limiter_states = [limiter.state for limiter in self.torque_limiters]
                for index, state in enumerate(limiter_states):
                    if state != previous_limiter_states[index]:
                        log = (
                            logger_mp.warning
                            if state in ("soft-limit", "hard-release", "stale-hold")
                            else logger_mp.info
                        )
                        log(
                            "Dex1 %s limiter state=%s tau_est=%.4f state_age=%.4f",
                            ("left", "right")[index],
                            state,
                            dual_gripper_torque[index],
                            state_ages[index],
                        )
                        previous_limiter_states[index] = state

                if dual_gripper_state_out and dual_gripper_action_out:
                    with dual_hand_data_lock:
                        dual_gripper_state_out[:] = dual_gripper_state - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])
                        dual_gripper_action_out[:] = dual_gripper_action - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])

                self.ctrl_dual_gripper(dual_gripper_action)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex1_1_Gripper_Controller has been closed.")

    def shutdown(self):
        self.running = False
        if hasattr(self, "gripper_control_thread"):
            self.gripper_control_thread.join(timeout=1.0)
        if hasattr(self, "subscribe_state_thread"):
            self.subscribe_state_thread.join(timeout=1.0)

class Gripper_JointIndex(IntEnum):
    kGripper = 0


if __name__ == "__main__":
    import argparse
    from televuer import TeleVuerWrapper
    from teleimager import ImageClient

    parser = argparse.ArgumentParser()
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device tracking source')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire1', 'brainco'], help='Select end effector controller')
    args = parser.parse_args()
    logger_mp.info(f"args:{args}\n")

    ChannelFactoryInitialize(1) # 0 for real robot, 1 for simulation
    
    # image client
    img_client = ImageClient(host='127.0.0.1') #host='192.168.123.164'
    if not img_client.has_head_cam():
        logger_mp.error("Head camera is required. Please enable head camera on the image server side.")
    head_img_shape = img_client.get_head_shape()
    tv_binocular = img_client.head_is_binocular()

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVuerWrapper(binocular=tv_binocular, use_hand_tracking=args.xr_mode == "hand", img_shape=head_img_shape, return_hand_rot_data = False)

# end-effector
    if args.ee == "dex3":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
    elif args.ee == "dex1":
        left_gripper_value = Value('d', 0.0, lock=True)        # [input]
        right_gripper_value = Value('d', 0.0, lock=True)       # [input]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):\n")
    if user_input.lower() == 's':
        while True:
            head_img, head_img_fps = img_client.get_head_frame()
            tv_wrapper.set_display_image(head_img)
            tele_data = tv_wrapper.get_tele_data()
            if args.ee == "dex3" and args.xr_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex1" and args.xr_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.xr_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass

            # with dual_hand_data_lock:
            #     logger_mp.info(f"state : {list(dual_hand_state_array)} \naction: {list(dual_hand_action_array)} \n")
            time.sleep(0.01)
