#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Read-only PICO WebXR input check. This program never initializes DDS."""

import argparse
from pathlib import Path
import socket
import sys
import time

import numpy as np


LOCAL_TELEVUER_SRC = Path(__file__).resolve().parents[1] / "televuer" / "src"
sys.path.insert(0, str(LOCAL_TELEVUER_SRC))


def require_vuer_port_available(port=8012):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("0.0.0.0", port))
    except OSError as exc:
        raise SystemExit(
            f"FAIL: TCP port {port} is already in use. Stop the existing Vuer/teleop "
            "process normally before running this read-only check."
        ) from exc
    finally:
        probe.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--frequency", type=float, default=30.0)
    args = parser.parse_args()
    if args.duration <= 0.0 or args.frequency <= 0.0:
        parser.error("duration and frequency must be positive")

    require_vuer_port_available()
    from televuer import TeleVuerWrapper

    wrapper = TeleVuerWrapper(
        use_hand_tracking=False,
        binocular=False,
        img_shape=(480, 640),
        display_mode="pass-through",
        zmq=False,
        webrtc=False,
        arm_reference_mode="head_yaw",
    )
    deadline = time.monotonic() + args.duration
    samples = 0
    stale_samples = 0
    invalid_samples = 0
    max_age = 0.0
    left_trigger_min, left_trigger_max = np.inf, -np.inf
    right_trigger_min, right_trigger_max = np.inf, -np.inf
    try:
        while time.monotonic() < deadline:
            data = wrapper.get_tele_data()
            if data.motion_data_timestamp:
                age = max(0.0, time.monotonic() - data.motion_data_timestamp)
                samples += 1
                max_age = max(max_age, age)
                stale_samples += int(age > 0.25)
                invalid_samples += int(
                    not np.all(np.isfinite(data.left_wrist_pose))
                    or not np.all(np.isfinite(data.right_wrist_pose))
                    or not np.isfinite(data.left_ctrl_triggerValue)
                    or not np.isfinite(data.right_ctrl_triggerValue)
                )
                left_trigger_min = min(left_trigger_min, data.left_ctrl_triggerValue)
                left_trigger_max = max(left_trigger_max, data.left_ctrl_triggerValue)
                right_trigger_min = min(right_trigger_min, data.right_ctrl_triggerValue)
                right_trigger_max = max(right_trigger_max, data.right_ctrl_triggerValue)
            time.sleep(1.0 / args.frequency)
    finally:
        wrapper.close()

    if samples == 0:
        raise SystemExit("FAIL: no controller motion events received")
    trigger_ok = (
        left_trigger_max - left_trigger_min >= 1.0
        and right_trigger_max - right_trigger_min >= 1.0
    )
    passed = stale_samples == 0 and invalid_samples == 0 and trigger_ok
    print(
        "PASS" if passed else "FAIL",
        f"samples={samples}",
        f"max_age={max_age:.3f}s",
        f"stale_samples={stale_samples}",
        f"invalid_samples={invalid_samples}",
        f"left_trigger=[{left_trigger_min:.3f}, {left_trigger_max:.3f}]",
        f"right_trigger=[{right_trigger_min:.3f}, {right_trigger_max:.3f}]",
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
