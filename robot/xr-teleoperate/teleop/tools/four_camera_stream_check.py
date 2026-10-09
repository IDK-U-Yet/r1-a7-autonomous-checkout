#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Validate all four teleimager streams without opening V4L2 devices directly."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import cv2
import numpy as np


TELEOP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TELEOP_ROOT.parents[2]
sys.path.insert(0, str(TELEOP_ROOT / "teleimager" / "src"))

from teleimager.image_client import ImageClient  # noqa: E402


@dataclass(frozen=True)
class CameraExpectation:
    name: str
    config_key: str
    getter_name: str
    shape: tuple[int, int, int]


CAMERAS = (
    CameraExpectation("global", "head_camera", "get_head_frame", (360, 640, 3)),
    CameraExpectation(
        "left_wrist", "left_wrist_camera", "get_left_wrist_frame", (480, 640, 3)
    ),
    CameraExpectation(
        "right_wrist", "right_wrist_camera", "get_right_wrist_frame", (480, 640, 3)
    ),
    CameraExpectation(
        "scanner", "scanner_camera", "get_scanner_frame", (480, 640, 3)
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--expected-fps", type=float, default=15.0)
    parser.add_argument("--minimum-fps-ratio", type=float, default=0.9)
    parser.add_argument("--ready-timeout", type=float, default=5.0)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=WORKSPACE_ROOT / "runtime" / "logs" / "four_camera_check.jpg",
        help="Path for a 2x2 decoded-frame contact sheet.",
    )
    args = parser.parse_args()
    if args.duration <= 0 or args.expected_fps <= 0 or args.ready_timeout <= 0:
        parser.error("duration, expected-fps and ready-timeout must be positive")
    if not 0 < args.minimum_fps_ratio <= 1:
        parser.error("minimum-fps-ratio must be in (0, 1]")
    return args


def write_contact_sheet(frames, output_path):
    tiles = []
    for camera in CAMERAS:
        frame = frames[camera.name].bgr
        tile = np.zeros((480, 640, 3), dtype=np.uint8)
        height, width = frame.shape[:2]
        if width != 640 or height > 480:
            raise ValueError(f"cannot place {camera.name} shape {frame.shape} in contact sheet")
        offset_y = (480 - height) // 2
        tile[offset_y : offset_y + height, :width] = frame
        cv2.putText(
            tile, camera.name, (13, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4
        )
        cv2.putText(
            tile,
            camera.name,
            (13, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            1,
        )
        tiles.append(tile)
    contact_sheet = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), contact_sheet):
        raise RuntimeError(f"failed to write snapshot: {output_path}")
    return output_path


def wait_for_frames(client, timeout):
    deadline = time.monotonic() + timeout
    latest = {}
    while time.monotonic() < deadline:
        for camera in CAMERAS:
            frame = getattr(client, camera.getter_name)()
            if frame.bgr is not None and frame.frame_sequence is not None:
                latest[camera.name] = frame
        if len(latest) == len(CAMERAS):
            return latest
        time.sleep(0.05)
    missing = [camera.name for camera in CAMERAS if camera.name not in latest]
    raise RuntimeError(f"timed out waiting for decoded frames: {', '.join(missing)}")


def main():
    args = parse_args()
    client = None
    failures = []
    try:
        client = ImageClient(host=args.host, request_bgr=True)
        config = client.get_cam_config()
        initial = wait_for_frames(client, args.ready_timeout)
        started = time.monotonic()
        time.sleep(args.duration)
        final = wait_for_frames(client, args.ready_timeout)
        elapsed = time.monotonic() - started

        print("camera        configured  measured  decoded shape")
        print("------------  ----------  --------  -------------")
        for camera in CAMERAS:
            camera_config = config.get(camera.config_key, {})
            configured_fps = float(camera_config.get("fps", 0.0))
            first_frame = initial[camera.name]
            last_frame = final[camera.name]
            measured_fps = (
                last_frame.frame_sequence - first_frame.frame_sequence
            ) / elapsed
            decoded_shape = tuple(int(value) for value in last_frame.bgr.shape)
            print(
                f"{camera.name:<12}  {configured_fps:>7.2f} FPS  "
                f"{measured_fps:>6.2f} FPS  {decoded_shape}"
            )

            if not camera_config.get("enable_zmq", False):
                failures.append(f"{camera.name}: ZMQ is disabled")
            if abs(configured_fps - args.expected_fps) > 0.1:
                failures.append(
                    f"{camera.name}: configured {configured_fps:.2f} FPS, "
                    f"expected {args.expected_fps:.2f}"
                )
            if measured_fps < args.expected_fps * args.minimum_fps_ratio:
                failures.append(
                    f"{camera.name}: measured {measured_fps:.2f} FPS is below "
                    f"{args.expected_fps * args.minimum_fps_ratio:.2f}"
                )
            if decoded_shape != camera.shape:
                failures.append(
                    f"{camera.name}: decoded shape {decoded_shape}, expected {camera.shape}"
                )
        if not failures:
            snapshot_path = write_contact_sheet(final, args.snapshot)
            print(f"snapshot: {snapshot_path}")
    except Exception as exc:
        failures.append(str(exc))
    finally:
        if client is not None:
            client.close()

    if failures:
        print("FOUR CAMERA CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("FOUR CAMERA CHECK PASSED")


if __name__ == "__main__":
    main()
