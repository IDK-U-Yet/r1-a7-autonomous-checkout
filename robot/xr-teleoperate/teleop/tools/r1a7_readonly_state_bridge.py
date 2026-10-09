#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Expose R1-A7 measured state over local ZMQ without creating publishers."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from dataclasses import dataclass

import numpy as np
import zmq


ARM_INDICES = tuple(range(15, 29))


@dataclass
class LatestSample:
    value: object | None = None
    monotonic_ns: int | None = None


class R1A7ReadOnlyState:
    """DDS subscribers for dual-arm and Dex1 policy state."""

    def __init__(self, network_interface: str):
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        ChannelFactoryInitialize(0, networkInterface=network_interface)
        self._lock = threading.Lock()
        self._lowstate = LatestSample()
        self._left_gripper = LatestSample()
        self._right_gripper = LatestSample()
        self._subscribers = (
            ChannelSubscriber("rt/lowstate", LowState_),
            ChannelSubscriber("rt/dex1/left/state", MotorStates_),
            ChannelSubscriber("rt/dex1/right/state", MotorStates_),
        )
        self._subscribers[0].Init(self._on_lowstate, 1)
        self._subscribers[1].Init(
            lambda message: self._on_gripper(self._left_gripper, message), 1
        )
        self._subscribers[2].Init(
            lambda message: self._on_gripper(self._right_gripper, message), 1
        )

    def _on_lowstate(self, message) -> None:
        try:
            positions = np.asarray(
                [message.motor_state[index].q for index in ARM_INDICES],
                dtype=np.float64,
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        if positions.shape != (14,) or not np.all(np.isfinite(positions)):
            return
        with self._lock:
            self._lowstate = LatestSample(positions, time.monotonic_ns())

    def _on_gripper(self, target: LatestSample, message) -> None:
        try:
            position = float(message.states[0].q)
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        if not np.isfinite(position):
            return
        with self._lock:
            target.value = position
            target.monotonic_ns = time.monotonic_ns()

    def snapshot(self) -> dict:
        with self._lock:
            lowstate = LatestSample(self._lowstate.value, self._lowstate.monotonic_ns)
            left = LatestSample(
                self._left_gripper.value, self._left_gripper.monotonic_ns
            )
            right = LatestSample(
                self._right_gripper.value, self._right_gripper.monotonic_ns
            )
        missing = [
            name
            for name, sample in (
                ("rt/lowstate", lowstate),
                ("rt/dex1/left/state", left),
                ("rt/dex1/right/state", right),
            )
            if sample.monotonic_ns is None
        ]
        if missing:
            return {"status": "not_ready", "missing_topics": missing}
        state = np.concatenate(
            (np.asarray(lowstate.value), (float(left.value), float(right.value)))
        )
        return {
            "status": "ok",
            "state": state.tolist(),
            "sample_monotonic_ns": {
                "lowstate": lowstate.monotonic_ns,
                "left_gripper": left.monotonic_ns,
                "right_gripper": right.monotonic_ns,
            },
            "bridge_monotonic_ns": time.monotonic_ns(),
            "hardware_commands_sent": False,
        }

    def close(self) -> None:
        for subscriber in self._subscribers:
            subscriber.Close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--bind", default="tcp://127.0.0.1:55582")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = R1A7ReadOnlyState(args.network_interface)
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    stopped = threading.Event()

    def request_stop(_signum, _frame):
        stopped.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    print(
        json.dumps(
            {
                "status": "ready",
                "bind": args.bind,
                "network_interface": args.network_interface,
                "dds_domain": 0,
                "hardware_commands_sent": False,
            }
        ),
        flush=True,
    )
    try:
        while not stopped.is_set():
            if socket not in dict(poller.poll(200)):
                continue
            request = socket.recv_json()
            if request != {"request": "state"}:
                socket.send_json({"status": "error", "message": "invalid request"})
                continue
            socket.send_json(source.snapshot())
    finally:
        socket.close(linger=0)
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
