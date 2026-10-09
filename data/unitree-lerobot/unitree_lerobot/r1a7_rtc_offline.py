# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Offline RTC checkpoint probe. NPZ inputs only; no robot or camera clients."""

import argparse
import json
from pathlib import Path
import time

import numpy as np


def run_async_probe(session, state, images, task, log, steps):
    """Consume real model chunks on a simulated 15Hz command timeline only."""
    from unitree_lerobot.r1a7_rtc_async import RTCController
    controller = RTCController(session, prefetch_steps=7, initial_delay_steps=7)
    dispatched = waits = 0
    deadline = time.monotonic() + 60
    try:
        while dispatched < steps:
            started = time.monotonic()
            if started >= deadline:
                raise RuntimeError("offline asynchronous probe timed out")
            result = controller.poll()
            if result is not None:
                log.write(json.dumps({**result, "hardware_commands_sent": False}) + "\n")
            controller.request_if_needed(state, images, task)
            action = controller.pop()
            if action is None:
                waits += 1
                if controller.timeline.step:
                    controller.invalidate()
            else:
                if not np.isfinite(controller.last_hardware_action).all():
                    raise ValueError("offline asynchronous postprocessed action is not finite")
                dispatched += 1
            time.sleep(max(0, controller.interval - (time.monotonic()-started)))
        log.write(json.dumps({"event": "rtc_offline_async_complete", "simulated_actions": dispatched,
                              "wait_cycles_including_bootstrap": waits, "frequency": 15,
                              "hardware_commands_sent": False}) + "\n")
    finally:
        controller.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--observation-npz", type=Path, required=True,
                        help="state (16,) and four camera keys with RGB CHW uint8 arrays")
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--delay-steps", type=int, default=5)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--async-steps", type=int, default=0,
                        help="also consume N actions asynchronously at simulated 15Hz; no hardware")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 2:
        raise ValueError("trials must be at least 2")
    if args.async_steps < 0:
        raise ValueError("async-steps must be non-negative")
    from unitree_lerobot.r1a7_contract import CAMERA_KEYS, CAMERA_SHAPES
    from unitree_lerobot.r1a7_rtc_session import RTCSession

    with np.load(args.observation_npz, allow_pickle=False) as observation:
        state = observation["state"].copy()
        images = {key: observation[key].copy() for key in CAMERA_KEYS}
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("state must be finite (16,)")
    for key, image in images.items():
        h, w, c = CAMERA_SHAPES[key]
        if image.shape != (c, h, w) or image.dtype != np.uint8:
            raise ValueError(f"{key} requires RGB CHW uint8 {(c,h,w)}")
    session = RTCSession(dataset_root=args.dataset_root, policy_path=args.policy_path,
                         tokenizer_path=args.tokenizer_path, device_name=args.device, seed=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as log:
        prefix = None
        for trial in range(args.trials):
            started = time.monotonic()
            result = session.predict_normalized(state, images, args.task, prefix=prefix,
                                                 delay_steps=args.delay_steps if trial else 0)
            action = session.to_hardware_action(result[0])
            log.write(json.dumps({"event": "rtc_offline_probe", "trial": trial,
                                  "guided_prefix": prefix is not None,
                                  "elapsed_ms": (time.monotonic()-started)*1000,
                                  "shape": list(result.shape),
                                  "finite": bool(np.isfinite(action).all()),
                                  "hardware_commands_sent": False}) + "\n")
            prefix = result[5:].copy()
        if args.async_steps:
            run_async_probe(session, state, images, args.task, log, args.async_steps)
    print(f"Offline probe saved: {args.output}; hardware_commands_sent=false")


if __name__ == "__main__":
    main()
