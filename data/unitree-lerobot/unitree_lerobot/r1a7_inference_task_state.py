# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Inference task state driven by the checkout ``item_accepted`` event.

This module contains no robot or policy publisher. A real-time inference loop
must consume the generation counter and reset its policy action queue whenever
the generation changes.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Sequence

from teleop.utils.task_events import TaskEventServer


DEFAULT_TASK_BEFORE = (
    "Pick up a product from the blue tray on the left, present it to the scanner "
    "in the scanning area, and wait until the scan is accepted."
)
DEFAULT_TASK_AFTER = "Place the scanned product in the black tray on the right."
DEFAULT_EXPECTED_BARCODE = None


@dataclass(frozen=True)
class InferenceTaskSnapshot:
    run_id: str
    generation: int
    phase: str
    task: str
    accepted_event_id: str | None
    accepted_barcode: str | None
    policy_reset_required: bool


class InferenceTaskState:
    """Thread-safe two-phase task state for one inference run."""

    def __init__(
        self,
        *,
        expected_barcode: str | None = DEFAULT_EXPECTED_BARCODE,
        task_before: str = DEFAULT_TASK_BEFORE,
        task_after: str = DEFAULT_TASK_AFTER,
        run_id: str | None = None,
        scan_armed_after_ns: int = 0,
    ):
        expected_barcode = (
            None if expected_barcode is None else str(expected_barcode).strip() or None
        )
        task_before = str(task_before).strip()
        task_after = str(task_after).strip()
        if not task_before or not task_after:
            raise ValueError("task labels must not be empty")
        if task_before == task_after:
            raise ValueError("before and after task labels must differ")
        self.expected_barcode = expected_barcode
        self.task_before = task_before
        self.task_after = task_after
        self.run_id = run_id or f"inference_{uuid.uuid4().hex}"
        self._lock = threading.Lock()
        self._generation = 0
        self._accepted_event = None
        self._scan_armed = True
        self._scan_armed_after_ns = int(scan_armed_after_ns)

    def snapshot(self, *, last_consumed_generation: int | None = None):
        with self._lock:
            generation = self._generation
            event = None if self._accepted_event is None else dict(self._accepted_event)
            run_id = self.run_id
        after_accept = event is not None
        return InferenceTaskSnapshot(
            run_id=run_id,
            generation=generation,
            phase="place" if after_accept else "scan",
            task=self.task_after if after_accept else self.task_before,
            accepted_event_id=None if event is None else event["event_id"],
            accepted_barcode=None if event is None else event["barcode"],
            policy_reset_required=(
                last_consumed_generation is not None
                and generation != last_consumed_generation
            ),
        )

    def pause_for_next_cycle(self):
        """Reject new scans while the operator clears the work area."""
        with self._lock:
            if not self._scan_armed:
                raise RuntimeError("next cycle is already paused")
            self._scan_armed = False
            return {
                "run_id": self.run_id,
                "generation": self._generation,
                "phase": "place" if self._accepted_event is not None else "scan",
            }

    def pause_for_restart(self):
        """Invalidate queued inference and reject scans while returning to start."""
        with self._lock:
            self._scan_armed = False
            self._generation += 1
            event = self._accepted_event
            return {
                "run_id": self.run_id,
                "generation": self._generation,
                "phase": "place" if event is not None else "scan",
            }

    def confirm_completion(self):
        """Accept operator-confirmed placement and invalidate in-flight actions."""
        with self._lock:
            if self._accepted_event is None:
                raise RuntimeError("cannot complete before a scan is accepted")
            if not self._scan_armed:
                raise RuntimeError("inference cycle is already paused")
            self._scan_armed = False
            self._generation += 1
            return {
                "run_id": self.run_id,
                "generation": self._generation,
                "phase": "place",
                "accepted_event_id": self._accepted_event["event_id"],
                "accepted_barcode": self._accepted_event["barcode"],
            }

    def start_restarted_cycle(self):
        """Begin a fresh scan run after the robot reaches its start pose."""
        with self._lock:
            if self._scan_armed:
                raise RuntimeError("restart cycle requires scanning to be paused")
            previous_run_id = self.run_id
            self.run_id = f"inference_{uuid.uuid4().hex}"
            self._accepted_event = None
            self._generation += 1
            self._scan_armed_after_ns = time.time_ns()
            self._scan_armed = True
            return {
                "previous_run_id": previous_run_id,
                "run_id": self.run_id,
                "generation": self._generation,
                "phase": "scan",
            }

    def start_next_cycle(self):
        """Re-arm scanning only after the robot has reached its start pose."""
        with self._lock:
            if self._scan_armed:
                raise RuntimeError("next cycle requires an explicit pause")
            previous_run_id = self.run_id
            self.run_id = f"inference_{uuid.uuid4().hex}"
            self._accepted_event = None
            self._generation += 1
            self._scan_armed_after_ns = time.time_ns()
            self._scan_armed = True
            return {
                "previous_run_id": previous_run_id,
                "run_id": self.run_id,
                "generation": self._generation,
                "phase": "scan",
            }

    def handle_item_accepted(self, event):
        barcode = str(event.get("barcode", "")).strip()
        if not barcode:
            return {
                "status": "error",
                "message": "item_accepted event is missing a barcode",
                "inference_run_id": self.run_id,
            }
        if self.expected_barcode is not None and barcode != self.expected_barcode:
            return {
                "status": "error",
                "message": (
                    f"wrong product for this inference run: expected "
                    f"{self.expected_barcode}, received {barcode or '<empty>'}"
                ),
                "inference_run_id": self.run_id,
            }

        with self._lock:
            if not self._scan_armed:
                return {
                    "status": "error",
                    "message": "scanner is paused until the next cycle is ready",
                    "inference_run_id": self.run_id,
                }
            source_ns = event.get("source_wall_time_ns")
            if self._scan_armed_after_ns and (
                not isinstance(source_ns, int) or source_ns < self._scan_armed_after_ns
            ):
                return {
                    "status": "error",
                    "message": "scan predates the current inference cycle",
                    "inference_run_id": self.run_id,
                }
            if self._accepted_event is not None:
                return {
                    "status": "ok",
                    "duplicate": True,
                    "duplicate_scope": "episode",
                    "episode_id": self.run_id,
                    "inference_run_id": self.run_id,
                    "accepted_barcode": self._accepted_event["barcode"],
                    "generation": self._generation,
                    "phase": "place",
                    "active_task": self.task_after,
                    "policy_reset_required": False,
                }
            self._accepted_event = dict(event)
            self._generation += 1
            generation = self._generation
            run_id = self.run_id

        return {
            "status": "ok",
            "duplicate": False,
            "episode_id": run_id,
            "inference_run_id": run_id,
            "accepted_barcode": barcode,
            "generation": generation,
            "phase": "place",
            "active_task": self.task_after,
            "policy_reset_required": True,
        }


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Receive scanner item_accepted events and expose the PI0.5 task switch. "
            "This diagnostic receiver does not run a policy or control a robot."
        )
    )
    parser.add_argument("--bind", default="tcp://0.0.0.0:55580")
    parser.add_argument("--token", default=os.environ.get("R1_TASK_EVENT_TOKEN"))
    parser.add_argument(
        "--expected-barcode",
        default=DEFAULT_EXPECTED_BARCODE,
        help=(
            "Optional single-product guard. Omit to accept the first catalog-validated "
            "barcode received for the run."
        ),
    )
    parser.add_argument("--task-before", default=DEFAULT_TASK_BEFORE)
    parser.add_argument("--task-after", default=DEFAULT_TASK_AFTER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = InferenceTaskState(
        expected_barcode=args.expected_barcode,
        task_before=args.task_before,
        task_after=args.task_after,
    )

    def handle_event(event):
        reply = state.handle_item_accepted(event)
        print(json.dumps({"event": event, "reply": reply}, ensure_ascii=False), flush=True)
        return reply

    server = TaskEventServer(args.bind, handle_event, token=args.token)
    stop_requested = threading.Event()

    def request_stop(_signum, _frame):
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    server.start()
    print(
        json.dumps(
            {
                "status": "ready",
                "bind": args.bind,
                "expected_barcode": state.expected_barcode,
                "state": asdict(state.snapshot()),
                "hardware_commands_sent": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        while not stop_requested.wait(0.2):
            pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
