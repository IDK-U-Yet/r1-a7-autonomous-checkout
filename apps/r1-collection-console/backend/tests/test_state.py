from __future__ import annotations

import json
import os
import time
from pathlib import Path

from collection_console.state import (
    read_batch_snapshot,
    read_collector_snapshot,
    read_last_json_lines,
    read_scanner_snapshot,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_batch_distinguishes_scan_from_final_acceptance(tmp_path):
    plan_path = tmp_path / "plan.json"
    task_dir = tmp_path / "batch"
    write_json(
        plan_path,
        {
            "batch_id": "batch-1",
            "selection_mode": "ordered",
            "reset_seconds": 30,
            "products": [
                {"barcode": "123", "sku": "SKU-1", "name": "Product", "target_episodes": 2}
            ],
        },
    )
    write_json(
        task_dir / "batch_state.json",
        {
            "batch_id": "batch-1",
            "plan_path": str(plan_path),
            "status": "recording",
            "active_attempt": {
                "episode_id": 1,
                "barcode": "123",
                "sku": "SKU-1",
                "name": "Product",
                "ordinal": 1,
                "target_episodes": 2,
                "attempt_number": 1,
                "item_accepted_event_id": "event-1",
            },
            "accepted": [],
            "rejected": [],
            "last_finalized": None,
            "reset_deadline_wall_time_ns": None,
        },
    )

    snapshot = read_batch_snapshot(task_dir)

    assert snapshot["phase"] == "placing"
    assert snapshot["accepted"] == 0
    assert snapshot["remaining"] == 2


def test_batch_rebuilds_history_and_reset_countdown(tmp_path):
    now_ns = 1_000_000_000_000
    plan_path = tmp_path / "plan.json"
    task_dir = tmp_path / "batch"
    write_json(
        plan_path,
        {
            "batch_id": "batch-1",
            "products": [
                {"barcode": "123", "sku": "SKU-1", "name": "Product", "target_episodes": 3}
            ],
        },
    )
    write_json(
        task_dir / "batch_state.json",
        {
            "batch_id": "batch-1",
            "plan_path": str(plan_path),
            "status": "resetting",
            "active_attempt": None,
            "accepted": [
                {
                    "episode_id": 0,
                    "barcode": "123",
                    "name": "Product",
                    "ordinal": 1,
                    "saved_wall_time_ns": now_ns - 10,
                }
            ],
            "rejected": [
                {
                    "episode_id": 1,
                    "barcode": "123",
                    "rejection_reason": "operator_aborted_during_recording",
                    "rejected_wall_time_ns": now_ns - 5,
                }
            ],
            "last_finalized": {"episode_id": 0, "status": "accepted"},
            "reset_deadline_wall_time_ns": now_ns + 5_000_000_000,
        },
    )

    snapshot = read_batch_snapshot(task_dir, now_wall_time_ns=now_ns)

    assert snapshot["accepted"] == 1
    assert snapshot["rejected"] == 1
    assert snapshot["reset_remaining_seconds"] == 5.0
    assert [event["type"] for event in snapshot["history"]] == [
        "episode_rejected",
        "episode_accepted",
    ]


def test_scanner_skips_partial_json_and_reports_latest_health(tmp_path):
    path = tmp_path / "scanner.jsonl"
    now_ns = time.time_ns()
    path.write_text(
        json.dumps(
            {
                "event": "scanner_health",
                "wall_time_ns": now_ns,
                "scanner_health": {"ready": True, "state": "ready", "device": "/dev/test"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "event": "item_accepted",
                "wall_time_ns": now_ns + 1,
                "barcode": "123",
                "product": {"name": "Product"},
                "recorder_reply": {"episode_id": 4},
            }
        )
        + "\n{\"event\":",
        encoding="utf-8",
    )

    health, events = read_scanner_snapshot(path, now_wall_time_ns=now_ns + 2)

    assert health["online"] is True
    assert health["device"] == "/dev/test"
    assert events[0]["type"] == "item_accepted"
    assert events[0]["episode_id"] == 4
    assert len(read_last_json_lines(path, 10)) == 2


def test_collector_uses_diagnostics_mtime_for_liveness(tmp_path):
    path = tmp_path / "diagnostics.jsonl"
    path.write_text(
        json.dumps({"state": "active", "xr_age": 0.012, "fault_reason": None}) + "\n",
        encoding="utf-8",
    )
    now = time.time()
    os.utime(path, (now, now))

    online = read_collector_snapshot(path, now_wall_time=now + 1)
    offline = read_collector_snapshot(path, now_wall_time=now + 3)

    assert online["online"] is True
    assert online["xr_age_ms"] == 12.0
    assert offline["online"] is False
