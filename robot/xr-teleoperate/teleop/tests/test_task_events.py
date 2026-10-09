# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import json
import time
import uuid

from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.task_events import TaskEventServer


BEFORE = (
    "Pick up a product from the blue tray on the left, present it to the scanner "
    "in the scanning area, and wait until the scan is accepted."
)
AFTER = "Place the scanned product in the black tray on the right."


def make_event(event_id="evt-1", barcode="6920999707541"):
    return {
        "event_id": event_id,
        "event": "item_accepted",
        "barcode": barcode,
        "transaction_id": "session-1",
        "order_line_id": "line-1",
        "scan_id": "scan-1",
        "source_wall_time_ns": None,
        "receiver_monotonic_ns": time.monotonic_ns(),
        "receiver_wall_time_ns": time.time_ns(),
    }


def test_episode_writer_switches_task_without_splitting_raw_episode(tmp_path):
    task_dir = tmp_path / "checkout"
    writer = EpisodeWriter(
        task_dir,
        task_goal="Checkout one product.",
        rerun_log=False,
        task_before_accept=BEFORE,
        task_after_accept=AFTER,
    )
    assert writer.create_episode()

    before_time = time.monotonic_ns()
    before_context = writer.task_context_for_timestamp(before_time)
    writer.add_item(colors={}, states={}, actions={}, metadata=before_context)

    event = make_event()
    reply = writer.accept_item_event(event)
    assert reply["status"] == "ok"
    after_context = writer.task_context_for_timestamp(event["receiver_monotonic_ns"])
    writer.add_item(colors={}, states={}, actions={}, metadata=after_context)

    duplicate = writer.accept_item_event(event)
    assert duplicate["status"] == "ok"
    assert duplicate["duplicate"] is True
    rejected = writer.accept_item_event(make_event("evt-2", "6941750501192"))
    assert rejected["status"] == "ok"
    assert rejected["duplicate"] is True
    assert rejected["duplicate_scope"] == "episode"
    assert rejected["accepted_event_id"] == "evt-1"
    assert rejected["accepted_barcode"] == event["barcode"]

    writer.save_episode()
    writer.close()
    episode_dir = task_dir / "episode_0000"
    raw = json.loads((episode_dir / "data.json").read_text(encoding="utf-8"))
    assert len(raw["data"]) == 2
    assert [item["metadata"]["active_task"] for item in raw["data"]] == [BEFORE, AFTER]
    events = (episode_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 1
    assert json.loads(events[0])["event_id"] == "evt-1"


def test_task_event_server_validates_token_and_deduplicates():
    calls = []

    def callback(event):
        calls.append(event)
        return {"status": "ok", "episode_id": 3}

    server = TaskEventServer(
        f"inproc://task-events-{uuid.uuid4().hex}",
        callback,
        token="secret",
    )
    message = {
        "event_id": "evt-1",
        "event": "item_accepted",
        "barcode": "6920999707541",
        "token": "secret",
    }
    try:
        assert server.handle_message({**message, "token": "wrong"})["status"] == "error"
        first = server.handle_message(message)
        second = server.handle_message(message)
    finally:
        server.stop()

    assert first["status"] == "ok"
    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert second["duplicate_scope"] == "event"
    assert len(calls) == 1


def test_episode_writer_numbers_across_rejected_and_empty_directories(tmp_path):
    task_dir = tmp_path / "batch"
    (task_dir / "rejected" / "episode_0007").mkdir(parents=True)
    writer = EpisodeWriter(task_dir, rerun_log=False)
    try:
        assert writer.create_episode()
        assert writer.episode_id == 8
    finally:
        writer.close()

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty_writer = EpisodeWriter(empty_dir, rerun_log=False)
    try:
        assert empty_writer.create_episode()
        assert empty_writer.episode_id == 0
    finally:
        empty_writer.close()
