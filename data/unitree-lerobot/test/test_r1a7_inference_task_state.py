# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import uuid
import time
import pytest

from teleop.tools.barcode_task_event_bridge import TaskEventClient, TaskEventDeliveryError
from teleop.utils.task_events import TaskEventServer
from unitree_lerobot.r1a7_inference_task_state import InferenceTaskState


BARCODE = "6933949903337"


def make_event(event_id="item_accepted:test:scan-1", barcode=BARCODE):
    return {
        "event_id": event_id,
        "event": "item_accepted",
        "barcode": barcode,
        "transaction_id": "inference-test",
        "scan_id": "scan-1",
        "source_wall_time_ns": 123,
    }


def test_item_accepted_switches_task_and_requires_policy_reset_once():
    state = InferenceTaskState(expected_barcode=BARCODE, run_id="run-1")
    before = state.snapshot(last_consumed_generation=0)

    reply = state.handle_item_accepted(make_event())
    after = state.snapshot(last_consumed_generation=0)
    consumed = state.snapshot(last_consumed_generation=after.generation)

    assert before.phase == "scan"
    assert before.policy_reset_required is False
    assert reply["status"] == "ok"
    assert reply["policy_reset_required"] is True
    assert after.phase == "place"
    assert after.policy_reset_required is True
    assert consumed.policy_reset_required is False


def test_wrong_product_does_not_switch_inference_task():
    state = InferenceTaskState(expected_barcode=BARCODE, run_id="run-1")

    reply = state.handle_item_accepted(make_event(barcode="wrong"))

    assert reply["status"] == "error"
    assert state.snapshot().phase == "scan"
    assert state.snapshot().generation == 0


def test_product_agnostic_state_accepts_first_valid_barcode():
    state = InferenceTaskState(run_id="run-any-product")

    reply = state.handle_item_accepted(make_event(barcode="another-product"))

    assert reply["status"] == "ok"
    assert reply["accepted_barcode"] == "another-product"
    assert state.snapshot().phase == "place"


def test_product_agnostic_state_rejects_missing_barcode():
    state = InferenceTaskState(run_id="run-any-product")

    reply = state.handle_item_accepted(make_event(barcode=""))

    assert reply["status"] == "error"
    assert "missing a barcode" in reply["message"]
    assert state.snapshot().phase == "scan"


def test_later_scan_is_an_episode_duplicate_without_another_reset():
    state = InferenceTaskState(expected_barcode=BARCODE, run_id="run-1")
    state.handle_item_accepted(make_event())

    reply = state.handle_item_accepted(make_event(event_id="item_accepted:test:scan-2"))

    assert reply["status"] == "ok"
    assert reply["duplicate"] is True
    assert reply["duplicate_scope"] == "episode"
    assert reply["policy_reset_required"] is False
    assert state.snapshot().generation == 1


def test_next_cycle_requires_explicit_pause():
    state = InferenceTaskState(expected_barcode=BARCODE, run_id="run-1")
    with pytest.raises(RuntimeError):
        state.start_next_cycle()
    state.handle_item_accepted(make_event())
    with pytest.raises(RuntimeError):
        state.start_next_cycle()
    state.pause_for_next_cycle()
    assert state.handle_item_accepted(make_event(event_id="paused"))["status"] == "error"

    transition = state.start_next_cycle()
    snapshot = state.snapshot(last_consumed_generation=1)
    assert transition["previous_run_id"] == "run-1"
    assert snapshot.run_id != "run-1"
    assert snapshot.phase == "scan"
    assert snapshot.generation == 2
    assert snapshot.policy_reset_required
    assert snapshot.accepted_barcode is None

    assert state.handle_item_accepted(make_event(event_id="old"))["status"] == "error"
    fresh = make_event(event_id="fresh")
    fresh["source_wall_time_ns"] = time.time_ns()
    reply = state.handle_item_accepted(fresh)
    assert reply["status"] == "ok"
    assert reply["inference_run_id"] == snapshot.run_id
    assert state.snapshot().phase == "place"



def test_pause_during_scan_restarts_without_accepting_old_scan():
    state = InferenceTaskState(expected_barcode=BARCODE, run_id="run-1")
    paused = state.pause_for_next_cycle()
    assert paused["phase"] == "scan"
    assert state.handle_item_accepted(make_event(event_id="paused"))["status"] == "error"
    with pytest.raises(RuntimeError):
        state.pause_for_next_cycle()

    transition = state.start_next_cycle()
    assert transition["previous_run_id"] == "run-1"
    assert transition["run_id"] != "run-1"
    assert state.snapshot().phase == "scan"
    assert state.handle_item_accepted(make_event(event_id="stale"))["status"] == "error"
    fresh = make_event(event_id="fresh")
    fresh["source_wall_time_ns"] = time.time_ns()
    assert state.handle_item_accepted(fresh)["status"] == "ok"
    assert state.snapshot().phase == "place"


def test_operator_completion_requires_place_and_restarts_with_fresh_scan():
    state = InferenceTaskState(expected_barcode=BARCODE, run_id="run-1")
    with pytest.raises(RuntimeError, match="before a scan"):
        state.confirm_completion()

    state.handle_item_accepted(make_event())
    confirmed = state.confirm_completion()
    assert confirmed["phase"] == "place"
    assert confirmed["accepted_event_id"] == "item_accepted:test:scan-1"
    assert state.snapshot(last_consumed_generation=1).policy_reset_required
    assert state.handle_item_accepted(make_event(event_id="during-hold"))["status"] == "error"
    with pytest.raises(RuntimeError, match="already paused"):
        state.confirm_completion()

    transition = state.start_restarted_cycle()
    snapshot = state.snapshot(last_consumed_generation=confirmed["generation"])
    assert transition["previous_run_id"] == "run-1"
    assert snapshot.phase == "scan"
    assert snapshot.accepted_barcode is None
    assert snapshot.policy_reset_required
    assert state.handle_item_accepted(make_event(event_id="stale"))["status"] == "error"


def test_existing_scanner_client_switches_inference_receiver_protocol():
    state = InferenceTaskState(expected_barcode=BARCODE, run_id="run-1")
    endpoint = f"inproc://inference-task-{uuid.uuid4().hex}"
    server = TaskEventServer(endpoint, state.handle_item_accepted)
    server.start()
    try:
        reply = TaskEventClient(endpoint, attempts=1).send(make_event())
    finally:
        server.stop()

    assert reply["status"] == "ok"
    assert reply["phase"] == "place"
    assert reply["active_task"] == state.task_after
    assert reply["policy_reset_required"] is True
    assert state.snapshot().accepted_barcode == BARCODE


def test_existing_scanner_client_reports_wrong_product_rejection():
    state = InferenceTaskState(expected_barcode=BARCODE, run_id="run-1")
    endpoint = f"inproc://inference-task-{uuid.uuid4().hex}"
    server = TaskEventServer(endpoint, state.handle_item_accepted)
    server.start()
    try:
        try:
            TaskEventClient(endpoint, attempts=1).send(make_event(barcode="wrong"))
        except TaskEventDeliveryError as exc:
            assert "wrong product" in str(exc)
        else:
            raise AssertionError("wrong product was unexpectedly accepted")
    finally:
        server.stop()

    assert state.snapshot().phase == "scan"
