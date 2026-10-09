# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import json
import shutil
from pathlib import Path

import pytest

from teleop.utils.collection_batch import CollectionBatch, CollectionBatchError


def write_plan(
    path,
    *,
    reset_seconds=30,
    second_product=True,
    selection_mode=None,
):
    products = [
        {
            "barcode": "111",
            "sku": "SKU-1",
            "name": "Product one",
            "target_episodes": 2,
        }
    ]
    if second_product:
        products.append(
            {
                "barcode": "222",
                "sku": "SKU-2",
                "name": "Product two",
                "target_episodes": 1,
            }
        )
    plan = {
        "version": 1,
        "batch_id": "test-batch",
        "reset_seconds": reset_seconds,
        "products": products,
    }
    if selection_mode is not None:
        plan["selection_mode"] = selection_mode
    path.write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    return path


def create_episode_dir(task_dir, episode_id):
    episode_dir = task_dir / f"episode_{episode_id:04d}"
    episode_dir.mkdir(parents=True)
    (episode_dir / "data.json").write_text("{}", encoding="utf-8")
    return episode_dir


def accept_attempt(batch, episode_id, barcode):
    episode_dir = create_episode_dir(batch.task_dir, episode_id)
    batch.start_attempt(episode_id, episode_dir)
    batch.validate_barcode(barcode)
    batch.mark_item_accepted(f"event-{episode_id}")
    batch.mark_saving()
    return batch.finish_save({"event_id": f"event-{episode_id}", "barcode": barcode})


def test_tracks_products_allows_early_start_and_operator_retry(tmp_path):
    now = [100.0]
    batch = CollectionBatch(
        tmp_path / "batch",
        write_plan(tmp_path / "plan.json"),
        wall_clock=lambda: now[0],
    )

    accepted, first = accept_attempt(batch, 0, "111")
    assert accepted is True
    assert batch.status == "resetting"
    assert batch.reset_remaining_seconds() == pytest.approx(30.0)
    assert batch.progress_summary()["accepted"] == 1

    second_dir = create_episode_dir(batch.task_dir, 1)
    batch.start_attempt(1, second_dir)
    assert batch.status == "recording"
    assert batch.state["active_attempt"]["ordinal"] == 2
    batch.mark_item_accepted("event-1")
    batch.mark_saving()
    batch.finish_save({"event_id": "event-1", "barcode": "111"})

    rejected = batch.reject_last_accepted()
    assert rejected["episode_id"] == 1
    assert not second_dir.exists()
    assert (batch.rejected_dir / "episode_0001").is_dir()
    assert batch.accepted_count == 1
    assert batch.next_slot().barcode == "111"
    assert batch.next_slot().ordinal == 2


def test_rejects_wrong_or_missing_barcode_without_counting_episode(tmp_path):
    batch = CollectionBatch(
        tmp_path / "batch",
        write_plan(tmp_path / "plan.json"),
    )
    episode_dir = create_episode_dir(batch.task_dir, 0)
    batch.start_attempt(0, episode_dir)
    with pytest.raises(CollectionBatchError, match="wrong product"):
        batch.validate_barcode("222")
    batch.mark_saving()
    accepted, rejected = batch.finish_save(None)

    assert accepted is False
    assert rejected["rejection_reason"] == "missing_item_accepted"
    assert batch.accepted_count == 0
    assert (batch.rejected_dir / "episode_0000").is_dir()
    with pytest.raises(CollectionBatchError, match="no accepted episode"):
        batch.reject_last_accepted()


def test_restart_rejects_interrupted_attempt_and_resumes_slot(tmp_path):
    plan = write_plan(tmp_path / "plan.json")
    task_dir = tmp_path / "batch"
    batch = CollectionBatch(task_dir, plan)
    episode_dir = create_episode_dir(task_dir, 0)
    batch.start_attempt(0, episode_dir)

    resumed = CollectionBatch(task_dir, plan)

    assert resumed.resumed is True
    assert resumed.status == "ready"
    assert resumed.accepted_count == 0
    assert resumed.rejected_count == 1
    assert (task_dir / "rejected" / "episode_0000").is_dir()
    assert resumed.next_slot().ordinal == 1


def test_abort_active_attempt_rejects_without_advancing_slot(tmp_path):
    batch = CollectionBatch(
        tmp_path / "batch",
        write_plan(tmp_path / "plan.json"),
    )
    episode_dir = create_episode_dir(batch.task_dir, 0)
    batch.start_attempt(0, episode_dir)
    batch.mark_item_accepted("event-before-operator-abort")

    batch.mark_aborting()
    rejected = batch.finish_abort()

    assert rejected["episode_id"] == 0
    assert rejected["rejection_reason"] == "operator_aborted_during_recording"
    assert batch.accepted_count == 0
    assert batch.rejected_count == 1
    assert batch.next_slot().barcode == "111"
    assert batch.next_slot().ordinal == 1
    assert not episode_dir.exists()
    assert (batch.rejected_dir / "episode_0000").is_dir()


def test_restart_recovers_attempt_interrupted_while_aborting(tmp_path):
    plan = write_plan(tmp_path / "plan.json")
    task_dir = tmp_path / "batch"
    batch = CollectionBatch(task_dir, plan)
    episode_dir = create_episode_dir(task_dir, 0)
    batch.start_attempt(0, episode_dir)
    batch.mark_aborting()

    resumed = CollectionBatch(task_dir, plan)

    assert resumed.status == "ready"
    assert resumed.accepted_count == 0
    assert resumed.rejected_count == 1
    assert resumed.next_slot().ordinal == 1
    assert (task_dir / "rejected" / "episode_0000").is_dir()


def test_flexible_plan_uses_first_accepted_barcode_instead_of_product_order(tmp_path):
    batch = CollectionBatch(
        tmp_path / "batch",
        write_plan(
            tmp_path / "plan.json",
            selection_mode="flexible",
        ),
    )
    episode_dir = create_episode_dir(batch.task_dir, 0)
    attempt = batch.start_attempt(0, episode_dir)

    assert attempt["barcode"] is None
    selected = batch.validate_barcode("222")
    assert selected["barcode"] == "222"
    assert selected["name"] == "Product two"
    assert selected["ordinal"] == 1
    batch.mark_item_accepted("event-0")
    batch.mark_saving()
    accepted, saved = batch.finish_save({"event_id": "event-0", "barcode": "222"})

    assert accepted is True
    assert saved["barcode"] == "222"
    assert batch.accepted_count == 1
    assert batch.next_slot().barcode == "111"


def test_flexible_plan_rejects_completed_quota_and_missing_scan(tmp_path):
    plan = write_plan(tmp_path / "plan.json", selection_mode="flexible")
    batch = CollectionBatch(tmp_path / "batch", plan)
    accept_attempt(batch, 0, "222")

    episode_dir = create_episode_dir(batch.task_dir, 1)
    batch.start_attempt(1, episode_dir)
    with pytest.raises(CollectionBatchError, match="quota is complete"):
        batch.validate_barcode("222")
    batch.mark_saving()
    accepted, rejected = batch.finish_save(None)

    assert accepted is False
    assert rejected["rejection_reason"] == "missing_item_accepted"
    assert batch.accepted_count == 1


def test_reset_timer_and_complete_state_survive_restart(tmp_path):
    now = [10.0]
    plan = write_plan(tmp_path / "plan.json", reset_seconds=5, second_product=False)
    batch = CollectionBatch(tmp_path / "batch", plan, wall_clock=lambda: now[0])
    accept_attempt(batch, 0, "111")
    accept_attempt(batch, 1, "111")
    assert batch.status == "resetting"

    now[0] = 16.0
    resumed = CollectionBatch(
        tmp_path / "batch", plan, wall_clock=lambda: now[0]
    )
    assert resumed.status == "complete"
    assert resumed.accepted_count == 2


def test_refuses_legacy_episodes_without_batch_state(tmp_path):
    task_dir = tmp_path / "batch"
    create_episode_dir(task_dir, 0)
    with pytest.raises(CollectionBatchError, match="archive the old batch"):
        CollectionBatch(task_dir, write_plan(tmp_path / "plan.json"))


def test_resume_repairs_crash_between_rejected_move_and_state_commit(tmp_path):
    plan = write_plan(tmp_path / "plan.json")
    task_dir = tmp_path / "batch"
    batch = CollectionBatch(task_dir, plan)
    accept_attempt(batch, 0, "111")
    source = task_dir / "episode_0000"
    rejected = task_dir / "rejected" / "episode_0000"
    rejected.parent.mkdir()
    shutil.move(source, rejected)

    resumed = CollectionBatch(task_dir, plan)

    assert resumed.accepted_count == 0
    assert resumed.rejected_count == 1
    assert resumed.state["rejected"][0]["episode_path"] == str(rejected)
    assert resumed.next_slot().ordinal == 1
