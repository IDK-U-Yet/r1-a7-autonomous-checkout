# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import json
from types import SimpleNamespace

import pytest

from teleop.tools.validate_r1_dataset import (
    configured_cameras,
    discover_episodes,
    duplicate_ratio_limit,
    validate_episode,
    vector_size,
)


def raw_sample():
    groups = {
        "left_arm": {"qpos": [0.0] * 7},
        "right_arm": {"qpos": [0.0] * 7},
        "waist": {"qpos": [0.0]},
        "left_ee": {"qpos": [0.0]},
        "right_ee": {"qpos": [0.0]},
        # Auxiliary whole-body telemetry is not part of the raw audit vector.
        "body": {"qpos": [0.0] * 35},
    }
    return {"states": groups, "actions": groups}


def test_r1_a7_raw_vector_is_arms_waist_and_grippers():
    sample = raw_sample()
    assert vector_size(sample, "states") == 17
    assert vector_size(sample, "actions") == 17


def test_waist_is_required_even_when_auxiliary_body_state_exists():
    sample = raw_sample()
    del sample["states"]["waist"]
    with pytest.raises(ValueError, match="states.waist.qpos"):
        vector_size(sample, "states")


def test_episode_camera_schema_supports_four_camera_resolutions():
    errors = []
    cameras = configured_cameras(
        {
            "camera_schema": {
                "global": {"color_key": "color_0", "shape": [360, 640, 3]},
                "left_wrist": {"color_key": "color_1", "shape": [480, 640, 3]},
                "right_wrist": {"color_key": "color_2", "shape": [480, 640, 3]},
                "scanner": {"color_key": "color_3", "shape": [480, 640, 3]},
            }
        },
        errors,
    )
    assert errors == []
    assert cameras["global"] == ("color_0", (360, 640, 3))
    assert cameras["scanner"] == ("color_3", (480, 640, 3))


def test_batch_discovery_ignores_rejected_episodes(tmp_path):
    accepted = tmp_path / "episode_0000" / "data.json"
    rejected = tmp_path / "rejected" / "episode_0001" / "data.json"
    accepted.parent.mkdir(parents=True)
    rejected.parent.mkdir(parents=True)
    accepted.write_text("{}", encoding="utf-8")
    rejected.write_text("{}", encoding="utf-8")
    assert discover_episodes(tmp_path) == [accepted]


def test_scanner_has_an_independent_duplicate_ratio_limit():
    args = SimpleNamespace(
        max_duplicate_ratio=0.05,
        max_scanner_duplicate_ratio=0.20,
    )
    assert duplicate_ratio_limit("global", args) == 0.05
    assert duplicate_ratio_limit("left_wrist", args) == 0.05
    assert duplicate_ratio_limit("right_wrist", args) == 0.05
    assert duplicate_ratio_limit("scanner", args) == 0.20


def test_task_switching_episode_requires_one_item_accepted_event(tmp_path):
    episode_dir = tmp_path / "episode_0000"
    episode_dir.mkdir()
    sample = raw_sample()
    sample.update(
        {
            "idx": 0,
            "colors": {},
            "metadata": {
                "record_monotonic_ns": 1,
                "active_task": "before",
                "camera_frames": {},
            },
        }
    )
    data_path = episode_dir / "data.json"
    data_path.write_text(
        json.dumps(
            {
                "info": {
                    "image": {"fps": 30},
                    "task_event_contract": {"event": "item_accepted"},
                },
                "data": [sample],
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        max_interval_factor=2.5,
        max_duplicate_ratio=0.05,
        max_scanner_duplicate_ratio=0.20,
        max_frame_age_ms=150.0,
        max_camera_skew_ms=100.0,
    )
    report = validate_episode(data_path, args)
    assert "events.jsonl is missing for a task-switching episode" in report["errors"]


def task_switching_report(tmp_path, event, tasks, record_times):
    episode_dir = tmp_path / "episode_0000"
    episode_dir.mkdir()
    samples = []
    for index, (task, record_time) in enumerate(zip(tasks, record_times)):
        sample = raw_sample()
        sample.update(
            {
                "idx": index,
                "colors": {},
                "metadata": {
                    "record_monotonic_ns": record_time,
                    "active_task": task,
                    "camera_frames": {},
                },
            }
        )
        samples.append(sample)
    data_path = episode_dir / "data.json"
    data_path.write_text(
        json.dumps(
            {
                "info": {
                    "image": {"fps": 30},
                    "task_event_contract": {"event": "item_accepted"},
                },
                "data": samples,
            }
        ),
        encoding="utf-8",
    )
    (episode_dir / "events.jsonl").write_text(
        json.dumps({"event": "item_accepted", **event}) + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        max_interval_factor=2.5,
        max_duplicate_ratio=0.05,
        max_scanner_duplicate_ratio=0.20,
        max_frame_age_ms=150.0,
        max_camera_skew_ms=100.0,
    )
    return validate_episode(data_path, args)


def test_task_transition_prefers_committed_queue_boundary(tmp_path):
    report = task_switching_report(
        tmp_path,
        event={
            "receiver_monotonic_ns": 150,
            "last_enqueued_frame_index": 1,
        },
        tasks=["before", "before", "after"],
        record_times=[100, 200, 300],
    )

    assert not any("active_task transition" in error for error in report["errors"])
    assert report["metrics"]["task_transition_boundary_source"] == (
        "last_enqueued_frame_index"
    )
    assert report["metrics"]["expected_task_transition_index"] == 2


def test_old_episode_falls_back_to_event_receive_time(tmp_path):
    report = task_switching_report(
        tmp_path,
        event={"receiver_monotonic_ns": 150},
        tasks=["before", "after", "after"],
        record_times=[100, 200, 300],
    )

    assert not any("active_task transition" in error for error in report["errors"])
    assert report["metrics"]["task_transition_boundary_source"] == (
        "receiver_monotonic_ns"
    )
    assert report["metrics"]["expected_task_transition_index"] == 1


def test_invalid_queue_boundary_is_not_silently_treated_as_old_data(tmp_path):
    report = task_switching_report(
        tmp_path,
        event={
            "receiver_monotonic_ns": 150,
            "last_enqueued_frame_index": "1",
        },
        tasks=["before", "after", "after"],
        record_times=[100, 200, 300],
    )

    assert "item_accepted event has invalid last_enqueued_frame_index" in report["errors"]
