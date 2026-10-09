# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "unitree_lerobot"
    / "utils"
    / "annotate_r1a7_recovery_intervals.py"
)
SPEC = importlib.util.spec_from_file_location("recovery_annotator", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exclusion_interval_expands_for_future_action_chunk():
    assert MODULE.exclusion_interval(100, 120, 50) == (51, 120)
    assert MODULE.exclusion_interval(20, 30, 50) == (0, 30)
    assert MODULE.exclusion_interval(300, 320, 50, segment_start=262) == (262, 320)


@pytest.mark.parametrize(
    "failure,recovery,chunk,segment_start",
    [(-1, 10, 50, 0), (10, 10, 50, 0), (10, 9, 50, 0), (1, 2, 0, 0), (100, 120, 50, 110)],
)
def test_exclusion_interval_rejects_invalid_values(failure, recovery, chunk, segment_start):
    with pytest.raises(ValueError):
        MODULE.exclusion_interval(failure, recovery, chunk, segment_start)


def test_load_review_episodes_uses_accepted_order_not_directory_sequence(tmp_path):
    raw_dir = tmp_path / "runtime" / "datasets" / "batch"
    raw_dir.mkdir(parents=True)
    accepted = []
    for ordinal, episode_id in enumerate((4, 9), start=1):
        episode_dir = raw_dir / f"episode_{episode_id:04d}"
        episode_dir.mkdir()
        frames = [{"idx": idx, "colors": {}} for idx in range(6)]
        (episode_dir / "data.json").write_text(json.dumps({"data": frames}), encoding="utf-8")
        event = {"last_enqueued_frame_index": 3}
        (episode_dir / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
        accepted.append(
            {
                "ordinal": ordinal,
                "episode_id": episode_id,
                "episode_path": str(episode_dir),
            }
        )
    (raw_dir / "batch_state.json").write_text(
        json.dumps({"batch_id": "batch", "accepted": accepted}), encoding="utf-8"
    )

    batch_id, episodes = MODULE.load_review_episodes(raw_dir, last_accepted=1)

    assert batch_id == "batch"
    assert len(episodes) == 1
    assert episodes[0].accepted_index == 1
    assert episodes[0].episode_id == 9
    assert episodes[0].scan_end_exclusive == 4


def test_load_review_episodes_selects_inclusive_accepted_ordinal_range(tmp_path):
    raw_dir = tmp_path / "runtime" / "datasets" / "batch"
    raw_dir.mkdir(parents=True)
    accepted = []
    for ordinal, episode_id in enumerate((4, 9, 14), start=60):
        episode_dir = raw_dir / f"episode_{episode_id:04d}"
        episode_dir.mkdir()
        (episode_dir / "data.json").write_text(
            json.dumps({"data": [{"idx": idx, "colors": {}} for idx in range(6)]}),
            encoding="utf-8",
        )
        (episode_dir / "events.jsonl").write_text(
            json.dumps({"last_enqueued_frame_index": 3}) + "\n", encoding="utf-8"
        )
        accepted.append({"ordinal": ordinal, "episode_id": episode_id, "episode_path": str(episode_dir)})
    (raw_dir / "batch_state.json").write_text(
        json.dumps({"batch_id": "batch", "accepted": accepted}), encoding="utf-8"
    )

    _, episodes = MODULE.load_review_episodes(raw_dir, ordinal_start=61, ordinal_end=62)

    assert [episode.ordinal for episode in episodes] == [61, 62]
    assert [episode.episode_id for episode in episodes] == [9, 14]


def test_review_type_distinguishes_normal_from_recovery():
    assert MODULE.reviewed_trajectory_type([]) == "reviewed_normal"
    assert MODULE.reviewed_trajectory_type([{"failure_action_start_frame": 10}]) == (
        "intentional_error_recovery"
    )
