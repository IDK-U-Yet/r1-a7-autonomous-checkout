# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import numpy as np

from unitree_lerobot.utils.constants import R1_A7_DEX1_4CAM_CONFIG
from unitree_lerobot.utils.convert_unitree_json_to_lerobot import (
    JsonDataset,
    assert_lerobot_version,
    discover_episode_jsons,
)


def test_converter_ignores_managed_rejected_directory(tmp_path):
    accepted = tmp_path / "episode_0000" / "data.json"
    rejected = tmp_path / "rejected" / "episode_0001" / "data.json"
    accepted.parent.mkdir(parents=True)
    rejected.parent.mkdir(parents=True)
    accepted.write_text("{}", encoding="utf-8")
    rejected.write_text("{}", encoding="utf-8")

    assert discover_episode_jsons(tmp_path) == [accepted]


def test_converter_does_not_fall_back_to_rejected_only_data(tmp_path):
    rejected = tmp_path / "rejected" / "episode_0001" / "data.json"
    rejected.parent.mkdir(parents=True)
    rejected.write_text("{}", encoding="utf-8")

    assert discover_episode_jsons(tmp_path) == []


def test_converter_requires_lerobot_062():
    assert_lerobot_version("0.6.2")


def test_converter_refuses_dagger_audit_rollout(tmp_path):
    import json
    import pytest
    episode = tmp_path / "dagger_trial"
    episode.mkdir()
    (episode / "data.json").write_text(json.dumps({"info": {"dagger_audit_only": True},
        "data": [{"metadata": {"active_task": "scan", "control_source": "policy"}}]}))
    with pytest.raises(ValueError, match="audit rollout"):
        JsonDataset(tmp_path, "R1_A7_DEX1_4CAM")


def test_converter_rejects_legacy_lerobot_version():
    import pytest

    with pytest.raises(RuntimeError, match="requires LeRobot 0.6.2"):
        assert_lerobot_version("0.4.1")


def test_r1a7_converter_excludes_raw_waist_from_model_vector():
    sample = {
        "states": {
            "left_arm": {"qpos": list(range(7))},
            "right_arm": {"qpos": list(range(7, 14))},
            "waist": {"qpos": [0.25]},
            "left_ee": {"qpos": [1.5]},
            "right_ee": {"qpos": [2.5]},
        },
        "actions": {
            "left_arm": {"qpos": list(range(7))},
            "right_arm": {"qpos": list(range(7, 14))},
            "waist": {"qpos": [0.3]},
            "left_ee": {"qpos": [3.5]},
            "right_ee": {"qpos": [4.5]},
        },
    }
    episode = {"data": [sample]}
    loader = JsonDataset.__new__(JsonDataset)

    state = loader._extract_data(
        episode, "states", R1_A7_DEX1_4CAM_CONFIG.json_state_data_name
    )
    action = loader._extract_data(
        episode, "actions", R1_A7_DEX1_4CAM_CONFIG.json_action_data_name
    )

    assert state.shape == (1, 16)
    assert action.shape == (1, 16)
    np.testing.assert_array_equal(state[0, 14:], [1.5, 2.5])
    np.testing.assert_array_equal(action[0, 14:], [3.5, 4.5])


def test_converter_applies_post_capture_episode_annotation(tmp_path):
    import json

    raw_dir = tmp_path / "raw"
    episode_dir = raw_dir / "episode_0007"
    episode_dir.mkdir(parents=True)
    samples = [
        {"metadata": {"active_task": "old scan task"}},
        {"metadata": {"active_task": "place the scanned product"}},
    ]
    (episode_dir / "data.json").write_text(
        json.dumps({"data": samples}), encoding="utf-8"
    )
    manifest = tmp_path / "episode_annotations.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "episode_id": "episode_0007",
                "episode_type": "standard_pick",
                "task_before_accept": "Pick from the blue tray, scan, and wait for acceptance.",
                "labels": {"initial_orientation": "O1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = JsonDataset(
        raw_dir, "Unitree_R1_A7_Dex1_4Cam", manifest
    )

    assert len(dataset.segment_refs) == 2
    assert dataset.segment_refs[0]["task"] == (
        "Pick from the blue tray, scan, and wait for acceptance."
    )
    assert dataset.segment_refs[1]["task"] == "place the scanned product"
    assert dataset.segment_refs[0]["episode_annotation"]["episode_type"] == "standard_pick"
    assert dataset.segment_refs[1]["episode_annotation"]["labels"]["initial_orientation"] == (
        "O1"
    )


def test_converter_requires_annotation_for_every_discovered_episode(tmp_path):
    import pytest

    episode_dir = tmp_path / "episode_0000"
    episode_dir.mkdir()
    (episode_dir / "data.json").write_text(
        '{"data":[{"metadata":{"active_task":"scan"}},'
        '{"metadata":{"active_task":"place"}}]}',
        encoding="utf-8",
    )
    manifest = tmp_path / "annotations.jsonl"
    manifest.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is incomplete"):
        JsonDataset(tmp_path, "Unitree_R1_A7_Dex1_4Cam", manifest)
