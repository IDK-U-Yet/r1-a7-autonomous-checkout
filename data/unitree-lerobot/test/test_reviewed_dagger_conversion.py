# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import json

import pytest

from unitree_lerobot.utils.convert_unitree_json_to_lerobot import JsonDataset
from unitree_lerobot.utils.reviewed_dagger import MANIFEST, digest


def fixture_export(root, source="hold", valid=True):
    path = root / "episode_0000" / "data.json"
    path.parent.mkdir()
    samples = [{"metadata": {"active_task": "scan", "control_source": source,
                             "sample_valid": valid, "expert_action_valid": False,
                             "continuity_id": c, "source_trial_id": "original_trial"}}
               for c in [0, 0, 1]]
    path.write_text(json.dumps({"info": {"dagger_reviewed_export": True}, "data": samples}))
    manifest = {"schema_version": 1, "export_type": "human_reviewed_dagger",
                "include_reviewed_hold": True, "include_reviewed_transition": True,
                "episodes": [{"path": "episode_0000/data.json", "sha256": digest(path),
                              "frames": 3, "approval": {"review_status": "approved", "revision": 1}}]}
    (root / MANIFEST).write_text(json.dumps(manifest))
    return path


def test_reviewed_hold_and_transition_preserved_with_boundary(tmp_path):
    fixture_export(tmp_path)
    dataset = JsonDataset(tmp_path, "Unitree_R1_A7_Dex1_4Cam")
    assert len(dataset.episodes_data_cached[0]["data"]) == 3
    assert len(dataset) == 2
    assert [(s["start_frame"], s["end_frame_exclusive"]) for s in dataset.segment_refs] == [(0, 2), (2, 3)]
    assert dataset.episodes_data_cached[0]["data"][0]["metadata"]["control_source"] == "hold"
    assert all(s["source_trial_id"] == "original_trial" for s in dataset.segment_refs)


@pytest.mark.parametrize("source,valid", [("policy", True), ("human", False)])
def test_review_does_not_authorize_policy_or_invalid_samples(tmp_path, source, valid):
    fixture_export(tmp_path, source, valid)
    with pytest.raises(ValueError, match="valid HUMAN/HOLD"):
        JsonDataset(tmp_path, "Unitree_R1_A7_Dex1_4Cam")


def test_review_requires_manifest(tmp_path):
    fixture_export(tmp_path)
    (tmp_path / MANIFEST).unlink()
    with pytest.raises(ValueError, match="missing approval manifest"):
        JsonDataset(tmp_path, "Unitree_R1_A7_Dex1_4Cam")


def test_review_checksum_rejects_changed_episode(tmp_path):
    path = fixture_export(tmp_path)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        JsonDataset(tmp_path, "Unitree_R1_A7_Dex1_4Cam")


def test_review_rejects_extra_episode(tmp_path):
    path = fixture_export(tmp_path)
    extra = tmp_path / "episode_0001"
    extra.mkdir()
    (extra / "data.json").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="episode set mismatch"):
        JsonDataset(tmp_path, "Unitree_R1_A7_Dex1_4Cam")


@pytest.mark.parametrize("key", ["include_reviewed_hold", "include_reviewed_transition"])
def test_explicit_authorization_required(tmp_path, key):
    fixture_export(tmp_path)
    path = tmp_path / MANIFEST
    manifest = json.loads(path.read_text())
    manifest[key] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="explicit reviewed authorization"):
        JsonDataset(tmp_path, "Unitree_R1_A7_Dex1_4Cam")
