# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "unitree_lerobot"
    / "utils"
    / "build_r1a7_training_filter.py"
)
SPEC = importlib.util.spec_from_file_location("training_filter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_merge_ranges_merges_overlap_and_adjacency():
    assert MODULE.merge_ranges([(10, 20), (18, 30), (30, 35), (40, 45)]) == [
        (10, 35),
        (40, 45),
    ]


def test_build_filter_maps_raw_interval_to_converted_indices(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    source = tmp_path / "episode_0083"
    segments = [
        {
            "output_episode_index": 0,
            "source_episode_path": str(source),
            "start_frame": 0,
            "end_frame_exclusive": 100,
        },
        {
            "output_episode_index": 1,
            "source_episode_path": str(source),
            "start_frame": 100,
            "end_frame_exclusive": 140,
        },
    ]
    (root / "meta" / "source_segments.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in segments), encoding="utf-8"
    )
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_frames": 140}), encoding="utf-8"
    )
    annotations = {
        "action_chunk_size": 50,
        "episodes": {
            "episode_0083": {
                "ordinal": 70,
                "reviewed": True,
                "intervals": [
                    {
                        "excluded_training_start_frame": 75,
                        "excluded_training_end_frame_exclusive": 90,
                    }
                ],
            }
        },
    }
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")

    result = MODULE.build_filter(annotations_path, root)

    assert result["excluded_training_starts"] == 15
    assert result["allowed_training_starts"] == 125
    assert result["excluded_ranges"] == [
        {
            "output_episode_index": 0,
            "local_start": 75,
            "local_end_exclusive": 90,
            "dataset_start": 75,
            "dataset_end_exclusive": 90,
        }
    ]
