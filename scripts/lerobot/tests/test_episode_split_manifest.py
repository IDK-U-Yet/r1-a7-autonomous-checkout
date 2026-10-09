from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
SPLIT_PATH = SCRIPT_DIR / "make_r1a7_episode_split.py"


def _load_split_module():
    spec = importlib.util.spec_from_file_location("make_r1a7_episode_split", SPLIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SPLIT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


split = _load_split_module()


def _make_segments(source_count: int, frames_per_source: int = 100) -> list[dict]:
    segments = []
    output_index = 0
    for source_index in range(source_count):
        scan_frames = frames_per_source // 2
        for task, start, end in (
            ("scan task", 0, scan_frames),
            ("place task", scan_frames, frames_per_source),
        ):
            segments.append(
                {
                    "output_episode_index": output_index,
                    "source_episode_index": source_index,
                    "source_episode_path": f"/raw/episode_{source_index:04d}",
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "task": task,
                }
            )
            output_index += 1
    return segments


class EpisodeSplitManifestTest(unittest.TestCase):
    def test_dagger_siblings_share_parent_trial_split(self):
        segments = _make_segments(20)
        for segment in segments:
            segment["source_trial_id"] = "trial_" + str(segment["source_episode_index"] // 2)
        manifest = split.build_manifest(segments, (0.8, 0.1, 0.1), seed=7, training_filter_path=None)
        assert manifest["dataset"]["total_source_episodes"] == 10
        for entry in manifest["source_episode_map"].values():
            assert len(entry["output_episode_indices"]) == 4
            assert entry["source_trial_id"].startswith("trial_")

    def test_each_source_lands_in_exactly_one_split(self):
        segments = _make_segments(20)
        manifest = split.build_manifest(segments, (0.8, 0.1, 0.1), seed=7, training_filter_path=None)
        for source_index, entry in manifest["source_episode_map"].items():
            outputs = entry["output_episode_indices"]
            self.assertEqual(len(outputs), 2)
            split_names = {name for name in split.SPLIT_NAMES if int(source_index) in [s for s in manifest["splits"][name]["source_episode_indices"]]}
            self.assertEqual(len(split_names), 1)
        self.assertTrue(all(manifest["validation"].values()))

    def test_scan_and_place_stay_together(self):
        segments = _make_segments(10)
        manifest = split.build_manifest(segments, (0.7, 0.2, 0.1), seed=3, training_filter_path=None)
        for entry in manifest["source_episode_map"].values():
            output_indices = entry["output_episode_indices"]
            self.assertEqual(output_indices, [output_indices[0], output_indices[0] + 1])

    def test_frames_and_episode_counts_sum_to_totals(self):
        segments = _make_segments(80)
        manifest = split.build_manifest(segments, split.DEFAULT_FRACTIONS, seed=split.DEFAULT_SEED, training_filter_path=None)
        self.assertEqual(sum(s["frames"] for s in manifest["splits"].values()), 80 * 100)
        self.assertEqual(sum(len(s["output_episode_indices"]) for s in manifest["splits"].values()), 160)
        counts = {name: len(manifest["splits"][name]["source_episode_indices"]) for name in split.SPLIT_NAMES}
        self.assertEqual(counts, {"train": 72, "val": 4, "test": 4})

    def test_manifest_is_deterministic_for_fixed_seed(self):
        segments = _make_segments(30)
        first = split.build_manifest(segments, split.DEFAULT_FRACTIONS, seed=11, training_filter_path=None)
        second = split.build_manifest(segments, split.DEFAULT_FRACTIONS, seed=11, training_filter_path=None)
        first["created_utc"] = second["created_utc"] = ""
        self.assertEqual(first, second)

    def test_filter_cross_check_counts_allowed_starts_per_split(self):
        segments = _make_segments(4, frames_per_source=50)
        total_frames = 4 * 50
        filter_payload = {
            "total_frames": total_frames,
            "allowed_training_starts": total_frames - 10,
            "excluded_ranges": [
                {"dataset_start": 5, "dataset_end_exclusive": 15},
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
            json.dump(filter_payload, stream)
            filter_path = Path(stream.name)
        try:
            manifest = split.build_manifest(segments, (0.5, 0.25, 0.25), seed=5, training_filter_path=filter_path)
        finally:
            filter_path.unlink()
        allowed_total = sum(s["allowed_training_starts"] for s in manifest["splits"].values())
        self.assertEqual(allowed_total, total_frames - 10)
        self.assertTrue(manifest["validation"]["filter_allowed_starts_match"])

    def test_filter_total_mismatch_rejected(self):
        segments = _make_segments(4, frames_per_source=50)
        filter_payload = {
            "total_frames": 12345,
            "allowed_training_starts": 12345,
            "excluded_ranges": [],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
            json.dump(filter_payload, stream)
            filter_path = Path(stream.name)
        try:
            with self.assertRaises(ValueError):
                split.build_manifest(segments, (0.5, 0.25, 0.25), seed=5, training_filter_path=filter_path)
        finally:
            filter_path.unlink()

    def test_more_than_two_segments_rejected(self):
        segments = _make_segments(3)
        extra = dict(segments[0])
        extra["output_episode_index"] = len(segments)
        extra["task"] = "extra"
        segments.append(extra)
        with self.assertRaises(ValueError):
            split.build_manifest(segments, split.DEFAULT_FRACTIONS, seed=1, training_filter_path=None)


if __name__ == "__main__":
    unittest.main()
