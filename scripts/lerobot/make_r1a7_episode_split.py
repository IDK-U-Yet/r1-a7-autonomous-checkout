#!/usr/bin/env python3
"""Build a leakage-free train/val/test manifest grouped by original R1-A7 teaching episodes.

Reads a LeRobot dataset's ``meta/source_segments.jsonl`` (one line per converted task
segment, as written by ``convert_unitree_json_to_lerobot.py``) and assigns each
ORIGINAL raw episode — together with all of its scan/place segments — to exactly one
split, so no trajectory leaks across splits.

The manifest is deterministic for a given seed, source mapping and filter: it can be
regenerated on another machine and must hash identically.

Example:
    python3 make_r1a7_episode_split.py \
        --source-segments runtime/datasets/source_segments_oreo_v1_80.jsonl \
        --training-filter runtime/datasets/training_filter_oreo_v1_80.json \
        --output runtime/datasets/episode_split_oreo_v1_80.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

SEGMENT_KEYS = (
    "output_episode_index",
    "source_episode_index",
    "source_episode_path",
    "start_frame",
    "end_frame_exclusive",
    "task",
)

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.90, 0.05, 0.05)
DEFAULT_SEED = 20260922


def load_segments(path: Path) -> list[dict]:
    segments = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not segments:
        raise ValueError(f"{path} contains no segments")
    for index, segment in enumerate(segments):
        missing = [key for key in SEGMENT_KEYS if key not in segment]
        if missing:
            raise ValueError(f"Segment line {index} is missing keys {missing}")
        if segment["output_episode_index"] != index:
            raise ValueError(
                f"output_episode_index must be sequential from 0: line {index} has "
                f"{segment['output_episode_index']}"
            )
        if segment["end_frame_exclusive"] <= segment["start_frame"]:
            raise ValueError(f"Segment line {index} has an empty frame range")
    return segments


def normalize_source_groups(segments: list[dict]) -> list[dict]:
    """All exported intervention segments from a rollout share one split."""
    parents = {}
    for segment in segments:
        trial = segment.get("source_trial_id")
        if trial:
            parents[trial] = min(parents.get(trial, int(segment["source_episode_index"])),
                                 int(segment["source_episode_index"]))
    result = []
    for segment in segments:
        copy = dict(segment)
        trial = copy.get("source_trial_id")
        if trial:
            copy.setdefault("exported_source_episode_index", copy["source_episode_index"])
            copy["source_episode_index"] = parents[trial]
        result.append(copy)
    return result


def group_by_source(segments: list[dict]) -> dict[int, list[dict]]:
    segments = normalize_source_groups(segments)
    grouped: dict[int, list[dict]] = {}
    for segment in segments:
        grouped.setdefault(int(segment["source_episode_index"]), []).append(segment)
    for source_index, group in grouped.items():
        group.sort(key=lambda s: s["start_frame"])
        if len(group) > 2 and not all(s.get("source_trial_id") for s in group):
            raise ValueError(
                f"Source episode {source_index} produced {len(group)} task segments; "
                "the conversion contract allows at most one scan->place transition"
            )
    return grouped


def _largest_remainder_counts(total: int, fractions: tuple[float, float, float]) -> dict[str, int]:
    raw = [fraction * total for fraction in fractions]
    counts = [int(value) for value in raw]
    remainders = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    while sum(counts) < total:
        counts[remainders.pop(0)] += 1
    return dict(zip(SPLIT_NAMES, counts, strict=True))


def assign_splits(
    source_indices: list[int],
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, list[int]]:
    shuffled = sorted(source_indices)
    random.Random(seed).shuffle(shuffled)
    counts = _largest_remainder_counts(len(shuffled), fractions)
    if min(counts.values()) < 1:
        raise ValueError(f"Not enough source episodes ({len(shuffled)}) for fractions {fractions}")
    splits: dict[str, list[int]] = {}
    cursor = 0
    for name in SPLIT_NAMES:
        size = counts[name]
        splits[name] = shuffled[cursor : cursor + size]
        cursor += size
    return splits


def _allowed_mask(filter_payload: dict, expected_frames: int) -> list[bool]:
    if int(filter_payload["total_frames"]) != expected_frames:
        raise ValueError(
            f"Training filter total_frames={filter_payload['total_frames']} "
            f"does not match segment coverage={expected_frames}"
        )
    allowed = [True] * expected_frames
    for row in filter_payload["excluded_ranges"]:
        start, end = int(row["dataset_start"]), int(row["dataset_end_exclusive"])
        if start < 0 or end <= start or end > expected_frames:
            raise ValueError(f"Invalid training filter range [{start}, {end})")
        for frame in range(start, end):
            allowed[frame] = False
    return allowed


def build_manifest(
    segments: list[dict],
    fractions: tuple[float, float, float],
    seed: int,
    training_filter_path: Path | None,
) -> dict:
    segments = normalize_source_groups(segments)
    grouped = group_by_source(segments)
    source_indices = sorted(grouped)
    splits = assign_splits(source_indices, fractions, seed)

    total_frames = sum(s["end_frame_exclusive"] - s["start_frame"] for s in segments)
    output_indices = {s["output_episode_index"] for s in segments}
    if output_indices != set(range(len(segments))):
        raise ValueError("Output episode indices are not a complete 0..N-1 range")

    source_map: dict[str, dict] = {}
    for source_index in source_indices:
        group = grouped[source_index]
        split_name = next(name for name, sources in splits.items() if source_index in sources)
        source_map[str(source_index)] = {
            "split": split_name,
            "output_episode_indices": [s["output_episode_index"] for s in group],
            "source_episode_path": group[0]["source_episode_path"],
            "source_trial_id": group[0].get("source_trial_id"),
            "segments": [
                {
                    "output_episode_index": s["output_episode_index"],
                    "start_frame": s["start_frame"],
                    "end_frame_exclusive": s["end_frame_exclusive"],
                    "task": s["task"],
                }
                for s in group
            ],
        }

    split_summaries: dict[str, dict] = {}
    for name in SPLIT_NAMES:
        sources = splits[name]
        segs = [s for source in sources for s in grouped[source]]
        split_summaries[name] = {
            "source_episode_indices": sorted(sources),
            "output_episode_indices": sorted(s["output_episode_index"] for s in segs),
            "frames": sum(s["end_frame_exclusive"] - s["start_frame"] for s in segs),
        }

    frames_sum_ok = sum(split_summaries[name]["frames"] for name in SPLIT_NAMES) == total_frames
    if not frames_sum_ok:
        raise ValueError("Split frames do not sum to dataset total")

    manifest: dict = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "fractions": dict(zip(SPLIT_NAMES, fractions, strict=True)),
        "dataset": {
            "total_source_episodes": len(source_indices),
            "total_output_episodes": len(segments),
            "total_frames": total_frames,
        },
        "splits": split_summaries,
        "source_episode_map": source_map,
        "validation": {
            "coverage_complete": output_indices == set(range(len(segments))),
            "assigned_frames_sum_to_total": frames_sum_ok,
            "each_source_in_one_split": all(
                sum(source in splits[name] for name in SPLIT_NAMES) == 1 for source in source_indices
            ),
        },
    }

    if training_filter_path is not None:
        filter_payload = json.loads(training_filter_path.read_text(encoding="utf-8"))
        allowed = _allowed_mask(filter_payload, total_frames)
        allowed_per_split = {name: 0 for name in SPLIT_NAMES}
        split_of_source = {
            source_index: name for name, sources in splits.items() for source_index in sources
        }
        cursor = 0
        for segment in segments:
            length = segment["end_frame_exclusive"] - segment["start_frame"]
            name = split_of_source[int(segment["source_episode_index"])]
            for frame in range(cursor, cursor + length):
                if allowed[frame]:
                    allowed_per_split[name] += 1
            cursor += length
        actual_allowed = sum(allowed_per_split.values())
        expected_allowed = int(filter_payload["allowed_training_starts"])
        if actual_allowed != expected_allowed:
            raise ValueError(
                f"Filter cross-check failed: split masks allow {actual_allowed} starts "
                f"but the filter declares {expected_allowed}"
            )
        for name in SPLIT_NAMES:
            split_summaries[name]["allowed_training_starts"] = allowed_per_split[name]
        manifest["validation"]["filter_allowed_starts_match"] = True
        manifest["training_filter"] = {
            "path": str(training_filter_path),
            "sha256": hashlib.sha256(training_filter_path.read_bytes()).hexdigest(),
            "allowed_training_starts": expected_allowed,
        }

    if not all(manifest["validation"].values()):
        raise ValueError(f"Manifest validation failed: {manifest['validation']}")

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-segments",
        type=Path,
        required=True,
        help="Path to the dataset's meta/source_segments.jsonl",
    )
    parser.add_argument(
        "--training-filter",
        type=Path,
        help="Optional meta/training_filter.json for an allowed-starts cross-check",
    )
    parser.add_argument("--output", type=Path, required=True, help="Where to write the manifest JSON")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs=3,
        default=list(DEFAULT_FRACTIONS),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fractions = tuple(args.fractions)
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise SystemExit(f"Fractions must sum to 1.0, got {fractions}")
    segments = load_segments(args.source_segments)
    manifest = build_manifest(segments, fractions, args.seed, args.training_filter)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output}")
    for name in SPLIT_NAMES:
        summary = manifest["splits"][name]
        extra = summary.get("allowed_training_starts")
        starts = f", allowed_training_starts={extra}" if extra is not None else ""
        print(
            f"  {name:5s}: sources={len(summary['source_episode_indices']):3d} "
            f"episodes={len(summary['output_episode_indices']):3d} frames={summary['frames']}{starts}"
        )


if __name__ == "__main__":
    main()
