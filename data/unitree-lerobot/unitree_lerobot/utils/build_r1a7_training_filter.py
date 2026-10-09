#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Map raw recovery annotations onto LeRobot v3 training sample indices.

The output is an audit manifest. It does not modify raw data, parquet files, or videos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if start < 0 or end <= start:
            raise ValueError(f"invalid exclusion range [{start}, {end})")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def build_filter(annotations_path: Path, dataset_root: Path) -> dict:
    annotations = read_json(annotations_path)
    segments_path = dataset_root / "meta" / "source_segments.jsonl"
    info_path = dataset_root / "meta" / "info.json"
    segments = sorted(read_jsonl(segments_path), key=lambda row: row["output_episode_index"])
    info = read_json(info_path)

    episode_starts: dict[int, int] = {}
    cursor = 0
    source_segments: dict[str, list[dict]] = {}
    for expected_index, segment in enumerate(segments):
        output_index = int(segment["output_episode_index"])
        if output_index != expected_index:
            raise ValueError("source_segments output episode indices are not contiguous")
        episode_starts[output_index] = cursor
        length = int(segment["end_frame_exclusive"]) - int(segment["start_frame"])
        if length <= 0:
            raise ValueError(f"empty converted segment: {segment}")
        cursor += length
        source_name = Path(segment["source_episode_path"]).name
        source_segments.setdefault(source_name, []).append(segment)
    if cursor != int(info["total_frames"]):
        raise ValueError(f"segment frames {cursor} != dataset total_frames {info['total_frames']}")

    raw_ranges_by_output: dict[int, list[tuple[int, int]]] = {}
    provenance: list[dict] = []
    for source_name, annotation in annotations.get("episodes", {}).items():
        if not annotation.get("reviewed", False):
            raise ValueError(f"annotation is not reviewed: {source_name}")
        for interval_index, interval in enumerate(annotation.get("intervals", []), start=1):
            raw_start = int(interval["excluded_training_start_frame"])
            raw_end = int(interval["excluded_training_end_frame_exclusive"])
            candidates = [
                segment
                for segment in source_segments.get(source_name, [])
                if int(segment["start_frame"]) <= raw_start < int(segment["end_frame_exclusive"])
                and raw_end <= int(segment["end_frame_exclusive"])
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"{source_name} interval {interval_index} maps to {len(candidates)} segments"
                )
            segment = candidates[0]
            output_index = int(segment["output_episode_index"])
            local_start = raw_start - int(segment["start_frame"])
            local_end = raw_end - int(segment["start_frame"])
            raw_ranges_by_output.setdefault(output_index, []).append((local_start, local_end))
            provenance.append(
                {
                    "source_episode": source_name,
                    "accepted_ordinal": int(annotation["ordinal"]),
                    "interval_index": interval_index,
                    "output_episode_index": output_index,
                    "local_range": [local_start, local_end],
                    "global_range": [
                        episode_starts[output_index] + local_start,
                        episode_starts[output_index] + local_end,
                    ],
                }
            )

    excluded_ranges = []
    excluded_count = 0
    for output_index, ranges in sorted(raw_ranges_by_output.items()):
        for local_start, local_end in merge_ranges(ranges):
            global_start = episode_starts[output_index] + local_start
            global_end = episode_starts[output_index] + local_end
            excluded_count += global_end - global_start
            excluded_ranges.append(
                {
                    "output_episode_index": output_index,
                    "local_start": local_start,
                    "local_end_exclusive": local_end,
                    "dataset_start": global_start,
                    "dataset_end_exclusive": global_end,
                }
            )

    return {
        "version": 1,
        "dataset_root": str(dataset_root.resolve()),
        "dataset_name": dataset_root.name,
        "annotations_path": str(annotations_path.resolve()),
        "annotations_sha256": hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
        "action_chunk_size": int(annotations["action_chunk_size"]),
        "total_frames": cursor,
        "excluded_training_starts": excluded_count,
        "allowed_training_starts": cursor - excluded_count,
        "excluded_ranges": excluded_ranges,
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or args.dataset_root / "meta" / "training_filter.json"
    payload = build_filter(args.annotations, args.dataset_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(output)
    print(f"Training filter: {output}")
    print(
        f"Allowed {payload['allowed_training_starts']}/{payload['total_frames']} starts; "
        f"excluded {payload['excluded_training_starts']} across "
        f"{len(payload['excluded_ranges'])} merged range(s)."
    )


if __name__ == "__main__":
    main()
