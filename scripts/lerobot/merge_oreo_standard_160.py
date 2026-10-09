#!/usr/bin/env python3
"""Merge the selected v2/80 and completed v3/80; preserve source provenance."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.aggregate import aggregate_datasets


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_source(root, require_annotations):
    info = read_json(root / "meta/info.json")
    assert info["codebase_version"] == "v3.0", root
    assert (info["total_episodes"], info["fps"]) == (160, 15), root
    for key in ("observation.state", "action"):
        assert info["features"][key]["shape"] == [16], (root, key)
    assert sum(v["dtype"] == "video" for v in info["features"].values()) == 4
    records = [json.loads(s) for s in (root / "meta/source_segments.jsonl").read_text().splitlines()]
    assert [r["output_episode_index"] for r in records] == list(range(160))
    counts = Counter(r["source_episode_path"] for r in records)
    assert len(counts) == 80 and set(counts.values()) == {2}
    if require_annotations:
        assert all(r["episode_annotation"]["episode_type"] == "standard_pick" for r in records)
        selection = read_json(root / "meta/source_selection.json")
        assert selection["selection"] == {
            "episode_type": "standard_pick", "collection_ordinal_inclusive": [1, 80]
        }
    data_files = sorted((root / "data").rglob("*.parquet"))
    assert sum(pq.ParquetFile(p).metadata.num_rows for p in data_files) == info["total_frames"]
    return info, records


def merge(old, new, output):
    assert not output.exists(), f"Output already exists: {output}"
    old_info, old_records = inspect_source(old, True)
    new_info, new_records = inspect_source(new, False)
    assert {r["task"] for r in old_records} == {r["task"] for r in new_records}
    assert len({r["task"] for r in old_records}) == 2
    # Keep MP4 shards intact; the official merger remaps video and parquet indices.
    aggregate_datasets(
        repo_ids=[f"local/{old.name}", f"local/{new.name}"],
        aggr_repo_id=f"local/{output.name}", roots=[old, new], aggr_root=output,
        concatenate_videos=False, concatenate_data=False,
    )
    info = read_json(output / "meta/info.json")
    expected_frames = old_info["total_frames"] + new_info["total_frames"]
    assert (info["total_episodes"], info["total_frames"]) == (320, expected_frames)
    info["splits"] = {"train": "0:320"}
    (output / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")

    tables = [pq.read_table(p) for p in sorted((output / "data").rglob("*.parquet"))]
    assert sum(len(t) for t in tables) == expected_frames
    episode_ids = np.concatenate([np.asarray(t["episode_index"]) for t in tables])
    frame_ids = np.concatenate([np.asarray(t["index"]) for t in tables])
    assert np.array_equal(np.unique(episode_ids), np.arange(320))
    assert np.array_equal(np.sort(frame_ids), np.arange(expected_frames))
    stats = read_json(output / "meta/stats.json")
    # Recompute exact model normalization; merged quantile summaries are not exact.
    for key in ("observation.state", "action"):
        values = np.concatenate([np.asarray(t[key].to_pylist(), dtype=np.float64) for t in tables])
        assert values.shape == (expected_frames, 16) and np.isfinite(values).all()
        stats[key] = {
            "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "count": [expected_frames],
            **{f"q{int(q * 100):02d}": np.quantile(values, q, axis=0).tolist()
               for q in (0.01, 0.10, 0.50, 0.90, 0.99)},
        }
    (output / "meta/stats.json").write_text(json.dumps(stats, indent=4) + "\n")

    episode_rows = []
    for path in sorted((output / "meta/episodes").rglob("*.parquet")):
        episode_rows.extend(pq.read_table(path).to_pylist())
    assert sorted(r["episode_index"] for r in episode_rows) == list(range(320))
    for row in episode_rows:
        for key, feature in info["features"].items():
            if feature["dtype"] != "video":
                continue
            video = output / info["video_path"].format(
                video_key=key, chunk_index=row[f"videos/{key}/chunk_index"],
                file_index=row[f"videos/{key}/file_index"],
            )
            assert video.is_file() and video.stat().st_size > 0, video
            assert row[f"videos/{key}/from_timestamp"] < row[f"videos/{key}/to_timestamp"]

    provenance = []
    for batch, records, offset in [("v2_standard_80", old_records, 0), ("v3_standard_80", new_records, 160)]:
        for record in records:
            provenance.append({**record, "source_batch": batch, "episode_type": "standard_pick",
                               "source_output_episode_index": record["output_episode_index"],
                               "output_episode_index": record["output_episode_index"] + offset})
    assert len({r["source_episode_path"] for r in provenance}) == 160
    (output / "meta/source_segments.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in provenance)
    )
    selection = {
        "dataset": output.name, "source_episodes": 160, "lerobot_episodes": 320,
        "total_frames": expected_frames, "fps": 15,
        "selection": {"episode_type": "standard_pick", "source_batch_counts": {"v2": 80, "v3": 80}},
        "sources": [str(old), str(new)], "excluded": "v2 tabletop_recovery: 20 raw episodes",
        "stats_scope": "exact state/action statistics over all selected frames; image statistics aggregated",
    }
    (output / "meta/source_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    checksums = []
    for directory in ("data", "meta", "videos"):
        for path in sorted((output / directory).rglob("*")):
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            checksums.append(f"{digest.hexdigest()}  {path.relative_to(output).as_posix()}\n")
    (output / "checksums.sha256").write_text("".join(checksums))
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(args.old.resolve(), args.new.resolve(), args.output.resolve())
