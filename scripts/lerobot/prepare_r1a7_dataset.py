#!/usr/bin/env python3
"""Create a training copy of an R1-A7 LeRobot v3 dataset with exact numeric stats."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
DEFAULT_FEATURES = ("observation.state", "action")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_feature(parquet_files: list[Path], feature: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for parquet_file in parquet_files:
        frame = pd.read_parquet(parquet_file, columns=[feature])
        if frame.empty:
            continue
        chunks.append(np.stack(frame[feature].to_numpy()).astype(np.float64, copy=False))
    if not chunks:
        raise ValueError(f"No samples found for feature {feature!r}")
    values = np.concatenate(chunks, axis=0)
    if values.ndim != 2:
        raise ValueError(f"Feature {feature!r} must be a vector, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Feature {feature!r} contains NaN or Inf")
    return values


def _allowed_mask(filter_payload: dict, expected_frames: int) -> np.ndarray:
    if int(filter_payload["total_frames"]) != expected_frames:
        raise ValueError(
            f"Training filter total_frames={filter_payload['total_frames']} "
            f"does not match dataset total_frames={expected_frames}"
        )
    allowed = np.ones(expected_frames, dtype=bool)
    for row in filter_payload["excluded_ranges"]:
        start = int(row["dataset_start"])
        end = int(row["dataset_end_exclusive"])
        if start < 0 or end <= start or end > expected_frames:
            raise ValueError(f"Invalid training filter range [{start}, {end})")
        allowed[start:end] = False
    actual_allowed = int(allowed.sum())
    if actual_allowed != int(filter_payload["allowed_training_starts"]):
        raise ValueError(
            f"Training filter allowed count={filter_payload['allowed_training_starts']} "
            f"but ranges produce {actual_allowed}"
        )
    return allowed


def _split_mask(dataset_root: Path, split_payload: dict, expected_frames: int, split_name: str) -> np.ndarray:
    """Select only frames from a manifest split, using the dataset's absolute episode ranges."""
    if int(split_payload["dataset"]["total_frames"]) != expected_frames:
        raise ValueError("Episode split total_frames does not match dataset")
    episodes = pd.read_parquet(
        dataset_root / "meta/episodes/chunk-000/file-000.parquet",
        columns=["episode_index", "dataset_from_index", "dataset_to_index"],
    )
    expected_episodes = int(split_payload["dataset"]["total_output_episodes"])
    if len(episodes) != expected_episodes or sorted(episodes["episode_index"].tolist()) != list(
        range(expected_episodes)
    ):
        raise ValueError("Episode metadata does not match split manifest")
    selected = set(split_payload["splits"][split_name]["output_episode_indices"])
    if not selected or any(not isinstance(index, int) or index < 0 or index >= expected_episodes for index in selected):
        raise ValueError(f"Invalid episode indices in split {split_name!r}")
    mask = np.zeros(expected_frames, dtype=bool)
    covered = np.zeros(expected_frames, dtype=bool)
    for row in episodes.itertuples(index=False, name=None):
        episode_index, start, end = map(int, row)
        if start < 0 or end <= start or end > expected_frames or covered[start:end].any():
            raise ValueError(f"Invalid or overlapping episode range [{start}, {end})")
        covered[start:end] = True
        if episode_index in selected:
            mask[start:end] = True
    if not covered.all() or int(mask.sum()) != int(split_payload["splits"][split_name]["frames"]):
        raise ValueError("Episode split frame coverage does not match dataset")
    return mask


def _exact_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    result: dict[str, list[float] | list[int]] = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }
    quantiles = np.quantile(values, QUANTILES, axis=0)
    for quantile, row in zip(QUANTILES, quantiles, strict=True):
        result[f"q{int(quantile * 100):02d}"] = row.tolist()
    return result


def prepare_dataset(
    source: Path,
    output: Path,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    reuse_output: bool = False,
    training_filter: Path | None = None,
    episode_split: Path | None = None,
    split_name: str = "train",
) -> dict[str, object]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output:
        raise ValueError("Source and output must differ; the original dataset is never modified")

    source_stats = source / "meta" / "stats.json"
    source_info = source / "meta" / "info.json"
    if not source_stats.is_file() or not source_info.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset root: {source}")

    if output.exists():
        if not reuse_output:
            raise FileExistsError(
                f"Output already exists: {output}. Pass --reuse-output to update only its stats."
            )
    else:
        shutil.copytree(source, output)

    before_hash = _sha256(source_stats)
    info = json.loads(source_info.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot dataset v3.0, got {info.get('codebase_version')!r}")

    expected_frames = int(info["total_frames"])
    allowed = np.ones(expected_frames, dtype=bool)
    filter_payload = None
    filter_hash = None
    if training_filter is not None:
        training_filter = training_filter.expanduser().resolve()
        filter_payload = json.loads(training_filter.read_text(encoding="utf-8"))
        allowed = _allowed_mask(filter_payload, expected_frames)
        filter_hash = _sha256(training_filter)
        output_filter = output / "meta" / "training_filter.json"
        if training_filter != output_filter.resolve():
            shutil.copy2(training_filter, output_filter)
    split_hash = None
    if episode_split is not None:
        episode_split = episode_split.expanduser().resolve()
        split_payload = json.loads(episode_split.read_text(encoding="utf-8"))
        allowed &= _split_mask(output, split_payload, expected_frames, split_name)
        expected_allowed = int(split_payload["splits"][split_name]["allowed_training_starts"])
        if training_filter is not None and int(allowed.sum()) != expected_allowed:
            raise ValueError(
                f"Split {split_name!r} allows {expected_allowed} starts, "
                f"but split and filter select {int(allowed.sum())}"
            )
        split_hash = _sha256(episode_split)
        output_split = output / "meta" / "episode_split.json"
        if episode_split != output_split.resolve():
            shutil.copy2(episode_split, output_split)
    parquet_files = sorted((output / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found below {output / 'data'}")

    output_stats_path = output / "meta" / "stats.json"
    stats = json.loads(output_stats_path.read_text(encoding="utf-8"))
    audit_features: dict[str, object] = {}
    for feature in features:
        metadata = info.get("features", {}).get(feature)
        if metadata is None:
            raise KeyError(f"Dataset metadata has no feature {feature!r}")
        values = _load_feature(parquet_files, feature)
        if values.shape[0] != expected_frames:
            raise ValueError(
                f"Feature {feature!r} has {values.shape[0]} rows, expected {expected_frames}"
            )
        expected_shape = tuple(metadata.get("shape", ()))
        if tuple(values.shape[1:]) != expected_shape:
            raise ValueError(
                f"Feature {feature!r} shape {values.shape[1:]} does not match {expected_shape}"
            )
        filtered_values = values[allowed]
        exact = _exact_stats(filtered_values)
        stats[feature] = exact
        audit_features[feature] = {
            "source_shape": list(values.shape),
            "selected_shape": list(filtered_values.shape),
            "q01": exact["q01"],
            "q99": exact["q99"],
        }

    output_stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    if _sha256(source_stats) != before_hash:
        raise RuntimeError("The source stats file changed while preparing the copy")

    audit: dict[str, object] = {
        "source": str(source),
        "output": str(output),
        "source_stats_sha256": before_hash,
        "output_stats_sha256": _sha256(output_stats_path),
        "method": (
            "numpy.quantile over split-and-filter-allowed Parquet rows"
            if episode_split is not None and filter_payload is not None
            else "numpy.quantile over training-filter-allowed Parquet rows"
            if filter_payload is not None
            else "numpy.quantile over every Parquet row"
        ),
        "quantiles": list(QUANTILES),
        "sample_selection": {
            "source_frames": expected_frames,
            "selected_frames": int(allowed.sum()),
            "excluded_frames": int(expected_frames - allowed.sum()),
            "training_filter_sha256": filter_hash,
            "episode_split_sha256": split_hash,
            "episode_split_name": split_name if episode_split is not None else None,
        },
        "features": audit_features,
    }
    audit_path = output / "meta" / "r1a7_exact_stats_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="Reuse an existing copied dataset and update only its numeric stats.",
    )
    parser.add_argument(
        "--training-filter",
        type=Path,
        help="Compute numeric stats only from sample starts allowed by this filter manifest.",
    )
    parser.add_argument("--episode-split", type=Path, help="Restrict numeric stats to one split.")
    parser.add_argument("--split-name", default="train")
    args = parser.parse_args()
    audit = prepare_dataset(
        args.source,
        args.output,
        reuse_output=args.reuse_output,
        training_filter=args.training_filter,
        episode_split=args.episode_split,
        split_name=args.split_name,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
