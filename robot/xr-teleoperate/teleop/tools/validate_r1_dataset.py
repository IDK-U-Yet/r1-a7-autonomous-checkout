#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


LEGACY_CAMERAS = {
    "global": ("color_0", (1080, 1920, 3)),
    "left_wrist": ("color_1", (480, 640, 3)),
    "right_wrist": ("color_2", (480, 640, 3)),
}
RAW_JOINT_GROUPS = ("left_arm", "right_arm", "waist", "left_ee", "right_ee")
EXPECTED_RAW_DIMENSION = 17
FRAME_AGE_PERCENTILE = 99
CAMERA_SKEW_PERCENTILE = 95


def configured_cameras(info, errors):
    raw_schema = info.get("camera_schema")
    if raw_schema is None:
        return LEGACY_CAMERAS
    if not isinstance(raw_schema, dict) or not raw_schema:
        errors.append("info.camera_schema must be a non-empty object")
        return {}

    cameras = {}
    used_color_keys = set()
    for name, camera in raw_schema.items():
        if not isinstance(name, str) or not name or not isinstance(camera, dict):
            errors.append("info.camera_schema contains an invalid camera entry")
            continue
        color_key = camera.get("color_key")
        shape = camera.get("shape")
        if not isinstance(color_key, str) or not color_key.startswith("color_"):
            errors.append(f"info.camera_schema.{name} has invalid color_key")
            continue
        if color_key in used_color_keys:
            errors.append(f"info.camera_schema reuses {color_key}")
            continue
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
        ):
            errors.append(f"info.camera_schema.{name} has invalid shape")
            continue
        used_color_keys.add(color_key)
        cameras[name] = (color_key, tuple(shape))
    return cameras


def percentile(values, quantile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def duplicate_ratio_limit(camera_name, args):
    if camera_name == "scanner":
        return args.max_scanner_duplicate_ratio
    return args.max_duplicate_ratio


def vector_size(sample, field):
    total = 0
    for group in RAW_JOINT_GROUPS:
        values = sample.get(field, {}).get(group, {}).get("qpos")
        if values is None:
            raise ValueError(f"missing {field}.{group}.qpos")
        total += int(np.asarray(values).size)
    return total


def discover_episodes(path):
    if path.is_file():
        return [path]
    direct = path / "data.json"
    if direct.is_file():
        return [direct]
    return sorted(
        candidate
        for candidate in path.rglob("data.json")
        if "rejected" not in candidate.relative_to(path).parts
    )


def validate_episode(json_path, args):
    errors = []
    warnings = []
    try:
        episode = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(json_path), "errors": [f"invalid JSON: {exc}"]}

    samples = episode.get("data")
    if not isinstance(samples, list) or not samples:
        return {"path": str(json_path), "errors": ["data contains no samples"]}

    info = episode.get("info", {})
    cameras = configured_cameras(info, errors)
    fps = info.get("image", {}).get("fps")
    task_event_required = isinstance(info.get("task_event_contract"), dict)
    try:
        fps = float(fps)
        if fps <= 0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append(f"invalid info.image.fps: {fps!r}")
        fps = None

    expected_indices = list(range(len(samples)))
    actual_indices = [sample.get("idx") for sample in samples]
    if actual_indices != expected_indices:
        errors.append("sample idx values are not consecutive from zero")

    record_times = []
    active_tasks = []
    camera_sequences = {name: [] for name in cameras}
    frame_ages = {name: [] for name in cameras}
    camera_skews = []

    for index, sample in enumerate(samples):
        for field in ("states", "actions"):
            try:
                size = vector_size(sample, field)
                if size != EXPECTED_RAW_DIMENSION:
                    errors.append(
                        f"sample {index}: {field} has {size} values, "
                        f"expected raw dimension {EXPECTED_RAW_DIMENSION}"
                    )
            except ValueError as exc:
                errors.append(f"sample {index}: {exc}")

        metadata = sample.get("metadata") or {}
        active_task = metadata.get("active_task")
        if active_task is not None:
            if not isinstance(active_task, str) or not active_task.strip():
                errors.append(f"sample {index}: invalid active_task")
            active_tasks.append(active_task)
        record_time = metadata.get("record_monotonic_ns")
        if not isinstance(record_time, int):
            errors.append(f"sample {index}: missing integer record_monotonic_ns")
        else:
            record_times.append(record_time)

        frame_metadata = metadata.get("camera_frames") or {}
        capture_times = []
        for camera_name, (color_key, expected_shape) in cameras.items():
            relative_path = (sample.get("colors") or {}).get(color_key)
            if not relative_path:
                errors.append(f"sample {index}: missing {color_key}")
            else:
                image_path = json_path.parent / relative_path
                image = cv2.imread(str(image_path))
                if image is None:
                    errors.append(f"sample {index}: unreadable image {image_path}")
                elif image.shape != expected_shape:
                    errors.append(
                        f"sample {index}: {color_key} shape {image.shape}, expected {expected_shape}"
                    )

            camera_meta = frame_metadata.get(camera_name)
            if not isinstance(camera_meta, dict):
                errors.append(f"sample {index}: missing metadata for {camera_name}")
                continue
            sequence = camera_meta.get("frame_sequence")
            capture_time = camera_meta.get("capture_monotonic_ns")
            age_ms = camera_meta.get("sample_age_ms")
            if not isinstance(sequence, int):
                errors.append(f"sample {index}: invalid {camera_name} frame_sequence")
            else:
                camera_sequences[camera_name].append(sequence)
            if not isinstance(capture_time, int):
                errors.append(f"sample {index}: invalid {camera_name} capture_monotonic_ns")
            else:
                capture_times.append(capture_time)
            if not isinstance(age_ms, (int, float)):
                errors.append(f"sample {index}: invalid {camera_name} sample_age_ms")
            else:
                frame_ages[camera_name].append(float(age_ms))

        if cameras and len(capture_times) == len(cameras):
            camera_skews.append((max(capture_times) - min(capture_times)) / 1_000_000.0)

    metrics = {"samples": len(samples), "configured_fps": fps}
    if active_tasks:
        if len(active_tasks) != len(samples):
            errors.append("active_task is missing from some samples")
        else:
            transitions = [
                index
                for index in range(1, len(active_tasks))
                if active_tasks[index] != active_tasks[index - 1]
            ]
            metrics["task_transition_count"] = len(transitions)
            metrics["active_tasks"] = list(dict.fromkeys(active_tasks))
            if len(transitions) > 1:
                errors.append("active_task changes more than once")

    events_path = json_path.parent / "events.jsonl"
    if events_path.exists():
        events = []
        for line_number, line in enumerate(
            events_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"events.jsonl line {line_number}: invalid JSON: {exc}")
        accepted_events = [event for event in events if event.get("event") == "item_accepted"]
        metrics["item_accepted_events"] = len(accepted_events)
        if len(accepted_events) > 1:
            errors.append("raw episode contains more than one item_accepted event")
        elif len(accepted_events) == 0 and task_event_required:
            errors.append("raw episode requires exactly one item_accepted event")
        elif len(accepted_events) == 1:
            event = accepted_events[0]
            event_time = event.get("receiver_monotonic_ns")
            last_enqueued_frame_index = event.get("last_enqueued_frame_index")
            if not isinstance(event_time, int):
                errors.append("item_accepted event has no integer receiver_monotonic_ns")
            if len(record_times) == len(samples) and len(active_tasks) == len(samples):
                expected_post_index = None
                boundary_source = None
                if last_enqueued_frame_index is not None:
                    if isinstance(last_enqueued_frame_index, bool) or not isinstance(
                        last_enqueued_frame_index, int
                    ):
                        errors.append(
                            "item_accepted event has invalid last_enqueued_frame_index"
                        )
                    else:
                        expected_post_index = last_enqueued_frame_index + 1
                        boundary_source = "last_enqueued_frame_index"
                elif isinstance(event_time, int):
                    expected_post_index = next(
                        (
                            index
                            for index, stamp in enumerate(record_times)
                            if stamp >= event_time
                        ),
                        None,
                    )
                    boundary_source = "receiver_monotonic_ns"

                if boundary_source is not None:
                    metrics["task_transition_boundary_source"] = boundary_source
                    metrics["expected_task_transition_index"] = expected_post_index
                if expected_post_index is None or not 0 < expected_post_index < len(samples):
                    errors.append("item_accepted event does not leave non-empty before/after task segments")
                elif boundary_source is not None:
                    actual_transitions = [
                        index
                        for index in range(1, len(active_tasks))
                        if active_tasks[index] != active_tasks[index - 1]
                    ]
                    if actual_transitions != [expected_post_index]:
                        source_description = (
                            "queue boundary"
                            if boundary_source == "last_enqueued_frame_index"
                            else "receive time"
                        )
                        errors.append(
                            "active_task transition does not match item_accepted "
                            f"{source_description}"
                        )
        elif len(active_tasks) == len(samples) and len(set(active_tasks)) > 1:
            errors.append("active_task changed without an item_accepted event")
    elif task_event_required:
        errors.append("events.jsonl is missing for a task-switching episode")
    if len(record_times) == len(samples) and len(record_times) > 1:
        intervals_ms = np.diff(record_times) / 1_000_000.0
        if np.any(intervals_ms <= 0):
            errors.append("record_monotonic_ns is not strictly increasing")
        metrics["record_interval_ms_p95"] = percentile(intervals_ms, 95)
        if fps is not None:
            max_interval_ms = 1000.0 / fps * args.max_interval_factor
            if float(np.max(intervals_ms)) > max_interval_ms:
                errors.append(
                    f"record interval max {float(np.max(intervals_ms)):.1f} ms exceeds "
                    f"{max_interval_ms:.1f} ms"
                )

    for camera_name in cameras:
        sequences = camera_sequences[camera_name]
        ages = frame_ages[camera_name]
        if len(sequences) == len(samples) and len(sequences) > 1:
            deltas = np.diff(sequences)
            if np.any(deltas < 0):
                errors.append(f"{camera_name} frame sequence moved backwards")
            duplicate_ratio = float(np.count_nonzero(deltas == 0) / len(deltas))
            metrics[f"{camera_name}_duplicate_ratio"] = duplicate_ratio
            duplicate_limit = duplicate_ratio_limit(camera_name, args)
            if duplicate_ratio > duplicate_limit:
                errors.append(
                    f"{camera_name} duplicate ratio {duplicate_ratio:.1%} exceeds "
                    f"{duplicate_limit:.1%}"
                )
        if len(ages) == len(samples):
            max_age = max(ages)
            high_age = percentile(ages, FRAME_AGE_PERCENTILE)
            metrics[f"{camera_name}_max_age_ms"] = max_age
            metrics[f"{camera_name}_frame_age_p99_ms"] = high_age
            if high_age > args.max_frame_age_ms:
                errors.append(
                    f"{camera_name} P99 frame age {high_age:.1f} ms exceeds "
                    f"{args.max_frame_age_ms:.1f} ms"
                )

    if len(camera_skews) == len(samples):
        max_skew = max(camera_skews)
        high_skew = percentile(camera_skews, CAMERA_SKEW_PERCENTILE)
        metrics["max_cross_camera_skew_ms"] = max_skew
        metrics["cross_camera_skew_p95_ms"] = high_skew
        if high_skew > args.max_camera_skew_ms:
            errors.append(
                f"cross-camera skew P95 {high_skew:.1f} ms exceeds "
                f"{args.max_camera_skew_ms:.1f} ms"
            )
        elif high_skew > 50.0:
            warnings.append(
                f"cross-camera skew P95 reached {high_skew:.1f} ms "
                f"(max {max_skew:.1f} ms); cameras are not hardware-synchronized"
            )

    return {
        "path": str(json_path),
        "metrics": metrics,
        "warnings": warnings,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate raw R1-A7 camera episodes.")
    parser.add_argument("path", type=Path, help="Raw root, task directory, episode directory, or data.json")
    parser.add_argument("--max-duplicate-ratio", type=float, default=0.05)
    parser.add_argument("--max-scanner-duplicate-ratio", type=float, default=0.20)
    parser.add_argument("--max-frame-age-ms", type=float, default=150.0)
    parser.add_argument("--max-camera-skew-ms", type=float, default=100.0)
    parser.add_argument("--max-interval-factor", type=float, default=2.5)
    args = parser.parse_args()

    episodes = discover_episodes(args.path)
    if not episodes:
        parser.error(f"no data.json found under {args.path}")

    reports = [validate_episode(path, args) for path in episodes]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    failed = sum(bool(report.get("errors")) for report in reports)
    print(f"Validated {len(reports)} episode(s): {len(reports) - failed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
