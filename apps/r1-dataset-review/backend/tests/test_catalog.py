import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset_review.catalog import Catalog, safe_path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "raw"
    write_json(root / "batch_state.json", {"accepted": [
        {"episode_id": 7, "episode_path": "/old/location/episode_0007", "ordinal": 1},
    ]})
    write_json(root / "episode_0007/data.json", {
        "info": {"collection": {"name": "sample", "barcode": "123"}},
        "data": [{"colors": {}, "metadata": {"record_monotonic_ns": i * 100_000_000}}
                 for i in range(6)],
    })
    (root / "episode_0007/events.jsonl").write_text(json.dumps({
        "event": "item_accepted", "last_enqueued_frame_index": 2,
    }) + "\n")
    return root


@pytest.fixture
def converted(tmp_path):
    root = tmp_path / "converted"
    write_json(root / "meta/info.json", {
        "codebase_version": "v3.0", "fps": 10, "total_frames": 6,
        "total_episodes": 2, "features": {},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    })
    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([
        {"episode_index": i, "length": 3, "dataset_from_index": i * 3,
         "dataset_to_index": (i + 1) * 3, "tasks": ["scan" if i == 0 else "place"],
         "data/chunk_index": 0, "data/file_index": 0} for i in range(2)
    ]), episode_path)
    data_path = root / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([
        {"episode_index": i // 3, "frame_index": i % 3, "timestamp": (i % 3) / 10}
        for i in range(6)
    ]), data_path)
    (root / "meta/source_segments.jsonl").write_text("".join(json.dumps({
        "output_episode_index": i, "source_batch": "batch-a", "source_episode_path": "/gone/raw/episode_0007",
        "start_frame": i * 3, "end_frame_exclusive": (i + 1) * 3,
    }) + "\n" for i in range(2)))
    return root


def test_raw_accepted_only_and_events(raw):
    dataset = Catalog([raw]).datasets[0]
    trajectory = dataset.trajectories("full")[0]
    assert trajectory.length == 6
    assert trajectory.name == "episode_0007"
    assert trajectory.detail()["events"][0]["frame_index"] == 3
    assert trajectory.detail()["barcode"] == "123"
    assert trajectory.timestamp(5) == pytest.approx(.5)
    with pytest.raises(ValueError):
        trajectory.locate(6)


def test_full_and_segments_without_original(converted):
    dataset = Catalog([converted]).datasets[0]
    assert len(dataset.trajectories("full")) == 1
    assert len(dataset.trajectories("segments")) == 2
    full = dataset.trajectories("full")[0]
    assert full.locate(3)[1] == 0
    assert full.timestamp(3) == pytest.approx(.3)
    assert full.detail()["events"][0]["event"] == "task_transition"


def test_missing_provenance_disables_full(converted):
    (converted / "meta/source_segments.jsonl").unlink()
    dataset = Catalog([converted]).datasets[0]
    assert not dataset.has_full_view
    assert len(dataset.trajectories("segments")) == 2
    with pytest.raises(ValueError, match="来源"):
        dataset.trajectories("full")


def test_invalid_mapping_rejected(converted):
    path = converted / "meta/source_segments.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["start_frame"] = 2
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError):
        Catalog([converted])


def test_path_escape(raw, tmp_path):
    with pytest.raises(ValueError):
        safe_path(raw, "../outside")
    (raw / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        safe_path(raw, "link/file")


def test_batch_namespace(converted):
    path = converted / "meta/source_segments.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1].update(source_batch="batch-b", start_frame=0, end_frame_exclusive=3)
    path.write_text("\n".join(map(json.dumps, rows)))
    dataset = Catalog([converted]).datasets[0]
    trajectories = dataset.trajectories("full")
    assert len(trajectories) == 2
    assert trajectories[0].id != trajectories[1].id
