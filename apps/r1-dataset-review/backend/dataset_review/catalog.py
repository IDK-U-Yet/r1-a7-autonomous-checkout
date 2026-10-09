from __future__ import annotations

import bisect
import hashlib
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


CAMERAS = (
    ("global", "全局相机", "color_0", "observation.images.cam_global"),
    ("scanner", "扫码区相机", "color_3", "observation.images.cam_scanner"),
    ("left_wrist", "左腕相机", "color_1", "observation.images.cam_left_wrist"),
    ("right_wrist", "右腕相机", "color_2", "observation.images.cam_right_wrist"),
)


def identity(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def safe_path(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("文件路径超出注册数据目录")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cache_lock(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return locked


@dataclass
class Segment:
    dataset: Dataset
    index: int
    length: int
    source_start: int
    task: str
    row: dict
    raw_dir: Path | None = None
    source_path: str = ""

    def raw(self) -> dict:
        if self.raw_dir is None:
            raise ValueError("不是原始图片数据")
        return self.dataset.raw_payload(self.raw_dir)

    def timestamps(self) -> np.ndarray:
        return self.dataset.timestamps(self)


class Trajectory:
    def __init__(self, dataset: Dataset, key: str, name: str, segments: list[Segment], view: str, batch: str):
        self.dataset, self.name, self.segments, self.view, self.batch = dataset, name, segments, view, batch
        self.id = identity(f"{dataset.id}:{view}:{key}")
        self.ends = np.cumsum([s.length for s in segments]).tolist()
        self.length = self.ends[-1]
        self._detail = None

    def locate(self, frame_index: int) -> tuple[Segment, int]:
        if not 0 <= frame_index < self.length:
            raise ValueError("帧编号超出当前轨迹")
        i = bisect.bisect_right(self.ends, frame_index)
        return self.segments[i], frame_index - (self.ends[i - 1] if i else 0)

    def timestamp(self, frame_index: int) -> float:
        segment, local = self.locate(frame_index)
        i = self.segments.index(segment)
        offset = sum(float(s.timestamps()[-1]) + 1 / self.dataset.fps for s in self.segments[:i])
        return offset + float(segment.timestamps()[local])

    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "source_batch": self.batch,
                "view": self.view, "total_frames": self.length,
                "tasks": list(dict.fromkeys(s.task for s in self.segments if s.task))}

    def detail(self) -> dict:
        if self._detail is not None:
            return self._detail
        payload, source_dir = {}, None
        first = self.segments[0]
        if first.raw_dir is not None:
            payload, source_dir = first.raw(), first.raw_dir
        elif first.source_path:
            source_dir = self.dataset.catalog.source_directory(first.source_path)
            if source_dir and (source_dir / "data.json").is_file():
                payload = self.dataset.raw_payload(source_dir)
        info = payload.get("info", {})
        product = info.get("collection", {})
        event_rows = []
        if source_dir and (source_dir / "events.jsonl").is_file():
            for line in safe_path(source_dir, "events.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("event") != "item_accepted":
                    continue
                boundary = event.get("last_enqueued_frame_index")
                if not isinstance(boundary, int):
                    continue
                local = boundary + 1 - first.source_start
                if 0 <= local < self.length:
                    event_rows.append({"event": "item_accepted", "frame_index": local,
                                       "timestamp_s": self.timestamp(local), "label": "扫码成功",
                                       "barcode": event.get("barcode"), "event_id": event.get("event_id")})
        boundaries = []
        offset = 0
        if first.raw_dir:
            frames = first.raw()["data"]
            previous = None
            for i, frame in enumerate(frames):
                task = frame.get("metadata", {}).get("active_task", "")
                if i == 0 or task != previous:
                    boundaries.append({"frame_index": i, "task": task})
                previous = task
        else:
            for segment in self.segments:
                boundaries.append({"frame_index": offset, "task": segment.task})
                offset += segment.length
        if not event_rows:
            for boundary in boundaries[1:]:
                index = boundary["frame_index"]
                event_rows.append({"event": "task_transition", "frame_index": index,
                                   "timestamp_s": self.timestamp(index), "label": "任务切换（非扫码事件）"})
        self._detail = {**self.summary(), "fps": self.dataset.fps,
                        "duration_s": self.timestamp(self.length - 1) + 1 / self.dataset.fps,
                        "product_name": product.get("name", ""), "barcode": product.get("barcode", ""),
                        "sku": product.get("sku", ""), "events": event_rows,
                        "task_boundaries": boundaries, "metadata_fingerprint": self.dataset.fingerprint,
                        "annotation_available": False,
                        "cameras": [{"key": key, "title": title} for key, title, _, _ in CAMERAS]}
        return self._detail


class Dataset:
    def __init__(self, catalog: Catalog, root: Path):
        self._lock = threading.RLock()
        self.catalog, self.root = catalog, root.resolve()
        self.id, self.name = identity(str(self.root)), self.root.name
        self._raw_cache: OrderedDict[Path, dict] = OrderedDict()
        self._time_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.segments: list[Segment] = []
        self._full: list[Trajectory] = []
        self._segments: list[Trajectory] = []
        self.has_full_view = False
        info_path = self.root / "meta/info.json"
        if info_path.is_file():
            self.format, self.info = "lerobot-v3", read_json(info_path)
            if not str(self.info.get("codebase_version", "")).startswith("v3."):
                raise ValueError("首版只支持 LeRobot v3 和本项目原始批次")
            self.fps = float(self.info["fps"])
            self._load_converted()
            signature = info_path.read_bytes()
            provenance = self.root / "meta/source_segments.jsonl"
            if provenance.is_file():
                signature += provenance.read_bytes()
        else:
            self.format, self.info = "raw", {}
            signature = (self.root / "batch_state.json").read_bytes()
            self.fps = 15.0
            self._load_raw()
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("数据 FPS 无效")
        metadata_files = ([safe_path(self.root, s.raw_dir / 'data.json') for s in self.segments]
                          if self.format == 'raw' else
                          sorted((self.root / 'meta/episodes').rglob('*.parquet')) + sorted((self.root / 'data').rglob('*.parquet')))
        for file in metadata_files:
            file = safe_path(self.root, file)
            stat = file.stat()
            signature += f'{file.relative_to(self.root)}:{stat.st_size}:{stat.st_mtime_ns}'.encode()
        self.fingerprint = hashlib.sha256(signature).hexdigest()
        identity_files = [info_path, self.root / 'meta/source_segments.jsonl'] if self.format != 'raw' else [self.root / 'batch_state.json']
        self._source_snapshot = {file: (file.stat().st_size, file.stat().st_mtime_ns)
                                 for file in [*metadata_files, *identity_files] if file.exists()}
        self._lookup = {t.id: t for t in self._full + self._segments}

    def source_unchanged(self):
        try:
            return all((file.stat().st_size, file.stat().st_mtime_ns) == stamp
                       for file, stamp in self._source_snapshot.items())
        except OSError:
            return False

    @cache_lock
    def raw_payload(self, directory: Path) -> dict:
        if directory not in self._raw_cache:
            value = read_json(safe_path(directory, "data.json"))
            self._raw_cache[directory] = value
            while len(self._raw_cache) > 2:
                self._raw_cache.popitem(last=False)
        self._raw_cache.move_to_end(directory)
        return self._raw_cache[directory]

    @cache_lock
    def timestamps(self, segment: Segment) -> np.ndarray:
        if segment.index not in self._time_cache:
            if segment.raw_dir:
                frames = segment.raw()["data"]
                clock = [f.get("metadata", {}).get("record_monotonic_ns") for f in frames]
                if all(isinstance(t, (int, float)) for t in clock):
                    values = (np.asarray(clock, dtype=np.float64) - clock[0]) / 1e9
                else:
                    values = np.arange(segment.length) / self.fps
            else:
                path = safe_path(self.root, self.info["data_path"].format(
                    chunk_index=segment.row["data/chunk_index"], file_index=segment.row["data/file_index"]))
                rows = pq.read_table(path, columns=["timestamp", "frame_index", "episode_index"],
                                     filters=[("episode_index", "=", segment.index)]).to_pylist()
                rows.sort(key=lambda row: row["frame_index"])
                if [r["frame_index"] for r in rows] != list(range(segment.length)):
                    raise ValueError("训练分段帧编号不连续或长度不符")
                values = np.asarray([r["timestamp"] for r in rows], dtype=np.float64)
                if len(values):
                    values = values - values[0]
            if len(values) != segment.length or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
                raise ValueError("轨迹时间戳无效或不递增")
            self._time_cache[segment.index] = values
            while len(self._time_cache) > 16:
                self._time_cache.popitem(last=False)
        self._time_cache.move_to_end(segment.index)
        return self._time_cache[segment.index]

    def _load_raw(self):
        state = read_json(self.root / "batch_state.json")
        seen = set()
        for entry in state.get("accepted", []):
            index = int(entry["episode_id"])
            if index in seen:
                raise ValueError("accepted 清单有重复 Episode")
            seen.add(index)
            directory = safe_path(self.root, f"episode_{index:04d}")
            payload = self.raw_payload(directory)
            frames = payload.get("data", [])
            if not frames:
                raise ValueError(f"{directory.name} 没有完整帧")
            if not self.segments and len(frames) > 1:
                clock = [f.get("metadata", {}).get("record_monotonic_ns") for f in frames[:50]]
                if all(isinstance(t, (int, float)) for t in clock):
                    interval = float(np.median(np.diff(clock))) / 1e9
                    if interval > 0:
                        value = 1 / interval
                        self.fps = float(round(value)) if abs(value - round(value)) < .2 else value
            task = frames[0].get("metadata", {}).get("active_task", payload.get("text", {}).get("goal", ""))
            segment = Segment(self, index, len(frames), 0, task, {}, directory, str(directory))
            self.segments.append(segment)
            self._full.append(Trajectory(self, str(index), directory.name, [segment], "full", self.name))
        self._segments = self._full[:]
        self.has_full_view = True

    def _load_converted(self):
        metadata = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        rows = []
        for path in metadata:
            table = pq.read_table(path)
            columns = [name for name in table.column_names if not name.startswith("stats/")]
            rows.extend(table.select(columns).to_pylist())
        rows.sort(key=lambda row: row["episode_index"])
        if [r["episode_index"] for r in rows] != list(range(int(self.info["total_episodes"]))):
            raise ValueError("Episode 元数据缺失或重复")
        provenance_path = self.root / "meta/source_segments.jsonl"
        provenance = {}
        if provenance_path.is_file():
            for line in provenance_path.read_text().splitlines():
                if line.strip():
                    value = json.loads(line)
                    index = int(value["output_episode_index"])
                    if index in provenance:
                        raise ValueError("来源清单分段重复")
                    provenance[index] = value
            if set(provenance) != set(range(len(rows))):
                raise ValueError("来源清单和 Episode 数量不匹配")
        groups = OrderedDict()
        for row in rows:
            index = int(row["episode_index"])
            source = provenance.get(index, {})
            length = int(row["length"])
            if length <= 0 or int(row["dataset_to_index"]) - int(row["dataset_from_index"]) != length:
                raise ValueError("分段长度和全局帧范围不符")
            start = int(source.get("start_frame", 0))
            if source and int(source["end_frame_exclusive"]) - start != length:
                raise ValueError("原始帧范围和训练分段长度不符")
            segment = Segment(self, index, length, start, " / ".join(row.get("tasks") or []), row,
                              source_path=source.get("source_episode_path", ""))
            self.segments.append(segment)
            batch = source.get("source_batch") or str(Path(segment.source_path).parent.name)
            name = Path(segment.source_path).name if source else f"episode_{index:04d}"
            self._segments.append(Trajectory(self, str(index), f"分段 {index:04d} · {name}", [segment], "segments", batch))
            if source:
                key = (batch, segment.source_path)
                groups.setdefault(key, []).append(segment)
        if provenance:
            for (batch, source_path), segments in groups.items():
                segments.sort(key=lambda segment: segment.source_start)
                cursor = 0
                for segment in segments:
                    if segment.source_start != cursor:
                        raise ValueError("完整轨迹来源帧范围有重叠或空缺")
                    cursor += segment.length
                self._full.append(Trajectory(self, f"{batch}:{source_path}", Path(source_path).name, segments, "full", batch))
            self.has_full_view = True

    def trajectories(self, view: str) -> list[Trajectory]:
        if view == "full":
            if not self.has_full_view:
                raise ValueError("没有可靠来源清单，请选择训练分段视图")
            return self._full
        if view == "segments":
            return self._segments
        raise ValueError("未知视图")

    def trajectory(self, trajectory_id: str) -> Trajectory:
        try:
            return self._lookup[trajectory_id]
        except KeyError as exc:
            raise ValueError("找不到注册轨迹") from exc

    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "format": self.format, "fps": self.fps,
                "full_count": len(self._full), "segment_count": len(self._segments),
                "has_full_view": self.has_full_view, "root": str(self.root), "fingerprint": self.fingerprint}


class Catalog:
    def __init__(self, roots: list[Path]):
        self.roots = list(dict.fromkeys(Path(root).resolve() for root in roots))
        self.datasets = [Dataset(self, root) for root in self.roots]

    def dataset(self, dataset_id: str) -> Dataset:
        for dataset in self.datasets:
            if dataset.id == dataset_id:
                return dataset
        raise ValueError("找不到注册数据集")

    def source_directory(self, value: str) -> Path | None:
        path = Path(value).resolve()
        for root in self.roots:
            if path.is_relative_to(root) and path.is_dir():
                return path
            if root.name == path.parent.name:
                fallback = safe_path(root, path.name)
                if fallback.is_dir():
                    return fallback
        return None
