"""Independent annotations; source datasets are never written."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from .catalog import safe_path


class ConflictError(ValueError):
    pass


STATUSES = {"unreviewed", "approved", "needs_review", "excluded"}
FIELDS = {"schema_version", "dataset_id", "trajectory_id", "total_frames", "metadata_fingerprint",
          "revision", "updated_at", "review_status", "episode_type", "orientation", "stages",
          "quality_intervals", "notes"}


class AnnotationStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.lock = threading.RLock()

    def _path(self, trajectory):
        if not all(re.fullmatch(r"[a-f0-9]{20}", value) for value in (trajectory.id, trajectory.dataset.id)):
            raise ValueError("标注身份无效")
        return safe_path(self.root, f"{trajectory.dataset.id}/{trajectory.id}.json")

    def empty(self, trajectory):
        return {"schema_version": 1, "dataset_id": trajectory.dataset.id, "trajectory_id": trajectory.id,
                "total_frames": trajectory.length, "metadata_fingerprint": trajectory.dataset.fingerprint,
                "revision": 0, "updated_at": None, "review_status": "unreviewed", "episode_type": "",
                "orientation": "", "stages": [], "quality_intervals": [], "notes": ""}

    def validate(self, trajectory, record):
        if not isinstance(record, dict) or set(record) != FIELDS:
            raise ValueError("标注字段不完整或含未知字段")
        expected = self.empty(trajectory)
        for key in ("schema_version", "dataset_id", "trajectory_id", "total_frames", "metadata_fingerprint"):
            if type(record[key]) is not type(expected[key]) or record[key] != expected[key]:
                raise ConflictError(f"标注来源/版本不匹配：{key}；请重新加载，禁止覆盖旧标注")
        if type(record["revision"]) is not int or record["revision"] < 0:
            raise ValueError("revision 必须为非负整数")
        if not isinstance(record['review_status'], str) or record["review_status"] not in STATUSES:
            raise ValueError("审核状态无效")
        if not isinstance(record['episode_type'], str) or record["episode_type"] not in {"", "standard_pick", "tabletop_recovery"}:
            raise ValueError("轨迹类型无效")
        if not isinstance(record['orientation'], str) or record["orientation"] not in {"", *(f"O{i}" for i in range(1, 9))}:
            raise ValueError("朝向无效")
        if not isinstance(record["notes"], str) or len(record["notes"]) > 10000:
            raise ValueError("备注必须为文本且不超过 10000 字")
        for layer in ("stages", "quality_intervals"):
            intervals = record[layer]
            if not isinstance(intervals, list) or len(intervals) > 200:
                raise ValueError("区间必须为列表且不超过 200 个")
            for interval in intervals:
                fields = {"label", "start_frame", "end_frame_exclusive"}
                if layer == "quality_intervals":
                    fields |= {"category", "exclude_from_training"}
                if not isinstance(interval, dict) or set(interval) != fields:
                    raise ValueError("区间字段不正确")
                start, end = interval["start_frame"], interval["end_frame_exclusive"]
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= trajectory.length:
                    raise ValueError(f"区间必须满足 0 ≤ 起点 < 终点 ≤ {trajectory.length}")
                if not isinstance(interval["label"], str) or not interval["label"].strip() or len(interval["label"]) > 200:
                    raise ValueError("区间标签必须为 1～200 字")
                if layer == "quality_intervals":
                    if not isinstance(interval['category'], str) or interval["category"] not in {"error", "pause", "recovery", "custom"}:
                        raise ValueError("质量区间类型无效")
                    if type(interval["exclude_from_training"]) is not bool:
                        raise ValueError("训练排除开关必须为布尔值")
            ordered = sorted(intervals, key=lambda item: item["start_frame"])
            if any(left["end_frame_exclusive"] > right["start_frame"] for left, right in zip(ordered, ordered[1:])):
                raise ValueError(f"{layer} 同层区间不能重叠；质量区间可以与动作阶段重叠")

    def get(self, trajectory):
        with self.lock:
            path = self._path(trajectory)
            if not path.exists():
                return self.empty(trajectory)
            value = json.loads(path.read_text(encoding="utf-8"))
            self.validate(trajectory, value)
            return value

    def save(self, trajectory, value):
        with self.lock:
            if not trajectory.dataset.source_unchanged():
                raise ConflictError('运行中源数据元信息已变化，请重启加载后核对标注，旧文件未覆盖')
            self.validate(trajectory, value)
            current = self.get(trajectory)
            if current["revision"] != value["revision"]:
                raise ConflictError("标注已被其他窗口更新；请重新加载后修改，旧内容未覆盖")
            saved = {**value, "revision": current["revision"] + 1,
                     "updated_at": datetime.now(timezone.utc).isoformat()}
            path = self._path(trajectory)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                                 prefix=".annotation-", suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    json.dump(saved, handle, ensure_ascii=False, indent=2, allow_nan=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            return saved
