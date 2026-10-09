from __future__ import annotations

import base64
import io
import threading
from collections import OrderedDict

import av
from PIL import Image

from .catalog import CAMERAS, Trajectory, safe_path


class FrameReader:
    """Bounded, serialized decoder. All four views belong to one recorded sample."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.containers = OrderedDict()

    def close(self):
        with self.lock:
            for container in self.containers.values():
                container.close()
            self.containers.clear()
            self.cache.clear()

    def _video(self, path, target, end, fps):
        if path not in self.containers:
            container = av.open(str(path))
            container.streams.video[0].codec_context.thread_count = 2
            self.containers[path] = container
            while len(self.containers) > 8:
                _, old = self.containers.popitem(last=False)
                old.close()
        self.containers.move_to_end(path)
        container = self.containers[path]
        stream = container.streams.video[0]
        container.seek(max(0, int(target / stream.time_base)), stream=stream, backward=True)
        best, distance = None, float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            delta = abs(timestamp - target)
            if timestamp < end - 1e-6 and delta < distance:
                best, distance = frame, delta
            if timestamp >= target:
                break
        if best is None or distance > .51 / fps + 1e-4:
            raise ValueError("视频中没有与当前帧时间对应的画面")
        return best.to_image(), float(best.pts * best.time_base)

    def bundle(self, trajectory: Trajectory, index: int):
        with self.lock:
            key = (trajectory.id, index)
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            segment, local = trajectory.locate(index)
            dataset = trajectory.dataset
            cameras = []
            for camera, title, color, feature in CAMERAS:
                item = {"key": camera, "title": title, "image": None, "error": None}
                try:
                    if segment.raw_dir:
                        sample = segment.raw()["data"][local]
                        relative = sample.get("colors", {}).get(color)
                        if not relative:
                            raise ValueError("该帧缺少相机图片")
                        path = safe_path(segment.raw_dir, relative)
                        with Image.open(path) as original:
                            picture = original.convert("RGB")
                        item["media_timestamp_s"] = float(segment.timestamps()[local])
                    else:
                        prefix = f"videos/{feature}/"
                        row = segment.row
                        start, end = float(row[prefix + "from_timestamp"]), float(row[prefix + "to_timestamp"])
                        target = start + float(segment.timestamps()[local])
                        path = safe_path(dataset.root, dataset.info["video_path"].format(
                            video_key=feature, chunk_index=row[prefix + "chunk_index"],
                            file_index=row[prefix + "file_index"]))
                        picture, timestamp = self._video(path, target, end, dataset.fps)
                        item["media_timestamp_s"] = timestamp
                    output = io.BytesIO()
                    picture.save(output, format="JPEG", quality=85)
                    item.update(image="data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode(),
                                width=picture.width, height=picture.height)
                except (OSError, ValueError, KeyError, av.error.FFmpegError) as error:
                    item["error"] = str(error)
                cameras.append(item)
            result = {"trajectory_id": trajectory.id, "frame_index": index,
                      "timestamp_s": trajectory.timestamp(index), "cameras": cameras}
            self.cache[key] = result
            while len(self.cache) > 32:
                self.cache.popitem(last=False)
            return result
