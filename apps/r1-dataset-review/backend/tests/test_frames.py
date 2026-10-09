import base64
import io
import json

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from dataset_review.catalog import CAMERAS, Catalog
from dataset_review.frames import FrameReader
from test_catalog import converted, raw  # noqa: F401


def test_raw_missing_and_real_frame(raw):
    path = raw / "episode_0007/data.json"
    payload = json.loads(path.read_text())
    Image.new("RGB", (32, 24), "red").save(raw / "episode_0007/test.jpg")
    payload["data"][0]["colors"]["color_0"] = "test.jpg"
    path.write_text(json.dumps(payload))
    reader = FrameReader()
    trajectory = Catalog([raw]).datasets[0].trajectories("full")[0]
    bundle = reader.bundle(trajectory, 0)
    assert bundle["cameras"][0]["width"] == 32
    assert bundle["cameras"][1]["image"] is None
    assert bundle["cameras"][1]["error"]
    with pytest.raises(ValueError):
        reader.bundle(trajectory, 6)
    reader.close()


def test_video_offsets_and_segment_boundary(converted):
    metadata = converted / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(metadata).to_pylist()
    for camera_index, (_, _, _, feature) in enumerate(CAMERAS):
        path = converted / f"videos/{feature}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
            for i in range(16):
                frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), i * 12, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        for i, row in enumerate(rows):
            start = .1 * camera_index + i * .6
            row.update({f"videos/{feature}/chunk_index": 0, f"videos/{feature}/file_index": 0,
                        f"videos/{feature}/from_timestamp": start, f"videos/{feature}/to_timestamp": start + .3})
    pq.write_table(pa.Table.from_pylist(rows), metadata)
    trajectory = Catalog([converted]).datasets[0].trajectories("full")[0]
    reader = FrameReader()
    for index in (0, 2, 3, 5, 1):
        bundle = reader.bundle(trajectory, index)
        for camera_index, camera in enumerate(bundle["cameras"]):
            assert camera["error"] is None
            expected = camera_index + (index if index < 3 else index + 3)
            assert camera["media_timestamp_s"] == pytest.approx(expected / 10)
            image = Image.open(io.BytesIO(base64.b64decode(camera["image"].split(",")[1])))
            assert abs(np.asarray(image).mean() - expected * 12) < 5
    reader.close()
