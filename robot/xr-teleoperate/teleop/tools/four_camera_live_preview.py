#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Serve a four-camera browser preview from the active teleimager ZMQ streams."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys
import time

from aiohttp import web


TELEOP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TELEOP_ROOT / "teleimager" / "src"))

from teleimager.image_client import ImageClient  # noqa: E402


@dataclass(frozen=True)
class CameraView:
    key: str
    title: str
    config_key: str
    getter_name: str


CAMERAS = (
    CameraView("global", "全局相机", "head_camera", "get_head_frame"),
    CameraView("left_wrist", "左腕相机", "left_wrist_camera", "get_left_wrist_frame"),
    CameraView("right_wrist", "右腕相机", "right_wrist_camera", "get_right_wrist_frame"),
    CameraView("scanner", "扫码区相机", "scanner_camera", "get_scanner_frame"),
)
CAMERA_BY_KEY = {camera.key: camera for camera in CAMERAS}
BOUNDARY = "r1a7-frame"


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R1-A7 四相机实时画面</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, "Noto Sans SC", "Microsoft YaHei", sans-serif;
      background: #111513;
      color: #eef2f0;
    }
    * { box-sizing: border-box; }
    html, body { min-width: 320px; min-height: 100%; margin: 0; }
    body { background: #111513; }
    header {
      min-height: 58px;
      padding: 10px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid #303936;
      background: #171c1a;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; letter-spacing: 0; }
    #overall { display: flex; align-items: center; gap: 8px; color: #aeb9b4; font-size: 13px; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #c74a42; }
    .dot.online { background: #3cab72; }
    main {
      width: min(1600px, 100%);
      margin: 0 auto;
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    section {
      min-width: 0;
      overflow: hidden;
      border: 1px solid #303936;
      border-radius: 6px;
      background: #080b0a;
    }
    .viewport {
      width: 100%;
      aspect-ratio: 4 / 3;
      overflow: hidden;
      background: #050706;
    }
    img { display: block; width: 100%; height: 100%; object-fit: contain; }
    .meta {
      min-height: 48px;
      padding: 8px 11px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-top: 1px solid #303936;
      background: #171c1a;
    }
    strong { display: block; font-size: 14px; font-weight: 650; letter-spacing: 0; }
    .metrics { color: #aeb9b4; font-size: 12px; text-align: right; white-space: nowrap; }
    .metrics.error { color: #ee8e86; }
    @media (max-width: 760px) {
      header { align-items: flex-start; flex-direction: column; gap: 5px; }
      main { grid-template-columns: 1fr; padding: 8px; gap: 8px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>R1-A7 四相机实时画面</h1>
    <div id="overall"><span class="dot"></span><span>正在连接</span></div>
  </header>
  <main>
    <section data-camera="global"><div class="viewport"><img src="/stream/global" alt="全局相机实时画面"></div><div class="meta"><strong>全局相机</strong><span class="metrics">--</span></div></section>
    <section data-camera="left_wrist"><div class="viewport"><img src="/stream/left_wrist" alt="左腕相机实时画面"></div><div class="meta"><strong>左腕相机</strong><span class="metrics">--</span></div></section>
    <section data-camera="right_wrist"><div class="viewport"><img src="/stream/right_wrist" alt="右腕相机实时画面"></div><div class="meta"><strong>右腕相机</strong><span class="metrics">--</span></div></section>
    <section data-camera="scanner"><div class="viewport"><img src="/stream/scanner" alt="扫码区相机实时画面"></div><div class="meta"><strong>扫码区相机</strong><span class="metrics">--</span></div></section>
  </main>
  <script>
    async function refreshStatus() {
      try {
        const response = await fetch('/api/status', {cache: 'no-store'});
        if (!response.ok) throw new Error('status');
        const cameras = await response.json();
        let online = 0;
        for (const [key, camera] of Object.entries(cameras)) {
          const section = document.querySelector(`[data-camera="${key}"]`);
          if (!section) continue;
          const metrics = section.querySelector('.metrics');
          if (camera.online) {
            online += 1;
            metrics.classList.remove('error');
            metrics.textContent = `${camera.width}×${camera.height} · ${camera.fps.toFixed(1)} FPS · ${Math.round(camera.age_ms)} ms`;
          } else {
            metrics.classList.add('error');
            metrics.textContent = '离线';
          }
        }
        const overall = document.querySelector('#overall');
        overall.querySelector('.dot').classList.toggle('online', online === 4);
        overall.querySelector('span:last-child').textContent = `${online}/4 路在线`;
      } catch (_error) {
        document.querySelector('#overall span:last-child').textContent = '状态服务断开';
      }
    }
    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--source-host", default="127.0.0.1")
    parser.add_argument("--preview-fps", type=float, default=15.0)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if not 1 <= args.preview_fps <= 30:
        parser.error("--preview-fps must be in 1..30")
    return args


async def index(_request):
    return web.Response(text=PAGE, content_type="text/html")


def latest_frame(app, camera):
    frame = getattr(app["image_client"], camera.getter_name)()
    if frame.jpg:
        app["last_frames"][camera.key] = frame
    return app["last_frames"].get(camera.key, frame)


async def status(request):
    config = request.app["camera_config"]
    now_ns = time.monotonic_ns()
    payload = {}
    for camera in CAMERAS:
        frame = latest_frame(request.app, camera)
        camera_config = config.get(camera.config_key, {})
        receive_ns = frame.receive_monotonic_ns
        age_ms = (now_ns - receive_ns) / 1_000_000 if receive_ns is not None else None
        height, width = camera_config.get("image_shape", (0, 0))
        payload[camera.key] = {
            "online": bool(frame.jpg) and age_ms is not None and age_ms < 1000,
            "fps": float(frame.fps),
            "age_ms": age_ms,
            "width": int(width),
            "height": int(height),
            "sequence": frame.frame_sequence,
        }
    return web.json_response(payload, headers={"Cache-Control": "no-store"})


async def stream(request):
    camera = CAMERA_BY_KEY.get(request.match_info["key"])
    if camera is None:
        raise web.HTTPNotFound()
    interval = 1.0 / request.app["preview_fps"]
    response = web.StreamResponse(
        headers={
            "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        }
    )
    await response.prepare(request)
    last_sequence = None
    try:
        while True:
            frame = latest_frame(request.app, camera)
            if frame.jpg and frame.frame_sequence != last_sequence:
                last_sequence = frame.frame_sequence
                header = (
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame.jpg)}\r\n\r\n"
                ).encode("ascii")
                await response.write(header + frame.jpg + b"\r\n")
            await asyncio.sleep(interval)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return response


async def cleanup(app):
    app["image_client"].close()


def main():
    args = parse_args()
    client = ImageClient(host=args.source_host, request_bgr=False)
    app = web.Application()
    app["image_client"] = client
    app["camera_config"] = client.get_cam_config()
    app["preview_fps"] = args.preview_fps
    app["last_frames"] = {}
    app.router.add_get("/", index)
    app.router.add_get("/api/status", status)
    app.router.add_get("/stream/{key}", stream)
    app.on_cleanup.append(cleanup)
    print(f"Four-camera live preview: http://127.0.0.1:{args.port}", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
