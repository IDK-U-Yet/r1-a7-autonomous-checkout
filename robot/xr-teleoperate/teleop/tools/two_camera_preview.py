#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Read-only preview for the 640x480 top and upside-down added cameras."""

from __future__ import annotations

import argparse
import json
import signal
import time

from aiohttp import web

from three_camera_preview import (
    CameraCapture,
    CameraSpec,
    cleanup,
    discover_device,
    raise_keyboard_interrupt,
    status,
    stream,
)


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>双路相机测试</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, "Noto Sans SC", "Microsoft YaHei", sans-serif;
      background: #f2f4f3;
      color: #18211d;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; }
    header {
      min-height: 62px;
      padding: 12px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid #d5dbd8;
      background: #fff;
    }
    h1 { margin: 0; font-size: 19px; font-weight: 650; letter-spacing: 0; }
    #overall { display: flex; align-items: center; gap: 8px; color: #53615a; font-size: 13px; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #c33c34; }
    .dot.online { background: #237d4f; }
    main {
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 16px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    section {
      min-width: 0;
      overflow: hidden;
      border: 1px solid #cbd3cf;
      border-radius: 6px;
      background: #fff;
    }
    .viewport {
      width: 100%;
      aspect-ratio: 4 / 3;
      overflow: hidden;
      background: #111614;
    }
    section.top .viewport { aspect-ratio: 16 / 9; }
    img { display: block; width: 100%; height: 100%; object-fit: contain; }
    .meta {
      min-height: 62px;
      padding: 9px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-top: 1px solid #d5dbd8;
    }
    .name { min-width: 0; }
    strong { display: block; font-size: 14px; }
    .device {
      display: block;
      margin-top: 3px;
      overflow: hidden;
      color: #66746d;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 11px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .metrics { flex: 0 0 auto; color: #33433b; font-size: 12px; text-align: right; }
    .error { color: #b4312a; }
    @media (max-width: 760px) {
      header { align-items: flex-start; flex-direction: column; }
      main { grid-template-columns: 1fr; padding: 10px; }
      .meta { align-items: flex-start; flex-direction: column; }
      .metrics { text-align: left; }
    }
  </style>
</head>
<body>
  <header>
    <h1>双路相机测试 · 顶部 640×360 / 新增 640×480</h1>
    <div id="overall"><span class="dot"></span><span>正在连接</span></div>
  </header>
  <main>
    <section class="top" data-camera="top">
      <div class="viewport"><img src="/stream/top" alt="顶部相机实时画面"></div>
      <div class="meta"><div class="name"><strong>顶部相机</strong><span class="device"></span></div><div class="metrics">--</div></div>
    </section>
    <section data-camera="added">
      <div class="viewport"><img src="/stream/added" alt="新增相机实时画面"></div>
      <div class="meta"><div class="name"><strong>新增相机 · 已旋转 180°</strong><span class="device"></span></div><div class="metrics">--</div></div>
    </section>
  </main>
  <script>
    async function refresh() {
      try {
        const response = await fetch('/api/status', {cache: 'no-store'});
        const cameras = await response.json();
        let online = 0;
        for (const [key, camera] of Object.entries(cameras)) {
          const section = document.querySelector(`[data-camera="${key}"]`);
          if (!section) continue;
          section.querySelector('.device').textContent = camera.device;
          const metrics = section.querySelector('.metrics');
          if (camera.state === 'online') {
            online += 1;
            metrics.classList.remove('error');
            metrics.textContent = `${camera.width}×${camera.height} · ${camera.fps.toFixed(1)} FPS · ${Math.round(camera.age_ms)} ms`;
          } else {
            metrics.classList.add('error');
            metrics.textContent = camera.error || camera.state;
          }
        }
        const overall = document.querySelector('#overall');
        overall.querySelector('.dot').classList.toggle('online', online === 2);
        overall.querySelector('span:last-child').textContent = `${online}/2 路在线`;
      } catch (_error) {
        document.querySelector('#overall span:last-child').textContent = '状态服务断开';
      }
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


async def index(_request: web.Request) -> web.Response:
    return web.Response(text=PAGE, content_type="text/html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--preview-fps", type=float, default=15.0)
    parser.add_argument("--top-device")
    parser.add_argument("--added-device")
    parser.add_argument(
        "--added-rotation",
        type=int,
        choices=(0, 180),
        default=180,
        help="Rotate the added camera in software for mounting orientation.",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if not 30 <= args.jpeg_quality <= 95:
        parser.error("--jpeg-quality must be in 30..95")
    if not 1.0 <= args.preview_fps <= 30.0:
        parser.error("--preview-fps must be in 1..30")
    return args


def main() -> None:
    args = parse_args()
    specs = (
        CameraSpec("top", "CN02KX4NLG0004ABK00", 640, 360, 30),
        CameraSpec("added", "AY275520029", 640, 480, 30, args.added_rotation),
    )
    overrides = {"top": args.top_device, "added": args.added_device}
    cameras = {}
    for spec in specs:
        device = overrides[spec.key] or discover_device(spec.serial_hint)
        camera = CameraCapture(spec, device, args.jpeg_quality, args.preview_fps)
        cameras[spec.key] = camera
        camera.start()
        time.sleep(0.25)

    app = web.Application()
    app["cameras"] = cameras
    app.router.add_get("/", index)
    app.router.add_get("/api/status", status)
    app.router.add_get("/stream/{key}", stream)
    app.on_cleanup.append(cleanup)

    print(json.dumps({key: camera.device for key, camera in cameras.items()}, ensure_ascii=False))
    print(f"Preview: http://127.0.0.1:{args.port}")
    signal.signal(signal.SIGTERM, lambda *_args: raise_keyboard_interrupt())
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
