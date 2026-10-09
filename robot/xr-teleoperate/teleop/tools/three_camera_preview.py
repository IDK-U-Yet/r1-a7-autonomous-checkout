#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Read-only browser preview for the three locally attached UVC cameras."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import glob
import json
from pathlib import Path
import signal
import threading
import time

import cv2
from aiohttp import web


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>三路相机测试</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, "Noto Sans SC", "Microsoft YaHei", sans-serif;
      background: #f3f5f4;
      color: #17211d;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; }
    header {
      height: 64px;
      padding: 0 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid #d7ddda;
      background: #ffffff;
    }
    h1 { margin: 0; font-size: 20px; font-weight: 650; letter-spacing: 0; }
    #overall { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #52615a; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #c73e35; }
    .dot.online { background: #278052; }
    main {
      width: min(1480px, 100%);
      margin: 0 auto;
      padding: 18px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    section {
      min-width: 0;
      overflow: hidden;
      border: 1px solid #ccd5d0;
      border-radius: 6px;
      background: #ffffff;
    }
    section.head { grid-column: 1 / -1; }
    .viewport {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      overflow: hidden;
      background: #111614;
    }
    section:not(.head) .viewport { aspect-ratio: 4 / 3; }
    img { display: block; width: 100%; height: 100%; object-fit: contain; }
    .meta {
      min-height: 54px;
      padding: 9px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-top: 1px solid #d7ddda;
    }
    .name { min-width: 0; }
    .name strong { display: block; font-size: 14px; font-weight: 650; }
    .device {
      display: block;
      margin-top: 2px;
      overflow: hidden;
      color: #67756e;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 11px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .metrics { flex: 0 0 auto; color: #33443c; font-size: 12px; text-align: right; }
    .error { color: #b4312a; }
    @media (max-width: 760px) {
      header { height: 56px; padding: 0 14px; }
      main { grid-template-columns: 1fr; padding: 10px; gap: 10px; }
      section.head { grid-column: auto; }
      section:not(.head) .viewport { aspect-ratio: 16 / 9; }
      .meta { align-items: flex-start; flex-direction: column; gap: 4px; }
      .metrics { text-align: left; }
    }
  </style>
</head>
<body>
  <header>
    <h1>三路相机测试</h1>
    <div id="overall"><span class="dot"></span><span>正在连接</span></div>
  </header>
  <main>
    <section class="head" data-camera="head">
      <div class="viewport"><img src="/stream/head" alt="全局相机实时画面"></div>
      <div class="meta"><div class="name"><strong>全局相机</strong><span class="device"></span></div><div class="metrics">--</div></div>
    </section>
    <section data-camera="wrist_jr0002">
      <div class="viewport"><img src="/stream/wrist_jr0002" alt="JR0002 腕部相机实时画面"></div>
      <div class="meta"><div class="name"><strong>腕部相机 JR0002</strong><span class="device"></span></div><div class="metrics">--</div></div>
    </section>
    <section data-camera="wrist_jr0001">
      <div class="viewport"><img src="/stream/wrist_jr0001" alt="JR0001 腕部相机实时画面"></div>
      <div class="meta"><div class="name"><strong>腕部相机 JR0001</strong><span class="device"></span></div><div class="metrics">--</div></div>
    </section>
  </main>
  <script>
    async function refresh() {
      try {
        const response = await fetch('/api/status', {cache: 'no-store'});
        const cameras = await response.json();
        let online = 0;
        for (const [key, status] of Object.entries(cameras)) {
          const section = document.querySelector(`[data-camera="${key}"]`);
          if (!section) continue;
          section.querySelector('.device').textContent = status.device;
          const metrics = section.querySelector('.metrics');
          if (status.state === 'online') {
            online += 1;
            metrics.classList.remove('error');
            metrics.textContent = `${status.width}×${status.height} · ${status.fps.toFixed(1)} FPS · ${Math.round(status.age_ms)} ms`;
          } else {
            metrics.classList.add('error');
            metrics.textContent = status.error || status.state;
          }
        }
        const overall = document.querySelector('#overall');
        overall.querySelector('.dot').classList.toggle('online', online === 3);
        overall.querySelector('span:last-child').textContent = `${online}/3 路在线`;
      } catch (error) {
        document.querySelector('#overall span:last-child').textContent = '状态服务断开';
      }
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class CameraSpec:
    key: str
    serial_hint: str
    width: int
    height: int
    fps: int
    rotation_degrees: int = 0


class CameraCapture:
    def __init__(
        self, spec: CameraSpec, device: str, jpeg_quality: int, preview_fps: float
    ):
        self.spec = spec
        self.device = device
        self.jpeg_quality = jpeg_quality
        self.preview_fps = preview_fps
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._capture_thread = threading.Thread(
            target=self._run, name=f"camera-{spec.key}", daemon=True
        )
        self._preview_thread = threading.Thread(
            target=self._encode_preview, name=f"preview-{spec.key}", daemon=True
        )
        self._raw_frame = None
        self._raw_sequence = 0
        self._jpeg: bytes | None = None
        self._jpeg_sequence = 0
        self._last_frame_at = 0.0
        self._fps = 0.0
        self._width = 0
        self._height = 0
        self._state = "starting"
        self._error = ""

    def start(self) -> None:
        self._capture_thread.start()
        self._preview_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._capture_thread.join(timeout=3.0)
        self._preview_thread.join(timeout=3.0)

    def snapshot(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._jpeg, self._jpeg_sequence

    def status(self) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            age_ms = (now - self._last_frame_at) * 1000.0 if self._last_frame_at else 0.0
            state = self._state
            if state == "online" and age_ms > 1000.0:
                state = "stale"
            return {
                "device": self.device,
                "state": state,
                "error": self._error,
                "fps": self._fps,
                "width": self._width,
                "height": self._height,
                "age_ms": age_ms,
                "sequence": self._raw_sequence,
                "rotation_degrees": self.spec.rotation_degrees,
            }

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._state = "error"
            self._error = message

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                self._set_error("无法打开设备")
                return
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
            cap.set(cv2.CAP_PROP_FPS, self.spec.fps)
            if self.spec.key == "head":
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

            window_started = time.monotonic()
            window_frames = 0
            consecutive_failures = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        self._set_error("连续读取失败")
                        return
                    time.sleep(0.01)
                    continue
                consecutive_failures = 0
                if self.spec.rotation_degrees == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                now = time.monotonic()
                window_frames += 1
                elapsed = now - window_started
                with self._lock:
                    self._raw_frame = frame
                    self._raw_sequence += 1
                    self._last_frame_at = now
                    self._height, self._width = frame.shape[:2]
                    self._state = "online"
                    self._error = ""
                    if elapsed >= 1.0:
                        self._fps = window_frames / elapsed
                if elapsed >= 1.0:
                    window_started = now
                    window_frames = 0
        except Exception as exc:
            self._set_error(f"{type(exc).__name__}: {exc}")
        finally:
            cap.release()

    def _encode_preview(self) -> None:
        interval = 1.0 / self.preview_fps
        last_raw_sequence = -1
        while not self._stop.wait(interval):
            with self._lock:
                frame = self._raw_frame
                raw_sequence = self._raw_sequence
            if frame is None or raw_sequence == last_raw_sequence:
                continue
            encoded, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if not encoded:
                continue
            with self._lock:
                self._jpeg = buffer.tobytes()
                self._jpeg_sequence += 1
            last_raw_sequence = raw_sequence


def discover_device(serial_hint: str) -> str:
    candidates = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    matches = [path for path in candidates if serial_hint in Path(path).name]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one video-index0 device containing {serial_hint!r}, found {matches}"
        )
    return matches[0]


async def index(_request: web.Request) -> web.Response:
    return web.Response(text=PAGE, content_type="text/html")


async def status(request: web.Request) -> web.Response:
    cameras: dict[str, CameraCapture] = request.app["cameras"]
    return web.json_response({key: camera.status() for key, camera in cameras.items()})


async def stream(request: web.Request) -> web.StreamResponse:
    cameras: dict[str, CameraCapture] = request.app["cameras"]
    key = request.match_info["key"]
    camera = cameras.get(key)
    if camera is None:
        raise web.HTTPNotFound(text="unknown camera")

    response = web.StreamResponse(
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-store, no-cache, must-revalidate",
        }
    )
    await response.prepare(request)
    last_sequence = -1
    try:
        while True:
            jpeg, sequence = camera.snapshot()
            if jpeg is not None and sequence != last_sequence:
                await response.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode("ascii")
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                last_sequence = sequence
            await asyncio.sleep(0.01)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return response


async def cleanup(app: web.Application) -> None:
    for camera in app["cameras"].values():
        camera.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--preview-fps", type=float, default=15.0)
    parser.add_argument("--head-device")
    parser.add_argument("--jr0002-device")
    parser.add_argument("--jr0001-device")
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
        CameraSpec("wrist_jr0002", "JR0002", 640, 480, 30),
        CameraSpec("wrist_jr0001", "JR0001", 640, 480, 30),
        CameraSpec("head", "CN02KX4NLG0004ABK00", 1920, 1080, 30),
    )
    overrides = {
        "head": args.head_device,
        "wrist_jr0002": args.jr0002_device,
        "wrist_jr0001": args.jr0001_device,
    }
    cameras: dict[str, CameraCapture] = {}
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


def raise_keyboard_interrupt() -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    main()
