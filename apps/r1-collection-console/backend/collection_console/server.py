from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aiohttp import WSMsgType, web

from .state import build_file_snapshot


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
TELEIMAGER_SRC = WORKSPACE_ROOT / "robot" / "xr-teleoperate" / "teleop" / "teleimager" / "src"
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
BOUNDARY = "r1a7-collection-frame"


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


class CameraGateway:
    def __init__(self, host: str, client_factory: Callable[..., Any] | None = None):
        self.host = host
        self._client_factory = client_factory
        self._client = None
        self._config: dict[str, Any] = {}
        self._last_frames: dict[str, Any] = {}
        self._last_error: str | None = None
        self._last_connect_attempt = 0.0
        self._low_fps_since: dict[str, float | None] = {camera.key: None for camera in CAMERAS}

    def _factory(self):
        if self._client_factory is not None:
            return self._client_factory
        if str(TELEIMAGER_SRC) not in sys.path:
            sys.path.insert(0, str(TELEIMAGER_SRC))
        from teleimager.image_client import ImageClient

        return ImageClient

    def ensure_connected(self) -> None:
        if self._client is not None:
            return
        now = time.monotonic()
        if now - self._last_connect_attempt < 5.0:
            return
        self._last_connect_attempt = now
        try:
            self._client = self._factory()(host=self.host, request_bgr=False)
            self._config = self._client.get_cam_config()
            self._last_error = None
        except Exception as exc:  # hardware connection errors are reported to the UI
            self._client = None
            self._last_error = str(exc)

    def frame(self, camera: CameraView):
        self.ensure_connected()
        if self._client is None:
            return self._last_frames.get(camera.key)
        try:
            frame = getattr(self._client, camera.getter_name)()
        except Exception as exc:
            self._last_error = str(exc)
            return self._last_frames.get(camera.key)
        if frame is not None and getattr(frame, "jpg", None):
            self._last_frames[camera.key] = frame
        return self._last_frames.get(camera.key, frame)

    def status(self) -> dict[str, Any]:
        now_ns = time.monotonic_ns()
        now = time.monotonic()
        result: dict[str, Any] = {}
        for camera in CAMERAS:
            frame = self.frame(camera)
            config = self._config.get(camera.config_key, {})
            receive_ns = getattr(frame, "receive_monotonic_ns", None) if frame is not None else None
            age_ms = (now_ns - receive_ns) / 1_000_000 if receive_ns is not None else None
            fps = float(getattr(frame, "fps", 0.0) or 0.0) if frame is not None else 0.0
            online = bool(getattr(frame, "jpg", None)) and age_ms is not None and age_ms < 1000.0
            if online and 0 < fps < 12.0:
                self._low_fps_since[camera.key] = self._low_fps_since[camera.key] or now
            else:
                self._low_fps_since[camera.key] = None
            low_fps = (
                self._low_fps_since[camera.key] is not None
                and now - self._low_fps_since[camera.key] >= 3.0
            )
            image_shape = config.get("image_shape", (0, 0))
            height, width = image_shape if len(image_shape) >= 2 else (0, 0)
            result[camera.key] = {
                "key": camera.key,
                "title": camera.title,
                "online": online,
                "low_fps": low_fps,
                "fps": fps,
                "age_ms": age_ms,
                "width": int(width),
                "height": int(height),
                "sequence": getattr(frame, "frame_sequence", None) if frame is not None else None,
            }
        return {"source_host": self.host, "error": self._last_error, "items": result}

    def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()
        self._client = None


class CollectionMonitor:
    def __init__(
        self,
        task_dir: Path,
        scanner_log: Path,
        diagnostics_path: Path,
        camera_gateway: CameraGateway,
    ):
        self.task_dir = task_dir
        self.scanner_log = scanner_log
        self.diagnostics_path = diagnostics_path
        self.camera_gateway = camera_gateway

    def snapshot(self) -> dict[str, Any]:
        snapshot = build_file_snapshot(self.task_dir, self.scanner_log, self.diagnostics_path)
        snapshot["cameras"] = self.camera_gateway.status()
        return snapshot


CAMERA_GATEWAY_KEY = web.AppKey("camera_gateway", CameraGateway)
MONITOR_KEY = web.AppKey("monitor", CollectionMonitor)
PREVIEW_FPS_KEY = web.AppKey("preview_fps", float)
FRONTEND_DIST_KEY = web.AppKey("frontend_dist", Path)


def json_response(payload: Any) -> web.Response:
    return web.Response(
        text=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        content_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


async def snapshot_handler(request: web.Request) -> web.Response:
    monitor = request.app[MONITOR_KEY]
    payload = await asyncio.to_thread(monitor.snapshot)
    return json_response(payload)


async def events_handler(request: web.Request) -> web.Response:
    monitor = request.app[MONITOR_KEY]
    try:
        limit = max(1, min(500, int(request.query.get("limit", "200"))))
    except ValueError:
        raise web.HTTPBadRequest(text="limit must be an integer")
    payload = await asyncio.to_thread(monitor.snapshot)
    return json_response(payload["events"][:limit])


async def health_handler(request: web.Request) -> web.Response:
    return json_response({"status": "ok", "read_only": True})


async def stream_handler(request: web.Request) -> web.StreamResponse:
    camera = CAMERA_BY_KEY.get(request.match_info["key"])
    if camera is None:
        raise web.HTTPNotFound()
    gateway = request.app[CAMERA_GATEWAY_KEY]
    interval = 1.0 / request.app[PREVIEW_FPS_KEY]
    response = web.StreamResponse(
        headers={
            "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        }
    )
    await response.prepare(request)
    last_marker = None
    try:
        while True:
            frame = gateway.frame(camera)
            jpg = getattr(frame, "jpg", None) if frame is not None else None
            marker = (
                getattr(frame, "frame_sequence", None),
                getattr(frame, "receive_monotonic_ns", None),
            ) if frame is not None else None
            if jpg and marker != last_marker:
                last_marker = marker
                header = (
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpg)}\r\n\r\n"
                ).encode("ascii")
                await response.write(header + jpg + b"\r\n")
            await asyncio.sleep(interval)
    except (ConnectionResetError, asyncio.CancelledError, RuntimeError):
        pass
    return response


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    monitor = request.app[MONITOR_KEY]
    websocket = web.WebSocketResponse(heartbeat=20.0, compress=True)
    await websocket.prepare(request)
    last_payload = None
    try:
        while not websocket.closed:
            payload = await asyncio.to_thread(monitor.snapshot)
            serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if serialized != last_payload:
                await websocket.send_str(serialized)
                last_payload = serialized
            try:
                message = await websocket.receive(timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                break
    finally:
        await websocket.close()
    return websocket


async def index_handler(request: web.Request) -> web.FileResponse:
    index_path = request.app[FRONTEND_DIST_KEY] / "index.html"
    if not index_path.is_file():
        raise web.HTTPServiceUnavailable(
            text="Frontend is not built. Run the collection console frontend build first."
        )
    return web.FileResponse(index_path, headers={"Cache-Control": "no-store"})


async def cleanup(app: web.Application) -> None:
    await asyncio.to_thread(app[CAMERA_GATEWAY_KEY].close)


def create_app(
    *,
    task_dir: Path,
    scanner_log: Path,
    diagnostics_path: Path,
    source_host: str = "127.0.0.1",
    preview_fps: float = 15.0,
    frontend_dist: Path = FRONTEND_DIST,
    camera_gateway: CameraGateway | None = None,
) -> web.Application:
    gateway = camera_gateway or CameraGateway(source_host)
    monitor = CollectionMonitor(task_dir, scanner_log, diagnostics_path, gateway)
    app = web.Application()
    app[CAMERA_GATEWAY_KEY] = gateway
    app[MONITOR_KEY] = monitor
    app[PREVIEW_FPS_KEY] = preview_fps
    app[FRONTEND_DIST_KEY] = frontend_dist
    app.router.add_get("/api/snapshot", snapshot_handler)
    app.router.add_get("/api/events", events_handler)
    app.router.add_get("/healthz", health_handler)
    app.router.add_get("/stream/{key}", stream_handler)
    app.router.add_get("/ws/status", websocket_handler)
    assets = frontend_dist / "assets"
    if assets.is_dir():
        app.router.add_static("/assets", assets, show_index=False, append_version=True)
    app.router.add_get("/", index_handler)
    app.on_cleanup.append(cleanup)
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the read-only R1-A7 collection console.")
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--scanner-log", type=Path, required=True)
    parser.add_argument("--diagnostics-path", type=Path, required=True)
    parser.add_argument("--source-host", default="127.0.0.1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preview-fps", type=float, default=15.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in 1..65535")
    if not 1 <= args.preview_fps <= 30:
        raise SystemExit("--preview-fps must be in 1..30")
    app = create_app(
        task_dir=args.task_dir.expanduser().resolve(),
        scanner_log=args.scanner_log.expanduser().resolve(),
        diagnostics_path=args.diagnostics_path.expanduser().resolve(),
        source_host=args.source_host,
        preview_fps=args.preview_fps,
    )
    print(f"R1-A7 collection console: http://127.0.0.1:{args.port}", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
