from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from collection_console.server import CAMERAS, create_app


class FakeCameraGateway:
    def __init__(self):
        self.closed = False

    def status(self):
        return {
            "source_host": "test",
            "error": None,
            "items": {
                camera.key: {
                    "key": camera.key,
                    "title": camera.title,
                    "online": False,
                    "low_fps": False,
                    "fps": 0.0,
                    "age_ms": None,
                    "width": 0,
                    "height": 0,
                    "sequence": None,
                }
                for camera in CAMERAS
            },
        }

    def frame(self, _camera):
        return None

    def close(self):
        self.closed = True


def test_read_only_http_and_websocket_contract(tmp_path: Path):
    async def exercise():
        frontend = tmp_path / "frontend"
        frontend.mkdir()
        (frontend / "index.html").write_text("<main>console</main>", encoding="utf-8")
        gateway = FakeCameraGateway()
        app = create_app(
            task_dir=tmp_path / "batch",
            scanner_log=tmp_path / "scanner.jsonl",
            diagnostics_path=tmp_path / "diagnostics.jsonl",
            frontend_dist=frontend,
            camera_gateway=gateway,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            health = await (await client.get("/healthz")).json()
            assert health == {"status": "ok", "read_only": True}

            snapshot = await (await client.get("/api/snapshot")).json()
            assert snapshot["batch"]["phase"] == "waiting"
            assert set(snapshot["cameras"]["items"]) == {
                "global",
                "left_wrist",
                "right_wrist",
                "scanner",
            }

            index = await client.get("/")
            assert index.status == 200
            assert await index.text() == "<main>console</main>"

            websocket = await client.ws_connect("/ws/status")
            message = await websocket.receive_json(timeout=2.0)
            assert message["batch"]["accepted"] == 0
            await websocket.close()
        finally:
            await client.close()
        assert gateway.closed is True

    asyncio.run(exercise())
