from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web

from .catalog import Catalog, safe_path
from .frames import FrameReader
from .annotations import AnnotationStore, ConflictError
from .export import build_export, write_export


def create_app(catalog, frontend, annotation_root=None):
    reader = FrameReader()
    decode_slot = asyncio.Semaphore(1)
    store = AnnotationStore(annotation_root or Path(__file__).resolve().parents[4] / 'runtime/annotations/dataset-review')
    if any(store.root.is_relative_to(dataset.root) for dataset in catalog.datasets):
        raise ValueError('标注目录不能放入原始数据目录')

    @web.middleware
    async def errors(request, handler):
        try:
            if urlsplit('//'+request.host).hostname not in {'127.0.0.1', 'localhost', '::1'}:
                raise PermissionError('只允许本机访问')
            return await handler(request)
        except ConflictError as error:
            return web.json_response({'error': str(error)}, status=409)
        except PermissionError as error:
            return web.json_response({'error': str(error)}, status=403)
        except (ValueError, KeyError, FileNotFoundError) as error:
            return web.json_response({"error": str(error)}, status=400)

    app = web.Application(middlewares=[errors], client_max_size=256 * 1024)

    def dataset(request):
        return catalog.dataset(request.match_info["dataset"])

    def trajectory(request):
        return dataset(request).trajectory(request.match_info["trajectory"])

    async def datasets(request):
        return web.json_response([d.summary() for d in catalog.datasets])

    async def trajectories(request):
        def listing():
            return [{**t.summary(), 'review_status': store.get(t)['review_status']}
                    for t in dataset(request).trajectories(request.query.get('view', 'full'))]
        return web.json_response(await asyncio.to_thread(listing))

    async def annotation(request):
        value = trajectory(request)
        if request.method == 'GET':
            return web.json_response(await asyncio.to_thread(store.get, value))
        if request.headers.get('Origin') != f'{request.scheme}://{request.host}' or request.headers.get('X-R1-Annotation') != '1':
            raise PermissionError('拒绝跨站或无来源的标注写入')
        try:
            saved = await asyncio.to_thread(store.save, value, await request.json())
        except OSError as error:
            return web.json_response({'error': f'标注写入失败，旧文件未覆盖：{error}'}, status=500)
        return web.json_response(saved)

    async def detail(request):
        value = trajectory(request)
        result = await asyncio.to_thread(value.detail)
        return web.json_response({**result, "timestamps_s": [value.timestamp(i) for i in range(value.length)]})

    async def export_preview(request):
        enabled = request.query.get('include_training_filter', 'false')
        if enabled not in {'true', 'false'}:
            raise ValueError('训练过滤开关无效')
        summary, _ = await asyncio.to_thread(build_export, dataset(request), store,
            request.query.get('view', 'full'), enabled == 'true', int(request.query.get('action_chunk_size', '50')))
        return web.json_response(summary)

    async def export_save(request):
        if request.headers.get('Origin') != f'{request.scheme}://{request.host}' or request.headers.get('X-R1-Annotation') != '1':
            raise PermissionError('拒绝跨站导出写入')
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {'view', 'include_training_filter', 'action_chunk_size', 'expected_snapshot_hash'}:
            raise ValueError('导出参数不正确')
        if not isinstance(body['expected_snapshot_hash'], str) or len(body['expected_snapshot_hash']) != 64:
            raise ValueError('请先预览导出')
        value = dataset(request)
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
        output = safe_path(store.root, f'{value.id}/exports/{name}')
        try:
            result = await asyncio.to_thread(write_export, value, store, output, **body)
        except OSError as error:
            return web.json_response({'error': f'导出未完成，请勿使用缺少 COMPLETE.json 的目录：{error}'}, status=500)
        return web.json_response(result)

    async def frame(request):
        async with decode_slot:
            result = await asyncio.to_thread(reader.bundle, trajectory(request), int(request.match_info["frame"]))
        return web.json_response(result)

    async def index(request):
        return web.FileResponse(frontend / "index.html")

    async def cleanup(app):
        reader.close()

    app.router.add_get("/api/datasets", datasets)
    app.router.add_get("/api/datasets/{dataset}/trajectories", trajectories)
    app.router.add_get("/api/datasets/{dataset}/trajectories/{trajectory}", detail)
    app.router.add_get("/api/datasets/{dataset}/trajectories/{trajectory}/frames/{frame}", frame)
    app.router.add_get('/api/datasets/{dataset}/trajectories/{trajectory}/annotation', annotation)
    app.router.add_put('/api/datasets/{dataset}/trajectories/{trajectory}/annotation', annotation)
    app.router.add_get('/api/datasets/{dataset}/export-preview', export_preview)
    app.router.add_post('/api/datasets/{dataset}/exports', export_save)
    app.router.add_get("/", index)
    if (frontend / "assets").is_dir():
        app.router.add_static("/assets", frontend / "assets")
    app.on_cleanup.append(cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description="四路数据集只读同步回放，不连接机器人")
    parser.add_argument("--dataset-root", type=Path, action="append", required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument('--annotation-root', type=Path, help='独立标注存储目录，不能位于数据根内')
    args = parser.parse_args()
    frontend = Path(__file__).resolve().parents[2] / "frontend/dist"
    if not (frontend / "index.html").is_file():
        parser.error("前端尚未构建，请在 frontend 中执行 npm run build")
    catalog = Catalog(args.dataset_root)
    web.run_app(create_app(catalog, frontend, args.annotation_root), host="127.0.0.1", port=args.port, handler_cancellation=True)


if __name__ == "__main__":
    main()
