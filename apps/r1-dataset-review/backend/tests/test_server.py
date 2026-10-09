import asyncio

from aiohttp.test_utils import TestClient, TestServer

from dataset_review.catalog import Catalog
from dataset_review.server import create_app
from test_catalog import raw  # noqa: F401


def test_read_only_api_and_index_errors(raw, tmp_path):
    async def check():
        async with TestClient(TestServer(create_app(Catalog([raw]), tmp_path))) as client:
            response = await client.get('/api/datasets')
            dataset = (await response.json())[0]
            assert dataset['full_count'] == 1
            base = f"/api/datasets/{dataset['id']}/trajectories"
            trajectory = (await (await client.get(base)).json())[0]
            detail = await (await client.get(base + '/' + trajectory['id'])).json()
            assert len(detail['timestamps_s']) == 6
            frame = await client.get(base + '/' + trajectory['id'] + '/frames/5')
            assert (await frame.json())['frame_index'] == 5
            invalid = await client.get(base + '/' + trajectory['id'] + '/frames/6')
            assert invalid.status == 400
            assert (await client.post(base)).status == 405
    asyncio.run(check())


def test_annotation_write_origin_and_revision(raw, tmp_path):
    async def check():
        source = (raw / 'episode_0007/data.json').read_bytes()
        async with TestClient(TestServer(create_app(Catalog([raw]), tmp_path, tmp_path / 'labels'))) as client:
            dataset = (await (await client.get('/api/datasets')).json())[0]
            base = f"/api/datasets/{dataset['id']}/trajectories"
            row = (await (await client.get(base)).json())[0]
            path = base + '/' + row['id'] + '/annotation'
            value = await (await client.get(path)).json()
            value['review_status'] = 'approved'
            assert (await client.put(path, json=value)).status == 403
            assert (await client.put(path, json=value, headers={'Origin':'https://other.test','X-R1-Annotation':'1'})).status == 403
            headers = {'Origin': str(client.make_url('/')).rstrip('/'), 'X-R1-Annotation':'1'}
            response = await client.put(path, json=value, headers=headers)
            assert response.status == 200
            assert (await response.json())['revision'] == 1
            assert (await client.put(path, json=value, headers=headers)).status == 409
            assert (await (await client.get(base)).json())[0]['review_status'] == 'approved'
        assert (raw / 'episode_0007/data.json').read_bytes() == source
    asyncio.run(check())


def test_export_preview_origin_and_new_directory(raw, tmp_path):
    async def check():
        root = tmp_path / 'labels'
        async with TestClient(TestServer(create_app(Catalog([raw]), tmp_path, root))) as client:
            dataset = (await (await client.get('/api/datasets')).json())[0]
            base = f"/api/datasets/{dataset['id']}"
            preview = await (await client.get(base + '/export-preview')).json()
            assert preview['trajectory_count'] == 1
            assert (await client.get(base + '/export-preview?include_training_filter=true')).status == 400
            value = {'view': 'full', 'include_training_filter': False, 'action_chunk_size': 50,
                     'expected_snapshot_hash': preview['snapshot_sha256']}
            assert (await client.post(base + '/exports', json=value)).status == 403
            headers = {'Origin': str(client.make_url('/')).rstrip('/'), 'X-R1-Annotation': '1'}
            response = await client.post(base + '/exports', json=value, headers=headers)
            assert response.status == 200
            result = await response.json()
            assert result['files'] == ['annotations.json', 'summary.json', 'source_mapping.jsonl', 'interval_mapping.jsonl']
            from pathlib import Path
            assert Path(result['output_dir']).is_relative_to(root)
            assert (Path(result['output_dir']) / 'COMPLETE.json').is_file()
            assert (await client.post(base + '/exports', json={**value, 'output_dir': '/tmp/escape'}, headers=headers)).status == 400
    asyncio.run(check())
