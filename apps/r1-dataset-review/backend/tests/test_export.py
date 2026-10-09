import importlib.util
import json
from pathlib import Path

import pytest

from dataset_review.annotations import AnnotationStore, ConflictError
from dataset_review.catalog import Catalog
from dataset_review.export import build_export, write_export
from test_catalog import converted, raw  # noqa: F401


def setup(converted, tmp_path):
    dataset = Catalog([converted]).datasets[0]
    return dataset, AnnotationStore(tmp_path / 'labels'), dataset.trajectories('full')[0]


def mark(store, trajectory, start, end, exclude=True):
    record = store.get(trajectory)
    record['quality_intervals'] = [{'label': '恢复', 'category': 'recovery',
        'start_frame': start, 'end_frame_exclusive': end, 'exclude_from_training': exclude}]
    return store.save(trajectory, record)


@pytest.mark.parametrize('start,end,chunk', [(0, 1, 2), (2, 4, 2), (3, 4, 50), (5, 6, 1)])
def test_action_start_oracle_and_scan_boundary(converted, tmp_path, start, end, chunk):
    dataset, store, trajectory = setup(converted, tmp_path)
    mark(store, trajectory, start, end)
    summary, files = build_export(dataset, store, include_training_filter=True, action_chunk_size=chunk)
    payload = json.loads(files['training_filter.json'])
    excluded = {i for r in payload['excluded_ranges'] for i in range(r['dataset_start'], r['dataset_end_exclusive'])}
    oracle = {i for base in (0, 3) for i in range(base, base + 3)
              if set(range(i, min(i + chunk, base + 3))) & set(range(start, end))}
    assert excluded == oracle
    assert summary['allowed_training_starts'] == 6 - len(oracle)
    project = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location('split_filter', project / 'scripts/lerobot/make_r1a7_episode_split.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert sum(module._allowed_mask(payload, 6)) == summary['allowed_training_starts']
    sampler_spec = importlib.util.spec_from_file_location('review_sampler', project / 'data/unitree-lerobot/unitree_lerobot/lerobot/src/lerobot/datasets/sampler.py')
    sampler_module = importlib.util.module_from_spec(sampler_spec)
    sampler_spec.loader.exec_module(sampler_module)
    sampler = sampler_module.EpisodeAwareSampler([0, 3], [3, 6], excluded_ranges=[
        (r['dataset_start'], r['dataset_end_exclusive']) for r in payload['excluded_ranges']])
    assert set(sampler) == set(range(6)) - oracle


def test_recovery_retained_and_whole_exclusion(converted, tmp_path):
    dataset, store, trajectory = setup(converted, tmp_path)
    mark(store, trajectory, 1, 5, False)
    _, files = build_export(dataset, store, include_training_filter=True)
    assert json.loads(files['training_filter.json'])['excluded_ranges'] == []
    record = store.get(trajectory)
    record['review_status'] = 'excluded'
    store.save(trajectory, record)
    summary, _ = build_export(dataset, store, include_training_filter=True)
    assert summary['excluded_training_starts'] == 6


def test_snapshot_conflict_and_no_overwrite(converted, tmp_path):
    dataset, store, trajectory = setup(converted, tmp_path)
    summary, _ = build_export(dataset, store)
    mark(store, trajectory, 0, 1)
    output = tmp_path / 'export'
    with pytest.raises(ConflictError):
        write_export(dataset, store, output, expected_snapshot_hash=summary['snapshot_sha256'])
    assert not output.exists()
    result = write_export(dataset, store, output)
    assert (output / 'COMPLETE.json').is_file()
    assert 'training_filter.json' not in result['files']
    before = (output / 'annotations.json').read_bytes()
    with pytest.raises(FileExistsError):
        write_export(dataset, store, output)
    assert (output / 'annotations.json').read_bytes() == before
    with pytest.raises(ValueError):
        write_export(dataset, store, converted / 'export')


def test_raw_snapshot_only(raw, tmp_path):
    dataset = Catalog([raw]).datasets[0]
    store = AnnotationStore(tmp_path / 'labels')
    _, files = build_export(dataset, store)
    assert 'source_mapping.jsonl' in files
    with pytest.raises(ValueError, match='LeRobot'):
        build_export(dataset, store, include_training_filter=True)


def test_changed_metadata(converted, tmp_path):
    dataset, store, _ = setup(converted, tmp_path)
    (converted / 'meta/info.json').write_text('{}')
    with pytest.raises(ConflictError):
        build_export(dataset, store)


def test_same_name_different_batches_not_combined(converted, tmp_path):
    path = converted / 'meta/source_segments.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1].update(source_batch='batch-b', start_frame=0, end_frame_exclusive=3)
    path.write_text('\n'.join(map(json.dumps, rows)))
    dataset, store, first = setup(converted, tmp_path)
    mark(store, first, 0, 3)
    summary, files = build_export(dataset, store, include_training_filter=True)
    assert summary['excluded_training_starts'] == 3
    assert {r['output_episode_index'] for r in json.loads(files['training_filter.json'])['excluded_ranges']} == {0}


def test_segments_without_source_mapping(converted, tmp_path):
    (converted / 'meta/source_segments.jsonl').unlink()
    dataset = Catalog([converted]).datasets[0]
    store = AnnotationStore(tmp_path / 'labels')
    with pytest.raises(ValueError):
        build_export(dataset, store)
    mark(store, dataset.trajectories('segments')[1], 0, 1)
    summary, files = build_export(dataset, store, view='segments', include_training_filter=True)
    assert summary['excluded_training_starts'] == 1
    assert json.loads(files['training_filter.json'])['excluded_ranges'][0]['dataset_start'] == 3
