import copy

import pytest

from dataset_review.annotations import AnnotationStore, ConflictError
from dataset_review.catalog import Catalog
from test_catalog import raw  # noqa: F401


@pytest.fixture
def sample(raw, tmp_path):
    return AnnotationStore(tmp_path / 'annotations'), Catalog([raw]).datasets[0].trajectories('full')[0]


def test_roundtrip_and_conflict(sample):
    store, trajectory = sample
    original = store.get(trajectory)
    record = copy.deepcopy(original)
    record.update(episode_type='tabletop_recovery', orientation='O3', review_status='approved', notes='test')
    record['stages'] = [{'label': '抓取', 'start_frame': 0, 'end_frame_exclusive': 6}]
    saved = store.save(trajectory, record)
    assert saved['revision'] == 1
    assert store.get(trajectory) == saved
    with pytest.raises(ConflictError):
        store.save(trajectory, original)
    assert store.get(trajectory) == saved


@pytest.mark.parametrize('start,end', [(0, 0), (-1, 2), (0, 7), (1.0, 3), (False, 3)])
def test_bad_interval(sample, start, end):
    store, trajectory = sample
    value = store.get(trajectory)
    value['stages'] = [{'label': '抓取', 'start_frame': start, 'end_frame_exclusive': end}]
    with pytest.raises(ValueError):
        store.save(trajectory, value)


def test_overlap_and_recovery_retained(sample):
    store, trajectory = sample
    value = store.get(trajectory)
    value['stages'] = [{'label': '抓取', 'start_frame': 0, 'end_frame_exclusive': 3}]
    value['quality_intervals'] = [{'label': '恢复', 'category': 'recovery', 'start_frame': 0,
                                   'end_frame_exclusive': 3, 'exclude_from_training': False}]
    assert store.save(trajectory, value)['quality_intervals'][0]['exclude_from_training'] is False
    value = store.get(trajectory)
    value['stages'].append({'label': '扫码', 'start_frame': 2, 'end_frame_exclusive': 4})
    with pytest.raises(ValueError, match='重叠'):
        store.save(trajectory, value)


def test_source_change_and_failed_replace(sample, monkeypatch):
    store, trajectory = sample
    saved = store.save(trajectory, store.get(trajectory))
    def fail(*args):
        raise OSError('simulated disk error')
    monkeypatch.setattr('dataset_review.annotations.os.replace', fail)
    with pytest.raises(OSError):
        store.save(trajectory, saved)
    assert store.get(trajectory) == saved
    trajectory.dataset.fingerprint = 'changed'
    with pytest.raises(ConflictError):
        store.get(trajectory)


def test_path_escape_and_unknown_schema(sample, tmp_path):
    store, trajectory = sample
    value = store.get(trajectory)
    value['schema_version'] = 2
    with pytest.raises(ConflictError):
        store.save(trajectory, value)
    store.root.mkdir()
    (store.root / trajectory.dataset.id).symlink_to(tmp_path)
    with pytest.raises(ValueError, match='路径'):
        store.get(trajectory)


def test_source_metadata_change_on_restart(raw, tmp_path):
    store = AnnotationStore(tmp_path / 'labels')
    before = Catalog([raw]).datasets[0].trajectories('full')[0]
    store.save(before, store.get(before))
    source = raw / 'episode_0007/data.json'
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ConflictError):
        store.save(before, store.get(before))
    after = Catalog([raw]).datasets[0].trajectories('full')[0]
    with pytest.raises(ConflictError):
        store.get(after)
