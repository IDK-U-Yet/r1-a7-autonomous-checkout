"""Versioned annotation snapshots and episode-bounded action-start filters."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .annotations import AnnotationStore, ConflictError
from .catalog import Catalog


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def merge_ranges(ranges):
    result = []
    for start, end in sorted(ranges):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def build_export(dataset, store, view='full', include_training_filter=False, action_chunk_size=50):
    if type(include_training_filter) is not bool:
        raise ValueError('训练过滤开关必须是布尔值')
    if type(action_chunk_size) is not int or not 1 <= action_chunk_size <= 10000:
        raise ValueError('action_chunk_size 必须为 1～10000 的整数')
    if include_training_filter and dataset.format != 'lerobot-v3':
        raise ValueError('原始数据只能导出标注；训练过滤需要转换后的 LeRobot 数据集')
    with store.lock:
        if not dataset.source_unchanged():
            raise ConflictError('源元信息已变化，请重启工作台')
        trajectories = dataset.trajectories(view)
        records = [store.get(t) for t in trajectories]
        snapshot = {'schema_version': 1, 'dataset_root': str(dataset.root),
                    'dataset_id': dataset.id, 'metadata_fingerprint': dataset.fingerprint,
                    'view': view, 'annotations': records}
        digest = hashlib.sha256(encoded(snapshot).encode()).hexdigest()
        totals = Counter(r['review_status'] for r in records)
        warnings = []
        if totals['unreviewed']:
            warnings.append(f"{totals['unreviewed']} 条尚未审核，默认保留，并非自动通过")
        if totals['needs_review']:
            warnings.append(f"{totals['needs_review']} 条待复核，除明确排除的区间外仍保留")
        other_view = 'segments' if view == 'full' else 'full'
        if (other_view != 'full' or dataset._full) and any(
                store.get(t)['revision'] for t in dataset.trajectories(other_view)
                if t.id not in {t.id for t in trajectories}):
            warnings.append('另一个回放视图有独立标注，本次不合并；只导出当前视图')
        total = sum(s.length for s in dataset.segments)
        if dataset.format == 'lerobot-v3':
            cursor = 0
            for segment in sorted(dataset.segments, key=lambda s: s.index):
                if segment.row['dataset_from_index'] != cursor:
                    raise ValueError('LeRobot 全局帧范围存在缺口或重叠')
                cursor = int(segment.row['dataset_to_index'])
            if cursor != total or total != dataset.info['total_frames']:
                raise ValueError('LeRobot 总帧数与分段元信息不一致')
        mapped, provenance, excluded, sources = [], [], {}, []
        for trajectory, record in zip(trajectories, records):
            offset = 0
            for segment in trajectory.segments:
                global_start = int(segment.row['dataset_from_index']) if dataset.format == 'lerobot-v3' else None
                sources.append({'trajectory_id': trajectory.id, 'source_batch': trajectory.batch,
                                'source_episode_path': segment.source_path, 'output_episode_index': segment.index,
                                'source_frame_range': [segment.source_start, segment.source_start + segment.length],
                                'trajectory_frame_range': [offset, offset + segment.length],
                                'dataset_frame_range': [global_start, global_start + segment.length] if global_start is not None else None})
                ranges = excluded.setdefault(segment.index, [])
                if record['review_status'] == 'excluded':
                    ranges.append([0, segment.length])
                    provenance.append({'trajectory_id': trajectory.id, 'output_episode_index': segment.index,
                                       'reason': 'review_status=excluded', 'local_range': [0, segment.length]})
                for layer in ('stages', 'quality_intervals'):
                    for interval in record[layer]:
                        start = max(0, interval['start_frame'] - offset)
                        end = min(segment.length, interval['end_frame_exclusive'] - offset)
                        if start >= end:
                            continue
                        exclude = layer == 'quality_intervals' and interval['exclude_from_training']
                        expanded = [max(0, start - action_chunk_size + 1), end] if exclude else None
                        entry = {'trajectory_id': trajectory.id, 'source_batch': trajectory.batch,
                                 'source_episode_path': segment.source_path,
                                 'output_episode_index': segment.index, 'layer': layer, **interval,
                                 'trajectory_frame_range': [start + offset, end + offset],
                                 'source_frame_range': [segment.source_start + start, segment.source_start + end],
                                 'segment_local_range': [start, end],
                                 'dataset_frame_range': [global_start + start, global_start + end] if global_start is not None else None,
                                 'excluded_training_start_local_range': expanded}
                        mapped.append(entry)
                        if expanded:
                            ranges.append(expanded)
                            provenance.append({'trajectory_id': trajectory.id, 'output_episode_index': segment.index,
                                               'reason': interval['label'], 'category': interval['category'],
                                               'local_range': expanded})
                offset += segment.length
        filter_ranges = []
        for segment in dataset.segments:
            for start, end in merge_ranges(excluded.get(segment.index, [])):
                if dataset.format == 'lerobot-v3':
                    base = int(segment.row['dataset_from_index'])
                    filter_ranges.append({'output_episode_index': segment.index, 'local_start': start,
                                          'local_end_exclusive': end, 'dataset_start': base + start,
                                          'dataset_end_exclusive': base + end})
        count = sum(r['dataset_end_exclusive'] - r['dataset_start'] for r in filter_ranges)
        summary = {'dataset_id': dataset.id, 'view': view, 'snapshot_sha256': digest,
                   'metadata_fingerprint': dataset.fingerprint, 'trajectory_count': len(records),
                   'review_status_counts': dict(totals), 'total_frames': total,
                   'include_training_filter': include_training_filter,
                   'action_chunk_size': action_chunk_size if include_training_filter else None,
                   'excluded_training_starts': count if include_training_filter else None,
                   'allowed_training_starts': total - count if include_training_filter else None,
                   'warnings': warnings}
        files = {'annotations.json': encoded(snapshot), 'summary.json': encoded(summary),
                 'source_mapping.jsonl': ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sources),
                 'interval_mapping.jsonl': ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in mapped)}
        if include_training_filter:
            files['training_filter.json'] = encoded({
                'version': 1, 'dataset_root': str(dataset.root), 'dataset_name': dataset.name,
                'metadata_fingerprint': dataset.fingerprint, 'annotations_sha256': digest,
                'action_chunk_size': action_chunk_size, 'total_frames': total,
                'excluded_training_starts': count, 'allowed_training_starts': total - count,
                'excluded_ranges': filter_ranges, 'provenance': provenance})
        if not dataset.source_unchanged():
            raise ConflictError('导出时源元信息变化，已停止')
        return summary, files


def write_export(dataset, store, output, view='full', include_training_filter=False,
                 action_chunk_size=50, expected_snapshot_hash=None):
    with store.lock:
        summary, files = build_export(dataset, store, view, include_training_filter, action_chunk_size)
        if expected_snapshot_hash is not None and expected_snapshot_hash != summary['snapshot_sha256']:
            raise ConflictError('预览后标注已变化，请重新预览再导出')
        output = Path(output).resolve()
        if output.is_relative_to(dataset.root) or dataset.root.is_relative_to(output):
            raise ValueError('导出目录不能覆盖或位于数据根目录中')
        output.mkdir(parents=True, exist_ok=False)
        # Exclusive creates never overwrite earlier exports. COMPLETE is published last.
        for name, content in files.items():
            with (output / name).open('x', encoding='utf-8') as handle:
                handle.write(content)
        with (output / 'COMPLETE.json').open('x', encoding='utf-8') as handle:
            handle.write(encoded({'snapshot_sha256': summary['snapshot_sha256'], 'files': list(files)}))
        return {**summary, 'output_dir': str(output), 'files': list(files)}


def main():
    parser = argparse.ArgumentParser(description='独立导出人工标注及 LeRobot 动作块起始帧过滤')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--annotation-root', type=Path, required=True, help='工作台标注根目录，不是 dataset_id 子目录')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--view', choices=['full', 'segments'], default='full')
    parser.add_argument('--include-training-filter', action='store_true')
    parser.add_argument('--action-chunk-size', type=int, default=50)
    args = parser.parse_args()
    dataset = Catalog([args.dataset_root]).datasets[0]
    store = AnnotationStore(args.annotation_root)
    if store.root.is_relative_to(dataset.root):
        parser.error('标注目录不能位于源数据内')
    result = write_export(dataset, store, args.output_dir, args.view, args.include_training_filter, args.action_chunk_size)
    print(encoded(result))


if __name__ == '__main__':
    main()
