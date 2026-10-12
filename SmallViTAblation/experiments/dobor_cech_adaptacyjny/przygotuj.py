"""Prepare one shared protocol on Eden after extracting archives to node-local disk."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .cechy import extract_features
from .dane import (read_jsonl, write_json, write_jsonl, sha256, split_operations,
                   balanced_sample, validate_partitions, safe_image, coco)


def prepare(data_root, context_root, out, config_path, annotations=None):
    data_root, context_root, out = map(Path, (data_root, context_root, out))
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    if config['candidate_limit'] < config['budget'] or config['support_frames'] > config['budget']:
        raise ValueError('candidate_limit >= budget >= support_frames required')
    if config['reference_frames'] > config['diagnostic_frames']:
        raise ValueError('reference_frames must fit D')
    source = read_jsonl(data_root/'pool_manifest.jsonl')
    # Test records are discarded before opening images or extracting features.
    train = [r for r in source if r['split'] == 'train']
    val = [r for r in source if r['split'] == 'val']
    fit, d_all = split_operations(train, config['diagnostic_groups'], config['protocol_seed'])
    if config.get('integration_fixture'):
        fit_ids = set(balanced_sample(fit, config['fit_cap'], config['protocol_seed']+7))
        fit = [r for r in fit if r['frame_id'] in fit_ids]
        val_ids = set(balanced_sample(val, config['val_cap'], config['protocol_seed']+8))
        val = [r for r in val if r['frame_id'] in val_ids]
    d_ids = set(balanced_sample(d_all, config['diagnostic_frames'], config['protocol_seed']+1))
    diagnostic = [r for r in d_all if r['frame_id'] in d_ids]
    validate_partitions(fit, d_all, val)
    if len(fit) < config['candidate_limit']:
        raise ValueError('candidate_limit exceeds FIT pool; reduce it explicitly')
    if len({r['group_id'] for r in diagnostic}) < config['min_groups']:
        raise ValueError('insufficient diagnostic operation groups')
    extra = {}
    if annotations:
        for row in read_jsonl(annotations):
            if row['frame_id'] in extra:
                raise ValueError('duplicated feature annotation')
            extra[row['frame_id']] = row
    for rows in (fit, diagnostic, val):
        for row in rows:
            if len(row['annotations']) > 1:
                raise ValueError('this experiment requires at most one target bbox per frame')
            if rows is not val and len(row['annotations']) != 1:
                raise ValueError('FIT/D pilot requires one annotated instrument bbox per frame')
            if any(a.get('source_category', 'cystotome') != 'cystotome' for a in row['annotations']):
                raise ValueError('one surgical tool only: cystotome')
            row['image_path'] = str(safe_image(data_root, row))
            row['width'], row['height'] = row['native_width'], row['native_height']
    support_ids = balanced_sample(fit, config['support_frames'], config['protocol_seed']+2)
    reference_ids = balanced_sample(diagnostic, config['reference_frames'], config['protocol_seed']+3)
    # Context is restricted to train operations involved in FIT/D. Never val/test.
    train_groups = {str(r['group_id']) for r in fit+d_all}
    context = []
    seen = set()
    for row in read_jsonl(context_root/'pool_manifest.jsonl'):
        if row['split'] != 'train' or str(row['group_id']) not in train_groups:
            continue
        if row['frame_id'] in seen:
            raise ValueError('duplicate context frame')
        seen.add(row['frame_id'])
        row['image_path'] = str(safe_image(context_root, row))
        context.append(row)
    missing = {r['frame_id'] for r in fit+diagnostic} - seen
    if missing:
        raise ValueError(f'context lacks {len(missing)} labelled pool frames')
    context.sort(key=lambda r: (str(r['video_id']), float(r['frame_index'])))
    if config.get('integration_fixture'):
        # Small Eden-only test fixture: keep target history, never future frames.
        required = {(r['video_id'], float(r['frame_index'])-lag*config['context_step'])
                    for r in fit+diagnostic for lag in range(config['gle_context']+1)}
        context = [r for r in context if (r['video_id'], float(r['frame_index'])) in required]
    out.mkdir(parents=True, exist_ok=False)
    feature_rows = []
    names = None
    for row in fit+diagnostic:
        with Image.open(row['image_path']) as im:
            values = extract_features(row, im.convert('RGB'), extra.get(row['frame_id']))
        names = list(values) if names is None else names
        if list(values) != names:
            raise ValueError('feature schema changed')
        feature_rows.append([np.nan if values[n] is None else values[n] for n in names])
    np.save(out/'features.npy', np.array(feature_rows, dtype=np.float32))
    for name, rows in [('fit', fit), ('diagnostic', diagnostic), ('val', val), ('context', context)]:
        write_jsonl(out/f'{name}.jsonl', rows)
    for name, rows in [('train', fit), ('valid', val)]:
        write_json(out/'coco'/name/'_annotations.coco.json', coco(rows))
    write_json(out/'config.json', config)
    metadata = dict(schema_version=1, feature_names=names, fit_count=len(fit), diagnostic_count=len(diagnostic),
                    support_ids=support_ids, reference_ids=reference_ids,
                    excluded_fit_groups=sorted({str(r['group_id']) for r in d_all}),
                    pool_manifest_sha256=sha256(data_root/'pool_manifest.jsonl'),
                    context_manifest_sha256=sha256(context_root/'pool_manifest.jsonl'),
                    annotations_sha256=sha256(annotations) if annotations else None,
                    config_sha256=sha256(config_path),
                    assignment_sha256=hashlib.sha256(json.dumps({
                        'FIT': [r['frame_id'] for r in fit], 'D': [r['frame_id'] for r in diagnostic],
                        'val': [r['frame_id'] for r in val], 'support': support_ids,
                        'reference': reference_ids}, sort_keys=True).encode()).hexdigest(),
                    files_sha256={p.relative_to(out).as_posix(): sha256(p) for p in out.rglob('*') if p.is_file()},
                    val_used_for_selection=False, test_images_opened=False)
    write_json(out/'protocol.json', metadata)
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'context-root', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    p.add_argument('--annotations', type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.data_root, a.context_root, a.out, a.config, a.annotations), indent=2))


if __name__ == '__main__':
    main()
