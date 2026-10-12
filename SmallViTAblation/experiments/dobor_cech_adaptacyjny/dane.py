"""Deterministic protocol primitives; no torch or model imports."""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ARMS = ('random', 'glister', 'features_glister', 'features_kg_glister', 'features_placebo_glister')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def split_operations(rows, diagnostic_groups=12, seed=42):
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('split_operations accepts train rows only')
    groups = sorted({str(r['group_id']) for r in rows})
    if not 3 <= diagnostic_groups <= len(groups) - 3:
        raise ValueError('need at least three FIT and three diagnostic groups')
    chosen = set(np.random.default_rng(seed).choice(groups, diagnostic_groups, replace=False))
    ordered = sorted(rows, key=lambda r: str(r['frame_id']))
    return ([r for r in ordered if str(r['group_id']) not in chosen],
            [r for r in ordered if str(r['group_id']) in chosen])


def balanced_sample(rows, count, seed):
    """Round-robin over shuffled operation groups, without replacement."""
    if count < 1 or count > len(rows):
        raise ValueError(f'sample count {count} outside 1..{len(rows)}')
    by = defaultdict(list)
    for row in sorted(rows, key=lambda r: str(r['frame_id'])):
        by[str(row['group_id'])].append(str(row['frame_id']))
    rng = np.random.default_rng(seed)
    groups = sorted(by)
    rng.shuffle(groups)
    for values in by.values():
        rng.shuffle(values)
    result = []
    while len(result) < count:
        for group in groups:
            if by[group]:
                result.append(by[group].pop())
                if len(result) == count:
                    break
    return result


def validate_partitions(fit, diagnostic, val):
    for name, rows, split in [('FIT', fit, 'train'), ('D', diagnostic, 'train'), ('val', val, 'val')]:
        if not rows or any(r['split'] != split for r in rows):
            raise ValueError(f'{name}: empty or invalid source split')
        ids = [str(r['frame_id']) for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f'{name}: duplicate frame_id')
    sets = [{str(r['group_id']) for r in rows} for rows in (fit, diagnostic, val)]
    id_sets = [{str(r['frame_id']) for r in rows} for rows in (fit, diagnostic, val)]
    for i in range(3):
        for j in range(i):
            if id_sets[i] & id_sets[j]:
                raise ValueError('frame_id leakage between FIT, D and val')
            if sets[i] & sets[j]:
                raise ValueError('group leakage between FIT, D and val')
    # The same pixels must not reappear with a new name in another partition.
    hashes = [{r['pixel_sha256'] for r in rows if r.get('pixel_sha256')} for rows in (fit, diagnostic, val)]
    for i in range(3):
        for j in range(i):
            if hashes[i] & hashes[j]:
                raise ValueError('pixel leakage between partitions')


def safe_image(root, row):
    root = Path(root).resolve()
    path = (root / row['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'invalid image path: {row["frame_id"]}')
    return path


def coco(rows):
    images, annotations = [], []
    for index, row in enumerate(rows, 1):
        images.append(dict(id=index, file_name=row['image_path'], width=row['native_width'], height=row['native_height']))
        for ann in row['annotations']:
            x, y, w, h = map(float, ann['bbox'])
            annotations.append(dict(id=len(annotations)+1, image_id=index, category_id=1,
                                    bbox=[x, y, w, h], area=w*h, iscrowd=0))
    return dict(images=images, annotations=annotations, categories=[dict(id=1, name='instrument')], info={}, licenses=[])
