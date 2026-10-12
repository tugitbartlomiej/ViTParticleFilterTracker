"""Wspólny kontrakt train: rzeczywisty czas, zgodność ID i brak GT w cechach.

Pool treningu jest rzadką unią historycznych wyborów. Nigdy nie budujemy z niej
fikcyjnie gęstych okien. Cache zawiera osobny gęsty kontekst tych samych grup.
Znaczniki mają jednostkę klatki, nie sekundy (FPS nie został potwierdzony).
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def temporal_windows(rows, context=8, step=5):
    if context < 1 or step <= 0:
        raise ValueError('context and step must be positive')
    ids = [str(r['frame_id']) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate frame IDs')
    by = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get('split') != 'train':
            raise ValueError('context must be explicit train only')
        by[str(r['video_id'])].append(i)
    windows = []
    for indices in by.values():
        indices.sort(key=lambda i: rows[i]['frame_index'])
        if len({str(rows[i]['group_id']) for i in indices}) != 1:
            raise ValueError('one video cannot belong to several groups')
        for end in range(context, len(indices)):
            seq = indices[end-context:end+1]
            gaps = np.diff([float(rows[i]['frame_index']) for i in seq])
            if np.all(np.isclose(gaps, step, atol=1e-6, rtol=0)):
                windows.append(seq)
    return np.asarray(windows, dtype=np.int64).reshape(-1, context+1)


def target_box(row):
    if len(row.get('annotations', [])) != 1:
        raise ValueError('pilot expects exactly one annotated cystotome')
    x, y, w, h = np.asarray(row['annotations'][0]['bbox'], dtype=float)
    width, height = float(row['native_width']), float(row['native_height'])
    if not np.isfinite([x,y,w,h,width,height]).all() or min(w,h,width,height) <= 0:
        raise ValueError('invalid target box')
    return np.array([(x+w/2)/width, (y+h/2)/height, w/width, h/height])


def load_data(pool_path, cache_path, l0_path):
    pool_all = read_jsonl(pool_path)
    pool = [r for r in pool_all if r.get('split') == 'train']
    val_groups = {str(r['group_id']) for r in pool_all if r.get('split') != 'train'}
    with np.load(cache_path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files if k != 'meta'}
        meta = json.loads(str(data['meta']))
    n = len(arrays['frame_ids'])
    rows = [dict(frame_id=str(arrays['frame_ids'][i]), video_id=str(arrays['video_id'][i]),
                 group_id=str(arrays['group_id'][i]), frame_index=float(arrays['frame_index'][i]),
                 split='train') for i in range(n)]
    if val_groups & {r['group_id'] for r in rows}:
        raise ValueError('evaluation groups in feature context')
    index = {r['frame_id']: i for i,r in enumerate(rows)}
    if len(index) != n or len({r['frame_id'] for r in pool}) != len(pool):
        raise ValueError('duplicate frame IDs')
    if any(r['frame_id'] not in index for r in pool):
        raise ValueError('candidate missing from feature cache')
    if any(str(r['group_id']) != rows[index[r['frame_id']]]['group_id'] for r in pool):
        raise ValueError('group mismatch between pool and cache')
    raw = json.loads(Path(l0_path).read_text(encoding='utf-8'))
    l0 = list(map(str, raw.get('frame_ids', raw.get('selected_frame_ids', []))))
    if not l0 or len(l0) != len(set(l0)) or not set(l0) <= {r['frame_id'] for r in pool}:
        raise ValueError('invalid shared initial selection')
    for key in ('features', 'inputs', 'observed'):
        if len(arrays[key]) != n or not np.isfinite(arrays[key]).all():
            raise ValueError(f'invalid cache array {key}')
    return pool, rows, arrays, l0, meta
