"""Jawne podzialy i rowny budzet; test obala wyciek operacji lub zmiane strumienia poza interwencja."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def load_pool(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    rows=[dict(r,frame_id=str(r['frame_id']),group_id=str(r['group_id'])) for r in rows]
    if len({r['frame_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate frame ids')
    groups = {}
    for r in rows:
        groups.setdefault(r['group_id'], set()).add(r['split'])
    if any(len(v) != 1 for v in groups.values()):
        raise ValueError('Operation leakage across splits')
    return rows


def sampling_indices(n, epoch_size, seed, epoch, new_index=None, presentations=1):
    """Paired schedules: same permutation; only logged slots replaced by the new frame."""
    if n < 1 or epoch_size < n:
        raise ValueError('epoch_size must cover old subset')
    rng = np.random.default_rng(seed + epoch * 100003)
    result = rng.permutation(n).tolist()
    result += rng.integers(0, n, size=epoch_size - n).tolist()
    slots = sorted(rng.choice(epoch_size, size=presentations, replace=False).tolist()) if new_index is not None else []
    if new_index is not None:
        if new_index != n or not 1 <= presentations <= epoch_size:
            raise ValueError('new sample must be appended and count valid')
        for j in slots:
            result[j] = new_index
    return result, slots


def make_dataset(rows, data_root, initial_ids, out, added=None):
    """Absolute COCO image paths avoid copying full frames; IDs preserve parent order."""
    out, data_root = Path(out), Path(data_root)
    byid = {r['frame_id']: r for r in rows}
    if len(byid)!=len(rows):raise ValueError('Duplicate frame ids')
    groups={}
    for row in rows:groups.setdefault(row['group_id'],set()).add(row['split'])
    if any(len(v)!=1 for v in groups.values()):raise ValueError('Operation leakage across splits')
    ids = list(initial_ids)
    if len(ids) != len(set(ids)) or any(byid[i]['split'] != 'train' for i in ids):
        raise ValueError('Invalid train selection')
    if added is not None:
        if added in ids or byid[added]['split'] != 'train':
            raise ValueError('Added frame not unused training data')
        ids.append(added)
    mapping = {}
    for split, dirname in [('train','train'), ('val','valid'), ('test','test')]:
        chosen = [byid[i] for i in ids] if split == 'train' else [r for r in rows if r['split'] == split]
        images, annotations = [], []
        for j, r in enumerate(chosen):
            path = (data_root / r['path']).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            images.append(dict(id=j, file_name=str(path), width=r['native_width'], height=r['native_height']))
            for a in r.get('annotations', []):
                bbox = a['bbox']
                if min(bbox[2:]) <= 0:
                    raise ValueError('Empty box')
                annotations.append(dict(id=len(annotations), image_id=j, category_id=1, bbox=bbox,
                                        area=bbox[2]*bbox[3], iscrowd=0))
        coco = dict(images=images, annotations=annotations, categories=[dict(id=1, name='instrument', supercategory='instrument')], info={})
        write_json(out/dirname/'_annotations.coco.json', coco)
        mapping[split] = [dict(frame_id=r['frame_id'], group_id=r['group_id']) for r in chosen]
    write_json(out/'mapping.json', mapping)
    return out
