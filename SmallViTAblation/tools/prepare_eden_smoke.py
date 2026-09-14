#!/usr/bin/env python3
"""Build a small, auditable original-frame COCO smoke dataset from Eden's old pool.

This is a plumbing check, not a scientific selection baseline. No test data are read.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tarfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def subset(source, split_name, split_map, budget):
    groups = defaultdict(list)
    for image in source['images']:
        if image.get('aug_source') or '_aug_' in image['file_name']:
            continue
        match = re.fullmatch(r'((?:train|test)\d+)_frame_(\d+)\.(?:jpg|png)', image['file_name'])
        if not match:
            raise ValueError(f"Unrecognized original frame: {image['file_name']}")
        group, frame = match.group(1), int(match.group(2))
        if split_map.get(group) != split_name:
            raise ValueError(f'Wrong split: {group} in {split_name}')
        groups[group].append((frame, image))
    if budget < len(groups) or budget > sum(map(len, groups.values())):
        raise ValueError('Budget must cover each video and fit the original-frame pool')
    quotas = {g: 0 for g in groups}
    for _ in range(budget):
        eligible = [g for g in sorted(groups) if quotas[g] < len(groups[g])]
        g = min(eligible, key=lambda k: quotas[k])
        quotas[g] += 1
    selected = []
    for group in sorted(groups):
        ordered = sorted(groups[group], key=lambda x: x[0])
        for k in range(quotas[group]):
            index = min(len(ordered)-1, int((k + 0.5) * len(ordered) / quotas[group]))
            selected.append(dict(ordered[index][1], video_id=group))
    ids = {im['id'] for im in selected}
    if len(ids) != budget or len({im['file_name'] for im in selected}) != budget:
        raise ValueError('Duplicate image IDs or paths')
    annotations = [a for a in source['annotations'] if a['image_id'] in ids]
    cats = {c['id'] for c in source['categories']}
    image_index = {im['id']: im for im in selected}
    for ann in annotations:
        box = ann['bbox']
        im = image_index[ann['image_id']]
        if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                or min(box[2:]) <= 0 or ann['category_id'] not in cats
                or min(box[:2]) < -0.01 or box[0]+box[2] > im['width']+0.01
                or box[1]+box[3] > im['height']+0.01):
            raise ValueError(f'Invalid annotation {ann}')
    return {'info': {'purpose': 'smoke only'}, 'images': selected,
            'annotations': annotations, 'categories': source['categories']}, quotas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations-dir', required=True, type=Path)
    p.add_argument('--images-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--train-size', type=int, default=64)
    p.add_argument('--val-size', type=int, default=64)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f'Refusing to overwrite {a.output}')
    from PIL import Image
    split_path = a.annotations_dir / 'video_split.json'
    split_map = json.loads(split_path.read_text())['video_split']
    selected = {}
    provenance = {'purpose': 'technical_smoke_not_scientific_result',
                  'selection': 'equal_video_quotas_temporal_bin_midpoints',
                  'split_sha256': digest(split_path), 'splits': {}, 'images_sha256': {}}
    for name, size in [('train', a.train_size), ('val', a.val_size)]:
        src = a.annotations_dir / f'annotations_{name}.json'
        coco, quotas = subset(json.loads(src.read_text()), name, split_map, size)
        selected[name] = coco
        provenance['splits'][name] = {'source': str(src.resolve()),
            'source_sha256': digest(src), 'images': size, 'annotations': len(coco['annotations']),
            'videos': quotas}
    if selected['train']['categories'] != selected['val']['categories']:
        raise ValueError('Category mappings differ')
    seen = set()
    for name, coco in selected.items():
        for im in coco['images']:
            src = a.images_dir / im['file_name']
            with Image.open(src) as image:
                if image.size != (im['width'], im['height']):
                    raise ValueError(f'Image dimensions differ: {src}')
                image.verify()
            sha = digest(src)
            if sha in seen:
                raise ValueError(f'Duplicate image content: {src}')
            seen.add(sha)
            provenance['images_sha256'][im['file_name']] = sha
    (a.output / 'images').mkdir(parents=True)
    for name, coco in selected.items():
        for im in coco['images']:
            shutil.copy2(a.images_dir / im['file_name'], a.output / 'images' / im['file_name'])
            im['file_name'] = 'images/' + im['file_name']
        write_json(a.output / f'annotations_{name}.json', coco)
    shutil.copy2(split_path, a.output / 'video_split.json')
    write_json(a.output / 'smoke_data_manifest.json', provenance)
    (a.output / 'variants').mkdir()
    write_json(a.output / 'variants' / 'smoke64.json', {
        'seed': 42, 'budget': a.train_size, 'strategy': 'smoke_temporal_video_balanced',
        'annotations_train': 'annotations_train.json',
        'frame_ids': [im['file_name'] for im in selected['train']['images']],
        'data_manifest_sha256': digest(a.output / 'smoke_data_manifest.json')})
    archive = a.output.with_suffix('.tar')
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, 'w') as tar:
        for path in sorted(a.output.iterdir()):
            tar.add(path, arcname=path.name)
    write_json(archive.with_suffix('.tar.sha256.json'), {'sha256': digest(archive)})
    print(json.dumps(provenance['splits'], indent=2))
    print(archive)


if __name__ == '__main__':
    main()
