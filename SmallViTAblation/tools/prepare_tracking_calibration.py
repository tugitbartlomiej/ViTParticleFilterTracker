#!/usr/bin/env python3
"""Freeze annotated tracking calibration data from an audited candidate index.

No selection on validation, no test image materialization, no assumed negatives.
This is calibration, not a comparison of curation methods.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SmallViTAblation.pipeline.curation.data import ImageSource, write_json
from SmallViTAblation.pipeline.curation.selection import allocate


def temporal_sample(rows, budget):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row['group_id']].append(row)
    quotas = allocate({g: len(v) for g, v in by_group.items()}, budget, 'sqrt')
    selected = []
    for group in sorted(by_group):
        candidates = sorted(by_group[group], key=lambda r: (r['video_id'], r['frame_index']))
        n, k = len(candidates), quotas[group]
        selected.extend(candidates[(2 * i + 1) * n // (2 * k)] for i in range(k))
    assert len(selected) == budget and len({r['frame_id'] for r in selected}) == budget
    return selected


def materialize(rows, destination, source, pixel_splits):
    images, annotations = [], []
    for index, original in enumerate(rows, 1):
        image = source.image(original)
        if image.size != (original['native_width'], original['native_height']):
            raise ValueError(f"Incorrect source dimensions: {original['frame_id']}")
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
        previous = pixel_splits.setdefault(pixel_hash, original['split'])
        if previous != original['split']:
            raise ValueError('Exact decoded image overlap between train and validation')
        relative = 'images/' + original['video_id'] + f"_{original['frame_index']:08d}.jpg"
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.bytes(original))
        row = dict(original, path=relative, pixel_sha256=pixel_hash)
        row.pop('archive', None)
        row.pop('member', None)
        images.append(dict(id=index, file_name=relative, width=image.width,
                           height=image.height, frame_id=row['frame_id'],
                           group_id=row['group_id'], video_id=row['video_id']))
        for ann in row['annotations']:
            bbox = ann['bbox']
            annotations.append(dict(id=len(annotations)+1, image_id=index, category_id=1,
                                    bbox=bbox, area=bbox[2]*bbox[3], iscrowd=ann.get('iscrowd', 0)))
        yield row, images, annotations


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new output path; never mix prepared datasets')
    started = time.monotonic()
    rows = [json.loads(line) for line in (args.index_dir/'pool_manifest.jsonl').read_text().splitlines()]
    eligible = [r for r in rows if r['label_status'] == 'provided']
    train = temporal_sample([r for r in eligible if r['split'] == 'train'], 500)
    val = [r for r in eligible if r['split'] == 'val']
    if not val:
        raise ValueError('Validation is empty')
    groups_train, groups_val = ({r['group_id'] for r in pool} for pool in (train, val))
    assert not groups_train & groups_val
    args.out.mkdir(parents=True)
    source = ImageSource()
    pixel_splits = {}
    exported = {}
    try:
        for name, pool in [('train', train), ('val', val)]:
            final_rows, images, annotations = [], [], []
            for row, images, annotations in materialize(pool, args.out, source, pixel_splits):
                final_rows.append(row)
            exported[name] = final_rows
            write_json(args.out/f'annotations_{name}.json', dict(images=images,
                annotations=annotations, categories=[dict(id=1, name='instrument', supercategory='tool')]))
            print(f'[export] {name}: {len(images)} images', flush=True)
    finally:
        source.close()
    with (args.out/'pool_manifest.jsonl').open('w') as stream:
        for row in exported['train']+exported['val']:
            stream.write(json.dumps(row)+'\n')
    write_json(args.out/'split_manifest.json', json.loads((args.index_dir/'split_manifest.json').read_text()))
    for seed in [42, 1337, 2024]:
        write_json(args.out/f'variants/calibration_s{seed}.json', dict(
            run_id=f'tracking_uniform_b500_s{seed}_calibration', strategy='uniform_time',
            seed=seed, budget=500, annotations_train='annotations_train.json',
            frame_ids=[r['frame_id'] for r in train], scope='training_recipe_calibration',
            selection_seed_used=False, candidate_grid_stride=5))
    smoke_train = temporal_sample(train, 64)
    smoke_val = temporal_sample(val, 64)
    for name, pool in [('train', smoke_train), ('val', smoke_val)]:
        full = json.loads((args.out/f'annotations_{name}.json').read_text())
        ids = {r['frame_id'] for r in pool}
        ims = [im for im in full['images'] if im['frame_id'] in ids]
        image_ids = {im['id'] for im in ims}
        write_json(args.out/f'smoke_annotations_{name}.json', dict(
            images=ims, annotations=[a for a in full['annotations'] if a['image_id'] in image_ids],
            categories=full['categories']))
    write_json(args.out/'variants/smoke64.json', dict(seed=42, budget=64,
        annotations_train='smoke_annotations_train.json', frame_ids=[r['frame_id'] for r in smoke_train]))
    summary = dict(scope='tracking_recipe_calibration_only', source_target='cystotome_instrument_box',
        split_unit='conservative_operation_groups', patient_disjointness_verified=False,
        candidate_stride=5, annotation_policy='provided_bbox_only',
        empty_labels_not_interpreted_as_negative=True, test_images_exported=0,
        candidate_counts=dict(Counter(r['split'] for r in rows)),
        eligible_counts=dict(Counter(r['split'] for r in eligible)),
        train_images=len(train), val_images=len(val), train_groups=len(groups_train),
        val_groups=len(groups_val), preparation_seconds=time.monotonic()-started,
        candidate_index_sha256=hashlib.sha256((args.index_dir/'pool_manifest.jsonl').read_bytes()).hexdigest())
    write_json(args.out/'data_manifest.json', summary)
    archive = args.out.with_suffix('.tar')
    with tarfile.open(archive, 'w') as out:
        for path in sorted(args.out.rglob('*')):
            if path.is_file():
                out.add(path, arcname=path.relative_to(args.out), recursive=False)
    with archive.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    write_json(archive.with_suffix('.tar.sha256.json'), dict(sha256=sha, bytes=archive.stat().st_size))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
