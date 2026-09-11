"""Curate Cataract-LMM training frames with simple/balanced/dynamics profiles.

Examples from the repository root (no training is launched):
    python -m SmallViTAblation.pipeline.curation.run --subset tracking \
        --profile simple --budget 2500 --out SmallViTAblation/curated/tracking
    python -m SmallViTAblation.pipeline.curation.run --subset tracking \
        --profile dynamics --budget 2500 --out SmallViTAblation/curated/tracking
    python -m SmallViTAblation.pipeline.curation.run --subset segmentation \
        --profile balanced --budget 1500 --out SmallViTAblation/curated/segmentation

--split-map reuses an existing group->split mapping. Otherwise a deterministic
operation split is saved before features are read. --candidate-labels all
--no-export produces an annotation acquisition list, including unknown labels.
--export-evaluation exports the fixed candidate val/test sets without curation.
They are sampled by the declared input stride, not claimed to be every video frame.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .data import (ImageSource, assign_splits, digest, export_coco,
                   segmentation_rows, tracking_rows, write_json)
from .dynamics import diagnose
from .features import build_features
from .selection import choose, measurements

REPOSITORY = Path(__file__).resolve().parents[3]


def jsonl(path, records):
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', type=Path, default=REPOSITORY/'Dataset/Cataract-LMM')
    ap.add_argument('--subset', choices=('tracking', 'segmentation'), default='tracking')
    ap.add_argument('--profile', choices=('simple', 'balanced', 'dynamics', 'all'), default='balanced')
    ap.add_argument('--budget', type=int, required=True, help='Exact number of unique training images')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42, help='Selection and diagnostic seed')
    ap.add_argument('--split-seed', type=int, default=42, help='Independent, fixed data split seed')
    ap.add_argument('--split-map', type=Path)
    ap.add_argument('--split-fractions', type=float, nargs=3, default=(.7, .15, .15),
                    metavar=('TRAIN', 'VAL', 'TEST'))
    ap.add_argument('--audit', type=Path, help='Tracking operation audit; defaults to dataset root audit')
    ap.add_argument('--frame-stride', type=int, default=5,
                    help='Dense tracking candidate grid; segmentation keeps all provided images')
    ap.add_argument('--fps', type=float, help='Only provide a verified fixed frame rate; default time unit is frames')
    ap.add_argument('--candidate-labels', choices=('provided', 'all'), default='provided',
                    help='provided excludes unknown empty tracking labels, reported in manifest')
    ap.add_argument('--no-export', action='store_true', help='Write selection and measurements without copying images')
    ap.add_argument('--export-evaluation', action='store_true', help='Also export fixed val/test candidates, without selection')
    ap.add_argument('--accept-empty-tracking-labels', action='store_true',
                    help='Explicitly interpret empty tracking CSV boxes as negatives for COCO export')
    ap.add_argument('--cache-dir', type=Path, default=REPOSITORY/'SmallViTAblation/feature_cache/curation')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--encoder', default='dinov2_vits14')
    ap.add_argument('--hub-repo', default='facebookresearch/dinov2:main', help='Pinned Hub ref or local DINOv2 checkout recommended')
    ap.add_argument('--weights', type=Path, help='Local DINOv2 backbone state_dict')
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--device', help='e.g. cuda:0 or cpu; affects feature extraction only')
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--allocation', choices=('sqrt', 'equal', 'proportional'), default='sqrt')
    ap.add_argument('--anchor-fraction', type=float, default=.25)
    ap.add_argument('--min-gap-frames', type=int, default=5)
    ap.add_argument('--spacing-policy', choices=('relax', 'strict'), default='relax',
                    help='relax reports every spacing violation needed to reach exact K')
    ap.add_argument('--projection-dim', type=int, default=64)
    ap.add_argument('--importance-weight', type=float, default=.5)
    ap.add_argument('--signal', choices=('history_gain', 'residual'), default='history_gain')
    ap.add_argument('--lags', type=int, default=3)
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--dynamics-dim', type=int, default=24)
    ap.add_argument('--max-transition-gap-frames', type=float,
                    help='Default 1.5 times the tracking input stride; never bridge larger gaps')
    ap.add_argument('--ridge-alpha', type=float, default=.01)
    ap.add_argument('--max-fit-windows', type=int, default=20000)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.frame_stride < 1 or args.budget < 1:
        raise ValueError('Positive frame stride and budget required')
    if args.fps is not None and (not np.isfinite(args.fps) or args.fps <= 0):
        raise ValueError('FPS must be finite and positive')
    if args.image_size < 14 or args.projection_dim < 1:
        raise ValueError('Invalid feature resolution or dimension')
    for name in ('anchor_fraction', 'importance_weight', 'ridge_alpha', 'max_transition_gap_frames'):
        value = getattr(args, name)
        if value is not None and not np.isfinite(value):
            raise ValueError(f'{name} must be finite')
    if args.max_transition_gap_frames is not None and args.max_transition_gap_frames <= 0:
        raise ValueError('max-transition-gap-frames must be positive')
    if args.no_export and args.export_evaluation:
        raise ValueError('--no-export and --export-evaluation are mutually exclusive')
    started = time.perf_counter()
    if args.subset == 'tracking':
        rows, import_meta = tracking_rows(args.dataset_root, args.frame_stride, args.fps, args.audit)
    else:
        rows, import_meta = segmentation_rows(args.dataset_root, args.frame_stride, args.fps)
    rows.sort(key=lambda r: r['frame_id'])
    split = assign_splits(rows, args.split_seed, tuple(args.split_fractions), args.split_map)
    # Accepting empties and asking for them are distinct choices, both recorded.
    # U: every observable training frame may define temporal context and a
    # frozen representation. A: only frames whose label policy permits query
    # or training can be selected. This prevents label availability from
    # destroying the very temporal windows used by diagnostics.
    observation_pool = [r for r in rows if r['split'] == 'train']
    eligible_indices = [i for i, r in enumerate(observation_pool)
                        if args.candidate_labels == 'all' or r['label_status'] == 'provided']
    pool = [observation_pool[i] for i in eligible_indices]
    if len(pool) < args.budget:
        raise ValueError(f'Budget {args.budget} exceeds eligible training candidates {len(pool)}')
    args.out.mkdir(parents=True, exist_ok=True)
    source_files = {}
    for row in rows:
        path = row.get('archive') or row['path']
        if path not in source_files:
            stat = Path(path).stat()
            source_files[path] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    source_contract = dict(importer=import_meta, split=split, records_sha256=digest(rows),
                           source_files=source_files)
    contract_hash = digest(source_contract)
    root_manifest = args.out/'pool_identity.json'
    if root_manifest.exists() and json.loads(root_manifest.read_text())['identity'] != contract_hash:
        raise ValueError('Output directory already belongs to a different candidate pool/split')
    write_json(root_manifest, dict(identity=contract_hash, **source_contract))
    write_json(args.out/'split_manifest.json', split)
    jsonl(args.out/'pool_manifest.jsonl', rows)
    profiles = ('simple', 'balanced', 'dynamics') if args.profile == 'all' else (args.profile,)
    source = ImageSource()
    feature_sets = {}
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}
    try:
        for profile in profiles:
            run_config = dict(config, profile=profile)
            run_id = f'C-{profile}__{args.subset}__s{args.seed}__b{args.budget}__{digest(run_config)[:10]}'
            destination = args.out/run_id
            if destination.exists():
                raise FileExistsError(f'Run output already exists: {destination}; choose another --out')
            semantic = profile != 'simple'
            if semantic not in feature_sets:
                feature_sets[semantic] = build_features(observation_pool, source, args.cache_dir,
                    semantic=semantic, model=args.encoder, hub_repo=args.hub_repo,
                    weights=args.weights, image_size=args.image_size, device=args.device,
                    amp=not args.no_amp, batch_size=args.batch_size)
            x_observed, quality_observed, hashes_observed, feature_meta = feature_sets[semantic]
            errors_observed, dynamic_meta = None, None
            if profile == 'dynamics':
                errors_observed, dynamic_meta = diagnose(
                    observation_pool, x_observed, seed=args.seed, dim=args.dynamics_dim,
                    lags=args.lags, folds=args.folds, alpha=args.ridge_alpha,
                    max_gap_frames=(args.max_transition_gap_frames if args.max_transition_gap_frames is not None
                                    else 1.5*import_meta['candidate_stride']),
                    max_fit_windows=args.max_fit_windows)
            x = x_observed[eligible_indices]
            quality = quality_observed[eligible_indices]
            hashes = [hashes_observed[i] for i in eligible_indices]
            errors = ({key: values[eligible_indices] for key, values in errors_observed.items()}
                      if errors_observed is not None else None)
            selected, reasons, duplicates, embedding, selection_meta = choose(pool, x, quality, hashes,
                profile=profile, budget=args.budget, seed=args.seed, allocation=args.allocation,
                anchor_fraction=args.anchor_fraction, importance_weight=args.importance_weight,
                min_gap_frames=args.min_gap_frames, spacing_policy=args.spacing_policy,
                dynamics=errors, signal=args.signal, projection_dim=args.projection_dim)
            destination.mkdir(parents=True)
            result = dict(run_id=run_id, variant=f'C-{profile}', seed=args.seed, budget=args.budget,
                frame_ids=[pool[i]['frame_id'] for i in selected], selection_meta=selection_meta)
            metric_start = time.perf_counter()
            metrics = measurements(pool, selected, embedding, quality, args.seed)
            metrics['measurement_wall_seconds'] = time.perf_counter()-metric_start
            metrics['dynamics'] = dynamic_meta
            chosen_set = set(selected)
            selected_ids = {pool[i]['frame_id'] for i in selected}
            eligible_by_id = {row['frame_id']: i for i, row in enumerate(pool)}
            def scored_rows():
                for observed_i, row in enumerate(observation_pool):
                    eligible_i = eligible_by_id.get(row['frame_id'])
                    yield dict(frame_id=row['frame_id'], group_id=row['group_id'], video_id=row['video_id'],
                        frame_index=row['frame_index'], eligible_for_selection=eligible_i is not None,
                        selected=row['frame_id'] in selected_ids,
                        reason=reasons.get(eligible_i) if eligible_i is not None else None,
                        duplicate_of=duplicates.get(row['frame_id']),
                        quality=dict(zip(feature_meta['quality_columns'],
                                         map(float, quality_observed[observed_i]))),
                        diagnostics={key: finite(values[observed_i])
                                     for key, values in (errors_observed or {}).items()})
            jsonl(destination/'frame_scores.jsonl', scored_rows())
            jsonl(destination/'selected_manifest.jsonl', [pool[i] for i in selected])
            # Persist the selection before export; unknown labels cannot erase its audit.
            write_json(destination/'variant.json', result)
            write_json(destination/'curation_metrics.json', metrics)
            run_manifest = dict(status='selected', created_at=datetime.now(timezone.utc).isoformat(),
                config=run_config, code_sha256=code, pool_identity=contract_hash,
                split=split, source=import_meta, features=feature_meta,
                split_frame_counts=dict(Counter(r['split'] for r in rows)),
                eligible_train_frames=len(pool), unknown_train_labels_excluded=sum(
                    r['split'] == 'train' and r['label_status'] == 'unknown' for r in rows)
                    if args.candidate_labels == 'provided' else 0,
                annotation_policy=args.candidate_labels, labels_used_for_scoring=False,
                observation_pool_frames=len(observation_pool),
                acquisition_pool_frames=len(pool),
                pool_roles={'U': 'all observable train frames',
                            'A': 'frames eligible under annotation policy'},
                training_performed=False, detector_evaluation_performed=False)
            write_json(destination/'run_manifest.json', run_manifest)
            if not args.no_export:
                export_start = time.perf_counter()
                run_manifest['export'] = export_coco([pool[i] for i in selected], destination, source,
                                                     args.accept_empty_tracking_labels)
                run_manifest['export_wall_seconds'] = time.perf_counter()-export_start
                result['annotations_train'] = str((destination/'annotations.coco.json').resolve())
                write_json(destination/'variant.json', result)
            run_manifest.update(status='completed', elapsed_session_seconds=time.perf_counter()-started)
            write_json(destination/'run_manifest.json', run_manifest)
            print(f'[selected] {profile}: {len(selected)} frames -> {destination}', flush=True)
        if args.export_evaluation:
            for name in ('val', 'test'):
                dest = args.out/'evaluation'/name
                if dest.exists():
                    raise FileExistsError(f'Evaluation output already exists: {dest}')
                fixed = [r for r in rows if r['split'] == name]
                report = export_coco(fixed, dest, source, args.accept_empty_tracking_labels)
                write_json(dest/'export_manifest.json', dict(split=name, pool_identity=contract_hash,
                    candidate_stride=import_meta['candidate_stride'], curated=False, **report))
    finally:
        source.close()


if __name__ == '__main__':
    main()
