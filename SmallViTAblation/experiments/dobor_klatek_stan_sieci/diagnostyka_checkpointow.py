#!/usr/bin/env python3
"""Fixed weak-label patch probes and representation drift after all pilot runs.

This post-training diagnostic never changes acquisition, detector training or
hyperparameters. Foreground means a patch centre inside the annotated bounding
box (with the selector's tiny-box fallback), not a segmentation label. Five
checkpoints are not used to fit another Langevin model along the training clock.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def select_probe(rows, l0_ids, *, val_count=192, seed=42):
    """Select val without consulting labels; round-robin shuffled operation groups."""
    if val_count <= 0 or len(l0_ids) != len(set(l0_ids)):
        raise ValueError('positive val_count and unique L0 frame IDs required')
    index = {str(row['frame_id']): row for row in rows}
    if len(index) != len(rows):
        raise ValueError('duplicate pool frame IDs')
    train_groups = {str(row['group_id']) for row in rows if row.get('split') == 'train'}
    val_groups = {str(row['group_id']) for row in rows if row.get('split') == 'val'}
    if train_groups & val_groups:
        raise ValueError('training and validation operation groups overlap')
    chosen = []
    for fid in l0_ids:
        row = index.get(str(fid))
        if row is None or row.get('split') != 'train':
            raise ValueError(f'L0 frame not in training pool: {fid}')
        chosen.append(row)
    grouped = defaultdict(list)
    for row in rows:
        if row.get('split') == 'val':
            grouped[str(row['group_id'])].append(row)
    if sum(map(len, grouped.values())) < val_count:
        raise ValueError('not enough validation frames for the fixed probe')
    rng = np.random.default_rng(seed)
    for group in sorted(grouped):
        group_rows = sorted(grouped[group], key=lambda r: str(r['frame_id']))
        grouped[group] = [group_rows[i] for i in rng.permutation(len(group_rows))]
    names = sorted(grouped)
    names = [names[i] for i in rng.permutation(len(names))]
    selected, offset = [], 0
    while len(selected) < val_count:
        for group in names:
            if offset < len(grouped[group]) and len(selected) < val_count:
                selected.append(grouped[group][offset])
        offset += 1
    return chosen + selected


def fit_patch_readout(tokens, targets, train_image_mask, *, seed=42, per_class=16, alpha=1.0):
    """Balanced ridge on +/-1 labels; every statistic is fitted only on L0 patches."""
    shape = np.shape(tokens)
    targets, mask = np.asarray(targets, dtype=bool), np.asarray(train_image_mask, dtype=bool)
    if len(shape) != 3 or targets.shape != shape[:2] or mask.shape != (shape[0],):
        raise ValueError('expected N,P,D tokens, N,P targets and N image mask')
    if per_class < 1 or alpha <= 0 or not mask.any():
        raise ValueError('positive sampling/regularization and training rows required')
    rng, samples = np.random.default_rng(seed), []
    for image_index in np.flatnonzero(mask):
        for label in (False, True):
            patches = np.flatnonzero(targets[image_index] == label)
            if len(patches) > per_class:
                patches = np.sort(rng.choice(patches, per_class, replace=False))
            samples.extend((int(image_index), int(patch)) for patch in patches)
    samples = np.asarray(samples, dtype=np.int64)
    x = np.asarray(tokens[samples[:, 0], samples[:, 1]], dtype=np.float64)
    y = targets[samples[:, 0], samples[:, 1]].astype(float)
    if not np.isfinite(x).all() or len(np.unique(y)) != 2:
        raise ValueError('finite L0 patch features and both weak classes required')
    mean = x.mean(0)
    scale = np.maximum(x.std(0), 1e-8)
    design = np.column_stack([(x - mean) / scale, np.ones(len(x))])
    weight = np.where(y > 0, .5 / np.sum(y > 0), .5 / np.sum(y == 0))
    gram = design.T @ (weight[:, None] * design)
    gram += np.diag([alpha] * shape[-1] + [0.])
    coefficients = np.linalg.solve(gram, design.T @ (weight * (2 * y - 1)))
    return dict(mean=mean, scale=scale, weights=coefficients,
                fit_image_indices=samples[:, 0], fit_patch_indices=samples[:, 1],
                foreground_count=int(y.sum()), background_count=int((1 - y).sum()),
                alpha=float(alpha), sampling_seed=seed)


def readout_logits(model, tokens):
    x = np.asarray(tokens, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('readout inputs must be finite')
    return (x - model['mean']) / model['scale'] @ model['weights'][:-1] + model['weights'][-1]


def binary_metrics(targets, logits):
    """AUROC with tied ranks, AP with tied score groups, and equal-class BCE."""
    y = np.asarray(targets).reshape(-1)
    score = np.asarray(logits, dtype=np.float64).reshape(-1)
    if y.shape != score.shape or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError('finite aligned binary targets and logits required')
    y = y.astype(bool)
    positive, negative = int(y.sum()), int((~y).sum())
    result = dict(tokens=len(y), foreground_tokens=positive, background_tokens=negative,
                  foreground_fraction=positive / len(y), auroc=None, average_precision=None, balanced_bce=None)
    if not positive or not negative:
        result['reason'] = 'both foreground and background are required'
        return result
    order = np.argsort(score, kind='stable')
    ordered = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered) != 0) + 1]
    ends = np.r_[starts[1:], len(y)]
    ranks = np.empty(len(y), dtype=float)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2
    result['auroc'] = float((ranks[y].sum() - positive * (positive + 1) / 2) / (positive * negative))
    descending = np.argsort(-score, kind='stable')
    ordered_y = y[descending]
    group_ends = np.r_[np.flatnonzero(np.diff(score[descending]) != 0), len(y) - 1]
    true_positives = np.cumsum(ordered_y)[group_ends]
    precision = true_positives / (group_ends + 1)
    recall = true_positives / positive
    result['average_precision'] = float(np.sum(np.diff(np.r_[0., recall]) * precision))
    loss = np.logaddexp(0., score) - y * score
    result['balanced_bce'] = float(.5 * (loss[y].mean() + loss[~y].mean()))
    return result


def linear_cka(first, second):
    x, y = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError('CKA requires aligned two-dimensional patch arrays')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('CKA inputs must be finite')
    x, y = x - x.mean(0), y - y.mean(0)
    denominator = np.linalg.norm(x.T @ x, 'fro') * np.linalg.norm(y.T @ y, 'fro')
    return None if denominator <= 1e-20 else float(np.clip(np.linalg.norm(x.T @ y, 'fro') ** 2 / denominator, 0, 1))


def discover_checkpoints(pilot_root):
    pilot = Path(pilot_root)
    plan = json.loads((pilot / 'build_manifest.json').read_text(encoding='utf-8'))
    expected_arms = {'random', 'random_large', 'readout_loss', 'koopman_langevin', 'placebo'}
    if set(plan['runs']) != expected_arms or plan['epochs'] != 10:
        raise ValueError('diagnostic expects the completed five-arm, ten-epoch pilot')
    found = []
    for arm, run_id in sorted(plan['runs'].items()):
        run = pilot / 'results' / run_id
        report = json.loads((run / 'validation_report.json').read_text(encoding='utf-8'))
        if report.get('status') != 'passed' or report.get('epochs_completed') != 10:
            raise ValueError(f'pilot run must finish before diagnostics: {run_id}')
        manifest = run / 'epoch_checkpoints/MANIFEST.json'
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        entries = payload['checkpoints']
        if payload['every_n_epochs'] != 2 or len(entries) != 5 or {item['epoch'] for item in entries} != {1, 3, 5, 7, 9}:
            raise ValueError(f'expected checkpoints after epochs 2,4,6,8,10: {manifest}')
        for entry in sorted(entries, key=lambda item: item['epoch']):
            checkpoint = (manifest.parent / entry['file']).resolve()
            if not checkpoint.is_relative_to(manifest.parent.resolve()) or sha256(checkpoint) != entry['sha256']:
                raise ValueError(f'invalid or changed checkpoint: {checkpoint}')
            found.append(dict(arm=arm, run_id=run_id, epoch_index=entry['epoch'],
                              completed_epochs=entry['epoch'] + 1, checkpoint=str(checkpoint),
                              checkpoint_sha256=entry['sha256'], checkpoint_manifest_sha256=sha256(manifest)))
    return plan, found


def run(project_dir, data_root, selection_root, pilot_root, out, *, batch_size=32):
    project, data, selection, pilot, output = map(Path, (project_dir, data_root, selection_root, pilot_root, out))
    if output.exists():
        raise FileExistsError(f'use a new immutable diagnostic directory: {output}')
    project = project.resolve()
    if str(project.parent) not in sys.path:
        sys.path.insert(0, str(project.parent))
    package = 'SmallViTAblation.experiments.dobor_klatek_stan_sieci'
    extraction = importlib.import_module(package + '.token_ekstrakcja')
    target_module = importlib.import_module(package + '.token_dopasowanie')
    if not Path(extraction.__file__).resolve().is_relative_to(project):
        raise ValueError('imported extractor does not belong to --project-dir')
    plan, checkpoints = discover_checkpoints(pilot)
    random_variants = []
    for file in (selection / 'variants').glob('*.json'):
        value = json.loads(file.read_text(encoding='utf-8'))
        if value.get('strategy') == 'random':
            random_variants.append((file, value))
    if len(random_variants) != 1:
        raise ValueError('selection must contain exactly one random variant')
    variant_path, variant = random_variants[0]
    l0_ids = variant.get('frame_ids', variant.get('selected_frame_ids'))[:500]
    if len(l0_ids) != 500 or hashlib.sha256('\n'.join(l0_ids).encode()).hexdigest() != plan['l0_sha256']:
        raise ValueError('initial 500 frames do not match the frozen pilot')
    pool_path = data / 'pool_manifest.jsonl'
    probe = select_probe(read_jsonl(pool_path), l0_ids, val_count=192, seed=42)
    for row in probe:
        image = (data / row['path']).resolve()
        if not image.is_relative_to(data.resolve()) or not image.is_file():
            raise FileNotFoundError(f'probe image missing or outside data-root: {image}')
    output.mkdir(parents=True)
    metadata_path, manifest_path = output / 'probe_metadata.jsonl', output / 'probe_manifest.jsonl'
    with metadata_path.open('x', encoding='utf-8') as metadata, manifest_path.open('x', encoding='utf-8') as manifest:
        for row in probe:
            metadata.write(json.dumps(row) + '\n')
            safe = {k: row[k] for k in ('frame_id', 'video_id', 'group_id', 'frame_index', 'split', 'path')}
            manifest.write(json.dumps(safe) + '\n')
    label_rows = {str(row['frame_id']): row for row in probe}
    val_ids = sorted(str(row['frame_id']) for row in probe if row['split'] == 'val')
    probe_hash = sha256(manifest_path)
    source_hash = {name: sha256(path) for name, path in (
        ('diagnostic', Path(__file__)), ('extractor', Path(extraction.__file__)),
        ('target_definition', Path(target_module.__file__)))}
    source_hash['pilot_plan'] = sha256(pilot / 'build_manifest.json')
    source_hash['selection_variant'] = sha256(variant_path)
    source_hash['pool_manifest'] = sha256(pool_path)
    results, previous, cka_pairs, reference_projection = [], {}, None, None
    start = time.monotonic()
    for checkpoint in checkpoints:
        cache = output / 'patches' / checkpoint['arm'] / f'epoch_{checkpoint["completed_epochs"]:03d}'
        meta = extraction.extract(data, manifest_path, checkpoint['checkpoint'], cache,
                                  batch_size=batch_size, resolution=384, dim=24, seed=42, splits=('train', 'val'))
        if meta['checkpoint_sha256'] != checkpoint['checkpoint_sha256'] or meta['manifest_sha256'] != probe_hash:
            raise ValueError('extraction provenance differs from the fixed diagnostic')
        grid = tuple(meta['grid_shape'])
        projection = (grid, meta['raw_channels'], meta['projection_sha256'])
        if reference_projection is None:
            reference_projection = projection
        elif projection != reference_projection:
            raise ValueError('patch grid or channel projection changed across checkpoints')
        rows = read_jsonl(cache / 'rows.jsonl')
        ids = [str(row['frame_id']) for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(label_rows):
            raise ValueError('checkpoint extraction changed probe frame identity')
        position = {fid: i for i, fid in enumerate(ids)}
        features = np.load(cache / 'tokens.npy', mmap_mode='r', allow_pickle=False)
        if features.shape != (len(probe), int(np.prod(grid)), 24):
            raise ValueError('unexpected patch feature shape')
        labels = np.stack([target_module.token_targets(label_rows[fid], grid) for fid in ids])
        fit_mask = np.array([fid in set(l0_ids) for fid in ids])
        model = fit_patch_readout(features, labels, fit_mask, seed=42, per_class=16, alpha=1.)
        validation_positions = [position[fid] for fid in val_ids]
        metrics = binary_metrics(labels[validation_positions], readout_logits(model, features[validation_positions]))
        readout_path = cache / 'readout_l0.npz'
        with readout_path.open('xb') as stream:
            np.savez_compressed(stream, **model, frame_ids=np.array(ids))
        if cka_pairs is None:
            patches = features.shape[1]
            flat = np.random.default_rng(42).choice(len(val_ids) * patches, min(4096, len(val_ids) * patches), replace=False)
            cka_pairs = [(val_ids[int(i // patches)], int(i % patches)) for i in sorted(flat)]
            write_json(output / 'CKA_SAMPLE.json', dict(seed=42, grid_shape=grid,
                                                      identities=[dict(frame_id=fid, patch_index=p) for fid, p in cka_pairs]))
        cka_features = np.stack([features[position[fid], patch] for fid, patch in cka_pairs]).astype(np.float64)
        last = previous.get(checkpoint['arm'])
        drift = None if last is None else linear_cka(last['features'], cka_features)
        result = dict(**checkpoint, metrics=metrics, readout=dict(
            fit_source='common_L0_only', train_frames=int(fit_mask.sum()), val_frames=len(val_ids),
            foreground_training_tokens=model['foreground_count'], background_training_tokens=model['background_count'],
            alpha=model['alpha'], sampling_seed=42, max_patches_per_class_per_image=16,
            score='balanced ridge regression on +/-1 labels, sigmoid for BCE; not calibrated probabilities',
            path=str(readout_path), sha256=sha256(readout_path)),
            cka_to_previous_checkpoint=drift, cka_previous_epoch=None if last is None else last['completed_epochs'],
            cka_patch_count=len(cka_pairs), cka_role='representation drift only; no semantic or prediction claim',
            feature_cache=str(cache), feature_meta_sha256=sha256(cache / 'meta.json'), probe_manifest_sha256=probe_hash)
        write_json(cache / 'diagnostic.json', result)
        results.append(result)
        previous[checkpoint['arm']] = dict(features=cka_features, completed_epochs=checkpoint['completed_epochs'])
        print(json.dumps(dict(arm=checkpoint['arm'], epoch=checkpoint['completed_epochs'], **metrics,
                              cka=drift)), flush=True)
    report = dict(schema_version=1, status='complete', rows=results,
                  probe=dict(train_frames=500, val_frames=192, selection_seed=42, val_selection='group_stratified_round_robin',
                             manifest=str(manifest_path), manifest_sha256=probe_hash,
                             label_metadata=str(metadata_path), label_metadata_sha256=sha256(metadata_path),
                             cka_sample_sha256=sha256(output / 'CKA_SAMPLE.json')),
                  particle='one patch token at the same image-grid coordinate',
                  target='weak bounding-box foreground/background, not semantic segmentation or instrument identity',
                  extraction_uses_labels=False, readout_fit_uses_val=False, used_for_ranking=False,
                  cka_only_measures_drift=True, langevin_fitted_on_training_clock=False,
                  training_clock_note='Only five checkpoints; no dynamical model is fitted to this axis.',
                  statistical_significance_tested=False, source_sha256=source_hash,
                  elapsed_seconds=time.monotonic() - start)
    write_json(output / 'DIAGNOSTYKA.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-dir', 'data-root', 'selection-root', 'pilot-root', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args(argv)
    result = run(args.project_dir, args.data_root, args.selection_root, args.pilot_root, args.out,
                 batch_size=args.batch_size)
    print(json.dumps(dict(status=result['status'], checkpoints=len(result['rows']), out=str(args.out))))


if __name__ == '__main__':
    main()
