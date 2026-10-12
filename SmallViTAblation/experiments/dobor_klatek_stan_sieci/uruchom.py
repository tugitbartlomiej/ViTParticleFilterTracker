#!/usr/bin/env python3
"""Build an immutable five-arm pilot queue, or summarize its completed val runs."""
from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import math
from pathlib import Path

import yaml


ARMS = {'random', 'readout_loss', 'koopman_langevin', 'placebo', 'random_large'}
ARM_BUDGETS = {arm: (3000 if arm == 'random_large' else 1500) for arm in sorted(ARMS)}
PROFILE = 'nano_static_pilot'


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(project_dir: Path, variants_dir: Path, output_root: Path,
          base_protocol: Path, *, seed: int = 42, epochs: int = 10) -> dict:
    if epochs <= 0:
        raise ValueError('epochs must be positive')
    project, variants, output = (Path(p).resolve() for p in (project_dir, variants_dir, output_root))
    base = Path(base_protocol).resolve()
    protocol = copy.deepcopy(yaml.safe_load(base.read_text(encoding='utf-8')))
    profile_source = base.parent / 'profiles.yaml'
    profiles = copy.deepcopy(yaml.safe_load(profile_source.read_text(encoding='utf-8')))
    if PROFILE not in profiles.get('profiles', {}):
        raise ValueError(f'missing base profile {PROFILE}')
    protocol['training']['epochs'] = epochs
    profiles['profiles'][PROFILE]['epochs'] = epochs
    copied_profile = profiles['profiles'][PROFILE]
    copied_profile['budgets'] = sorted(set(copied_profile.get('budgets', [])) | {1500, 3000})
    copied_profile['strategies'] = sorted(set(copied_profile.get('strategies', [])) | ARMS)
    if protocol.get('model', {}).get('init') != 'coco_pretrained':
        raise ValueError('pilot requires COCO initialization')

    selections, shared_l0, selection_seed = {}, None, None
    for path in sorted(variants.glob('*.json')):
        if 'manifest' in path.stem.lower():
            continue
        value = json.loads(path.read_text(encoding='utf-8'))
        strategy = value.get('strategy')
        if strategy not in ARMS or strategy in selections:
            raise ValueError(f'unknown or repeated pilot strategy in {path}: {strategy}')
        frames = value.get('frame_ids', value.get('selected_frame_ids'))
        if not isinstance(frames, list) or any(not isinstance(f, str) for f in frames):
            raise ValueError(f'{path}: frame_ids must be strings')
        expected_budget = ARM_BUDGETS[strategy]
        if (value.get('budget') != expected_budget or len(frames) != expected_budget
                or len(set(frames)) != expected_budget):
            raise ValueError(f'{path}: expected {expected_budget} unique frame_ids')
        if 'selected_frame_ids' in value and value['selected_frame_ids'] != frames:
            raise ValueError(f'{path}: contradictory frame id lists')
        if shared_l0 is None:
            shared_l0 = frames[:500]
            selection_seed = value.get('selection_seed', value.get('seed'))
        if frames[:500] != shared_l0:
            raise ValueError(f'{path}: the first 500 frames must be the same ordered L0')
        if value.get('selection_seed', value.get('seed')) != selection_seed or value.get('seed') is None:
            raise ValueError(f'{path}: missing or inconsistent selection seed')
        selections[strategy] = (path, value)
    if set(selections) != ARMS:
        raise ValueError(f'expected exactly five arms {sorted(ARMS)}, found {sorted(selections)}')

    config = output / 'config'
    destinations = [config / 'protocol.yaml', config / 'profiles.yaml',
                    output / 'queue.jsonl', output / 'build_manifest.json', output / 'queue']
    for path in destinations:
        if path.exists():
            raise FileExistsError(f'no implicit overwrite/resume: {path}')
    tasks = []
    for strategy, (path, value) in sorted(selections.items()):
        run_id = f'{path.stem}_s{seed}'
        if (output / 'results' / run_id).exists():
            raise FileExistsError(f'no implicit retraining: {output / "results" / run_id}')
        tasks.append(dict(run_id=run_id, arch='rf-detr-nano', stage='train', seed=seed,
                          train_seed=seed, selection_seed=selection_seed, budget=value['budget'],
                          strategy=strategy, profile=PROFILE, variant_file=str(path),
                          variant_sha256=sha256(path),
                          argv=['{python}', str(project / 'experiments/dobor_klatek_stan_sieci/trening.py'),
                                '--run-id', run_id, '--data-root', '{data}', '--out', '{out}',
                                '--protocol', '{protocol}', '--variant-file', str(path),
                                '--train-seed', str(seed), '--profile', PROFILE,
                                '--expected-epochs', str(epochs),
                                '--expected-variant-sha256', sha256(path)]))
    config.mkdir(parents=True, exist_ok=True)
    for path, value in ((config / 'protocol.yaml', protocol), (config / 'profiles.yaml', profiles)):
        with path.open('x', encoding='utf-8') as stream:
            yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)
    for task in tasks:
        task['argv'] += ['--expected-protocol-sha256', sha256(config / 'protocol.yaml'),
                         '--expected-profiles-sha256', sha256(config / 'profiles.yaml')]
    with (output / 'queue.jsonl').open('x', encoding='utf-8') as stream:
        for task in tasks:
            stream.write(json.dumps(task) + '\n')
    summary = dict(schema_version=1, tasks=len(tasks), training_seed=seed, selection_seed=selection_seed,
                   epochs=epochs, budgets=ARM_BUDGETS, shared_l0=500,
                   added={arm: budget - 500 for arm, budget in ARM_BUDGETS.items()},
                   checkpoint_interval_epochs=2,
                   l0_sha256=hashlib.sha256('\n'.join(shared_l0).encode()).hexdigest(),
                   initialization='coco_pretrained', primary_checkpoint_rule='last_epoch',
                   primary_metric='mAP_0.5:0.95', evaluation_set='val', test_evaluated=False,
                   source_protocol_sha256=sha256(base), source_profiles_sha256=sha256(profile_source),
                   protocol_sha256=sha256(config / 'protocol.yaml'),
                   profiles_sha256=sha256(config / 'profiles.yaml'), queue_sha256=sha256(output / 'queue.jsonl'),
                   code_sha256=sha256(Path(__file__)), runs={t['strategy']: t['run_id'] for t in tasks})
    with (output / 'build_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    return summary


def exposure_comparison(curves: dict) -> dict:
    """Integrate linearly interpolated epoch AP over common image-exposure proxy."""
    lo = max(curve[0]['image_presentations_proxy'] for curve in curves.values())
    hi = min(curve[-1]['image_presentations_proxy'] for curve in curves.values())
    if hi <= lo:
        return dict(status='insufficient_common_range', common_range=[lo, hi])
    averages = {}
    for arm, curve in curves.items():
        xs = [point['image_presentations_proxy'] for point in curve]
        ys = [point['ap'] for point in curve]
        def interpolate(x):
            i = min(max(bisect.bisect_right(xs, x) - 1, 0), len(xs) - 2)
            fraction = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + fraction * (ys[i + 1] - ys[i])
        grid = sorted({lo, hi, *(x for x in xs if lo < x < hi)})
        area = sum((b - a) * (interpolate(a) + interpolate(b)) / 2
                   for a, b in zip(grid, grid[1:]))
        averages[arm] = area / (hi - lo)
    return dict(status='computed', common_range=[lo, hi], normalized_area=averages,
                x_definition='completed_epochs * unique_training_frames; exposure proxy, not measured optimizer steps',
                y_definition='AP from per-epoch training validation; independent last-epoch AP is reported separately',
                interpolation='piecewise_linear_within_observed_common_range_only',
                note='Large random arm has twice the nominal image exposure per epoch; epoch count is not equal compute.')


def report(output_root: Path) -> dict:
    root = Path(output_root)
    plan = json.loads((root / 'build_manifest.json').read_text(encoding='utf-8'))
    if sha256(root / 'queue.jsonl') != plan['queue_sha256']:
        raise ValueError('queue differs from the immutable pilot plan')
    if set(plan['runs']) != ARMS:
        raise ValueError('pilot plan must have all five arms')
    rows, validation_sizes, curves = [], set(), {}
    for strategy, run_id in sorted(plan['runs'].items()):
        path = root / 'results' / run_id / 'validation_report.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        if (value.get('status') != 'passed' or value.get('evaluation_set') != 'val'
                or value.get('test_evaluated') is not False
                or value.get('training_seed') != plan['training_seed']
                or value.get('selection_seed') != plan['selection_seed']
                or value.get('epochs_completed') != plan['epochs']):
            raise ValueError(f'incomplete or incompatible validation report: {path}')
        metrics = value['checkpoints']['last_epoch']['metrics']
        ap = float(metrics['mAP_0.5:0.95'])
        if not math.isfinite(ap) or not 0 <= ap <= 1 or metrics['n_images'] <= 0:
            raise ValueError(f'invalid validation metrics: {path}')
        validation_sizes.add(metrics['n_images'])
        budget = plan['budgets'][strategy]
        history_path = path.parent / 'epoch_val_map.json'
        history = json.loads(history_path.read_text(encoding='utf-8'))['per_epoch_map_50_95_val']
        curve = []
        for epoch, epoch_ap in sorted(history.items(), key=lambda pair: int(pair[0])):
            epoch, epoch_ap = int(epoch), float(epoch_ap)
            if not 0 <= epoch < plan['epochs'] or not math.isfinite(epoch_ap) or not 0 <= epoch_ap <= 1:
                raise ValueError(f'invalid training curve: {history_path}')
            curve.append(dict(epoch_index=epoch, completed_epochs=epoch + 1, ap=epoch_ap,
                              image_presentations_proxy=(epoch + 1) * budget))
        if len(curve) != plan['epochs']:
            raise ValueError(f'incomplete training curve: {history_path}')
        curves[strategy] = curve
        rows.append(dict(strategy=strategy, run_id=run_id, ap=ap,
                         ap50=metrics.get('mAP_0.5'), n_images=metrics['n_images'], budget=budget,
                         best_epoch_ap_training_log=value.get('best_map_50_95_val_training_log')))
    if len(validation_sizes) != 1:
        raise ValueError('runs evaluated different numbers of validation frames')
    ap = {row['strategy']: row['ap'] for row in rows}
    deltas = {f'koopman_langevin_minus_{arm}': ap['koopman_langevin'] - ap[arm]
              for arm in ('readout_loss', 'placebo', 'random', 'random_large')}
    return dict(schema_version=1, status='complete', rows=rows, deltas=deltas,
                checkpoint_rule='last_epoch', metric='mAP_0.5:0.95', evaluation_set='val',
                epochs=plan['epochs'], training_seed=plan['training_seed'], selection_seed=plan['selection_seed'],
                training_curves=curves, exposure_comparison=exposure_comparison(curves),
                practical_threshold_vs_readout_loss=.004,
                practical_gate_met=(deltas['koopman_langevin_minus_readout_loss'] >= .004
                                    and deltas['koopman_langevin_minus_placebo'] > 0),
                statistical_significance_tested=False,
                interpretation='Exploratory single-seed pilot; no significance or confirmation claim.',
                test_evaluated=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path)
    parser.add_argument('--variants-dir', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--base-protocol', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args(argv)
    if args.report:
        result = report(args.output_root)
    else:
        if not all((args.project_dir, args.variants_dir, args.base_protocol)):
            parser.error('build requires --project-dir, --variants-dir and --base-protocol')
        result = build(args.project_dir, args.variants_dir, args.output_root,
                       args.base_protocol, seed=args.seed, epochs=args.epochs)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
