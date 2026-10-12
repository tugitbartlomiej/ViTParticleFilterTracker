"""Condition-wise D diagnostics and honest campaign summaries; no invented results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def condition_report(fit, diagnostic, losses, groups, names):
    rows = []
    groups, losses = np.asarray(groups), np.asarray(losses)
    for j, name in enumerate(names):
        values = fit[np.isfinite(fit[:, j]), j]
        if len(values) < 2 or np.ptp(values) == 0:
            continue
        low, high = map(float, np.quantile(values, [1/3, 2/3]))
        d = diagnostic[:, j]
        masks = [d <= low, (d > low) & (d <= high), d > high]
        for label, mask in zip(('low', 'middle', 'high'), masks):
            mask &= np.isfinite(d) & np.isfinite(losses)
            means = [float(np.mean(losses[mask & (groups == g)])) for g in sorted(set(groups[mask]))]
            rows.append(dict(feature=name, region=label, fit_tertiles=[low, high],
                             support=int(mask.sum()), group_count=len(means),
                             mean_task_loss=float(np.mean(means)) if means else None))
    return dict(dataset='D used to steer selection; not independent generalization evidence', conditions=rows)


def summarize(root, metric='val/mAP_50_95', target_ap=None):
    if target_ap is not None and not 0 <= target_ap <= 1:
        raise ValueError('target AP must lie in [0,1] and should be specified before the campaign')
    completed = sorted(Path(root).glob('*/completed.json'))
    runs = []
    provenance = []
    for path in completed:
        result = json.loads(path.read_text(encoding='utf-8'))
        config = json.loads(path.with_name('run_config.json').read_text(encoding='utf-8'))
        provenance.append((config['assignment_sha256'], config['initial_weights_sha256'],
                           config['features_sha256'], config['protocol_config_sha256'],
                           json.dumps({k: v['sha256'] for k, v in (config.get('input_archives') or {}).items()}, sort_keys=True),
                           json.dumps(config['code_sha256'], sort_keys=True)))
        timing_file = path.with_name('job_timing.json')
        timing = json.loads(timing_file.read_text()) if timing_file.is_file() else None
        if any(metric not in r['metrics'] for r in result['rounds']):
            raise ValueError(f'missing metric {metric}: {path}')
        reached = [r for r in result['rounds'] if target_ap is not None and r['metrics'][metric] >= target_ap]
        runs.append(dict(arm=result['arm'], seed=result['seed'], epochs=result['epochs'],
                         final_ap=result['rounds'][-1]['metrics'][metric],
                         target_ap=target_ap,
                         training_seconds_to_target=reached[0]['metric_elapsed_seconds'] if reached else None,
                         training_and_diagnosis_seconds=result['elapsed_seconds'], job_timing=timing,
                         optimizer_steps=result['optimizer_steps'],
                         exposures=result['trained_exposures'], unique_frames=result['trained_unique'],
                         rounds=result['rounds']))
    if len(set(provenance)) > 1:
        raise ValueError('runs differ in operation/frame split or initialization')
    identities = [(r['arm'], r['seed']) for r in runs]
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate arm/seed results')
    paired = []
    full = {r['seed']: r for r in runs if r['arm'] == 'features_kg_glister'}
    for baseline in ('random', 'glister', 'features_glister', 'features_placebo_glister'):
        other = {r['seed']: r for r in runs if r['arm'] == baseline}
        deltas = [dict(seed=s, delta_ap=full[s]['final_ap']-other[s]['final_ap']) for s in sorted(full.keys() & other.keys())]
        if deltas:
            paired.append(dict(comparison='features_kg_glister minus '+baseline, pairs=deltas,
                               mean_delta_ap=float(np.mean([d['delta_ap'] for d in deltas]))))
    return dict(completed_runs=len(runs), metric=metric, runs=runs, paired_comparisons=paired,
                interpretation='Compare independent val AP versus full elapsed time, updates and exposures; D is adaptive, test untouched.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--metric', default='val/mAP_50_95')
    p.add_argument('--target-ap', type=float)
    a = p.parse_args()
    print(json.dumps(summarize(a.results, a.metric, a.target_ap), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
