#!/usr/bin/env python3
"""Train Nano and independently evaluate both retained checkpoints on fixed val."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['run-id', 'data-root', 'out', 'protocol', 'variant-file']:
        ap.add_argument('--'+name, required=True)
    ap.add_argument('--train-seed', type=int, required=True)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--expected-epochs', type=int, required=True)
    args = ap.parse_args()
    project = Path(__file__).resolve().parents[2]
    run = Path(args.out)/args.run_id
    if run.exists():
        raise FileExistsError(f'Refusing implicit resume or mixed attempts: {run}')
    start = time.monotonic()
    subprocess.run([sys.executable, str(project/'pipeline/train/train_rfdetr.py'),
        '--run-id', args.run_id, '--data-root', args.data_root, '--out', args.out,
        '--protocol', args.protocol, '--variant-file', args.variant_file,
        '--train-seed', str(args.train_seed), '--role', 'smoke', '--profile', args.profile], check=True)
    manifest = json.loads((run/'run_manifest.json').read_text())
    if manifest['epochs_completed'] != args.expected_epochs or manifest['model'] != 'rf-detr-nano':
        raise RuntimeError('Incomplete training or incorrect model')
    validation = {}
    for filename, rule in [('last.pth', 'last_epoch'), ('best.pth', 'best_val')]:
        weights = run/filename
        output = run/('eval_'+rule)
        subprocess.run([sys.executable, str(project/'pipeline/eval/evaluate.py'),
            '--protocol', args.protocol, '--run-id', args.run_id,
            '--checkpoint-rule', rule, '--set', 'val', '--checkpoint', str(weights),
            '--annotations', str(Path(args.data_root)/'annotations_val.json'),
            '--images-dir', args.data_root, '--device', 'cuda',
            '--model-name', 'rf-detr-nano', '--resolution', str(manifest['resolution']),
            '--output-dir', str(output)], check=True)
        metrics = json.loads((output/'metrics.json').read_text())['results']['val']
        if metrics['n_images'] != manifest['dataset']['n_val_images']:
            raise RuntimeError('Independent evaluation omitted validation images')
        with weights.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        validation[rule] = dict(checkpoint_sha256=sha, metrics=metrics)
    (run/'validation_report.json').write_text(json.dumps(dict(status='passed',
        scope='training_recipe_calibration', training_seed=args.train_seed,
        total_seconds=time.monotonic()-start, evaluation_set='val',
        test_evaluated=False, checkpoints=validation), indent=2))


if __name__ == '__main__':
    main()
