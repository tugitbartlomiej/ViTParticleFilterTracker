#!/usr/bin/env python3
"""Real RF-DETR Nano training plus independent reload of best AND last weights."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--variant-file', type=Path, required=True)
    p.add_argument('--val-annotations', type=Path,
                   help='Optional fixed small validation set for technical smoke')
    p.add_argument('--eval-only', action='store_true',
                   help='Check already trained weights after an evaluation adapter fix; never retrain')
    a = p.parse_args()
    project = Path(__file__).resolve().parents[2]
    run_dir = a.out / a.run_id
    if run_dir.exists() and not a.eval_only:
        raise FileExistsError(f'Use a new run-id; refusing to mix smoke attempts: {run_dir}')
    start = time.monotonic()
    command = [sys.executable, str(project / 'pipeline/train/train_rfdetr.py'),
               '--run-id', a.run_id, '--data-root', str(a.data_root), '--out', str(a.out),
               '--protocol', str(a.protocol), '--variant-file', str(a.variant_file),
               '--train-seed', '42', '--role', 'smoke', '--profile', 'nano_smoke']
    val_annotations = a.val_annotations or (a.data_root / 'annotations_val.json')
    if not val_annotations.is_absolute():
        val_annotations = a.data_root / val_annotations
    command += ['--val-annotations', str(val_annotations)]
    if not a.eval_only:
        subprocess.run(command, check=True)
    training_seconds = time.monotonic()-start
    manifest = json.loads((run_dir / 'run_manifest.json').read_text())
    if a.eval_only:
        training_seconds = (datetime.fromisoformat(manifest['finished_at'])
                            - datetime.fromisoformat(manifest['started_at'])).total_seconds()
    if manifest['epochs_completed'] != 5 or manifest['model'] != 'rf-detr-nano':
        raise RuntimeError('Wrong smoke recipe or incomplete training')
    checks = {}
    for filename, rule in [('last.pth', 'last_epoch'), ('best.pth', 'best_val')]:
        weights = run_dir / filename
        eval_out = run_dir / ('eval_' + rule)
        subprocess.run([sys.executable, str(project / 'pipeline/eval/evaluate.py'),
            '--protocol', str(a.protocol), '--run-id', a.run_id,
            '--checkpoint-rule', rule, '--set', 'val', '--checkpoint', str(weights),
            '--annotations', str(val_annotations),
            '--images-dir', str(a.data_root), '--device', 'cuda',
            '--model-name', 'rf-detr-nano', '--resolution', '384',
            '--output-dir', str(eval_out), '--operating-threshold', '0.5'], check=True)
        with weights.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        metrics = json.loads((eval_out / 'metrics.json').read_text())
        expected = manifest['dataset']['n_val_images']
        if metrics['results']['val']['n_images'] != expected:
            raise RuntimeError('Reload evaluation omitted validation images')
        checks[rule] = {'checkpoint_sha256': sha, 'metrics': metrics['results']['val']}
    report = {'status': 'passed', 'scope': 'technical_smoke_only',
              'evaluation_only_job': a.eval_only,
              'training_source_commit': manifest.get('git_commit'),
              'evaluation_source_commit': os.environ.get('SMALLVIT_SOURCE_COMMIT'),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'training_seconds': training_seconds, 'total_seconds': time.monotonic()-start,
              'operating_threshold_note': '0.5 fixed for smoke; select on val in real experiments',
              'checkpoint_checks': checks}
    (run_dir / 'smoke_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
