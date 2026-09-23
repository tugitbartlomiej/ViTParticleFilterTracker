#!/usr/bin/env python3
"""Train Nano and independently evaluate both retained checkpoints on fixed val."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def archive_predictions(path: Path) -> dict:
    """Keep auditable predictions without exhausting the shared home quota."""
    archive = path.with_suffix(path.suffix + '.gz')
    with path.open('rb') as source, archive.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    with path.open('rb') as stream:
        source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    with archive.open('rb') as stream:
        archive_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    source_bytes = path.stat().st_size
    path.unlink()
    return dict(format='gzip', path=str(archive), sha256=archive_sha,
                uncompressed_sha256=source_sha, uncompressed_bytes=source_bytes,
                bytes=archive.stat().st_size)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['run-id', 'data-root', 'out', 'protocol', 'variant-file']:
        ap.add_argument('--'+name, required=True)
    ap.add_argument('--train-seed', type=int, required=True)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--expected-epochs', type=int, required=True)
    ap.add_argument('--compress-predictions', action='store_true')
    ap.add_argument('--keep-epoch-checkpoints', type=int, default=0)
    ap.add_argument('--val-loss', action='store_true')
    ap.add_argument('--early-stop', action='store_true')
    args = ap.parse_args()
    project = Path(__file__).resolve().parents[2]
    run = Path(args.out)/args.run_id
    if run.exists():
        raise FileExistsError(f'Refusing implicit resume or mixed attempts: {run}')
    start = time.monotonic()
    variant = json.loads(Path(args.variant_file).read_text())
    subprocess.run([sys.executable, str(project/'pipeline/train/train_rfdetr.py'),
        '--run-id', args.run_id, '--data-root', args.data_root, '--out', args.out,
        '--protocol', args.protocol, '--variant-file', args.variant_file,
        '--train-seed', str(args.train_seed), '--role', 'smoke', '--profile', args.profile,
        '--keep-epoch-checkpoints', str(args.keep_epoch_checkpoints)]
        + (['--val-loss'] if args.val_loss else [])
        + (['--early-stop'] if args.early_stop else []), check=True)
    manifest = json.loads((run/'run_manifest.json').read_text())
    if (args.expected_epochs > 0 and manifest['epochs_completed'] != args.expected_epochs)             or manifest['model'] != 'rf-detr-nano':
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
        prediction_archive = None
        if args.compress_predictions:
            prediction_archive = archive_predictions(output/'predictions_val.json')
        validation[rule] = dict(checkpoint_sha256=sha, metrics=metrics,
                                predictions_archive=prediction_archive)
    (run/'validation_report.json').write_text(json.dumps(dict(status='passed',
        scope=variant.get('scope', 'unspecified'), training_seed=args.train_seed,
        selection_seed=variant.get('selection_seed', variant.get('seed')),
        total_seconds=time.monotonic()-start, evaluation_set='val',
        test_evaluated=False, checkpoints=validation), indent=2))


if __name__ == '__main__':
    main()
