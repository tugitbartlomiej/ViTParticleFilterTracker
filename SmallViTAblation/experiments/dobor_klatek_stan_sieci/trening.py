#!/usr/bin/env python3
"""Train a fresh COCO Nano, retain both checkpoints, independently evaluate only last."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT.parent))
from SmallViTAblation.pipeline.train.run_with_validation import archive_predictions


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-id', 'data-root', 'out', 'protocol', 'variant-file', 'profile',
                 'expected-variant-sha256', 'expected-protocol-sha256', 'expected-profiles-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--train-seed', type=int, required=True)
    parser.add_argument('--expected-epochs', type=int, required=True)
    args = parser.parse_args(argv)
    run = Path(args.out) / args.run_id
    if Path(args.run_id).name != args.run_id:
        raise ValueError('run-id must be one path component')
    if run.exists():
        raise FileExistsError(f'refusing implicit resume or mixed attempts: {run}')
    protocol, variant_path = Path(args.protocol), Path(args.variant_file)
    expected = [(variant_path, args.expected_variant_sha256),
                (protocol, args.expected_protocol_sha256),
                (protocol.parent / 'profiles.yaml', args.expected_profiles_sha256)]
    for path, digest in expected:
        if sha256(path) != digest:
            raise ValueError(f'input differs from the queued experiment: {path}')
    variant = json.loads(variant_path.read_text(encoding='utf-8'))
    start = time.monotonic()
    subprocess.run([sys.executable, str(PROJECT / 'pipeline/train/train_rfdetr.py'),
                    '--run-id', args.run_id, '--data-root', args.data_root, '--out', args.out,
                    '--protocol', args.protocol, '--variant-file', args.variant_file,
                    '--train-seed', str(args.train_seed), '--role', 'smoke', '--profile', args.profile,
                    '--keep-epoch-checkpoints', '2'], check=True)
    training_seconds = time.monotonic() - start
    manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
    if (manifest['epochs_completed'] != args.expected_epochs or manifest['model'] != 'rf-detr-nano'
            or manifest['seed'] != args.train_seed or manifest['dataset']['n_images'] != variant['budget']):
        raise RuntimeError('incomplete training or incorrect model, seed or image budget')
    if not (run / 'best.pth').is_file():
        raise FileNotFoundError('trainer did not retain best.pth')
    checkpoint_manifest = run / 'epoch_checkpoints/MANIFEST.json'
    epoch_checkpoints = json.loads(checkpoint_manifest.read_text(encoding='utf-8'))
    expected_indices = set(range(1, args.expected_epochs, 2))
    actual_indices = {item['epoch'] for item in epoch_checkpoints['checkpoints']}
    if epoch_checkpoints['every_n_epochs'] != 2 or actual_indices != expected_indices:
        raise RuntimeError('periodic checkpoints do not match every second completed epoch')
    for item in epoch_checkpoints['checkpoints']:
        if sha256(checkpoint_manifest.parent / item['file']) != item['sha256']:
            raise RuntimeError('periodic checkpoint hash mismatch')
    weights, output = run / 'last.pth', run / 'eval_last_epoch'
    evaluation_start = time.monotonic()
    subprocess.run([sys.executable, str(PROJECT / 'pipeline/eval/evaluate.py'),
                    '--protocol', args.protocol, '--run-id', args.run_id,
                    '--checkpoint-rule', 'last_epoch', '--set', 'val', '--checkpoint', str(weights),
                    '--annotations', str(Path(args.data_root) / 'annotations_val.json'),
                    '--images-dir', args.data_root, '--device', 'cuda', '--model-name', 'rf-detr-nano',
                    '--resolution', str(manifest['resolution']), '--output-dir', str(output)], check=True)
    evaluation_seconds = time.monotonic() - evaluation_start
    metrics = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))['results']['val']
    if metrics['n_images'] != manifest['dataset']['n_val_images']:
        raise RuntimeError('independent evaluation omitted validation frames')
    predictions = archive_predictions(output / 'predictions_val.json')
    validation = dict(status='passed', scope='network_state_frame_selection_exploratory_pilot',
                      training_seed=args.train_seed,
                      selection_seed=variant.get('selection_seed', variant.get('seed')),
                      epochs_completed=manifest['epochs_completed'], budget=variant['budget'],
                      strategy=variant['strategy'], initialization='coco_pretrained',
                      total_seconds=time.monotonic() - start, evaluation_set='val', test_evaluated=False,
                      training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
                      checkpoints={'last_epoch': dict(checkpoint_sha256=sha256(weights), metrics=metrics,
                                                       predictions_archive=predictions)},
                      best_checkpoint_retained=True, best_checkpoint_sha256=sha256(run / 'best.pth'),
                      best_map_50_95_val_training_log=manifest.get('best_map_50_95_val'),
                      best_checkpoint_independently_evaluated=False,
                      epoch_checkpoints=epoch_checkpoints,
                      epoch_checkpoints_manifest_sha256=sha256(checkpoint_manifest),
                      epoch_curve_sha256=sha256(run / 'epoch_val_map.json'),
                      input_sha256={str(path.name): digest for path, digest in expected},
                      code=dict(wrapper_sha256=sha256(Path(__file__)),
                                trainer_sha256=sha256(PROJECT / 'pipeline/train/train_rfdetr.py'),
                                evaluator_sha256=sha256(PROJECT / 'pipeline/eval/evaluate.py'),
                                git_commit=manifest.get('git_commit')),
                      versions=dict(python=platform.python_version(),
                                    libraries=manifest.get('library_versions', {})))
    with (run / 'validation_report.json').open('x', encoding='utf-8') as stream:
        json.dump(validation, stream, indent=2)
    print(json.dumps(dict(status='passed', run_id=args.run_id,
                          ap=metrics['mAP_0.5:0.95'], seconds=validation['total_seconds'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
