#!/usr/bin/env python3
"""Queue file of stage 1: every arm of matrix_stage1/variants x three training seeds.

Reads ROOT, PROJECT_DIR, RESULTS_DIR and QUEUE_DIR from the environment, like the campaign
scripts. `--check <status.json>` verifies the final queue status instead of building.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SEEDS = [42, 1337, 2024]
BUDGET = 1500


def build() -> int:
    root, project = Path(os.environ['ROOT']), Path(os.environ['PROJECT_DIR'])
    results, queue = Path(os.environ['RESULTS_DIR']), Path(os.environ['QUEUE_DIR'])
    tasks = []
    for variant in sorted((root / 'matrix_stage1/variants').glob('matrix_*.json')):
        body = json.loads(variant.read_text(encoding='utf-8'))
        if len(body['frame_ids']) != BUDGET or len(set(body['frame_ids'])) != BUDGET:
            raise ValueError(f'{variant}: budget mismatch')
        for seed in SEEDS:
            run_id = f'{variant.stem}_s{seed}'
            if (results / run_id).exists():
                raise FileExistsError(f'No implicit retraining: {results / run_id}')
            tasks.append(dict(
                run_id=run_id, arch='rf-detr-nano', seed=seed, selection_seed=42, train_seed=seed, stage='train',
                strategy=body['strategy'], budget=BUDGET, profile='nano_static_pilot', variant_file=str(variant),
                argv=['{python}', str(project / 'pipeline/train/run_with_validation.py'), '--run-id', run_id,
                      '--data-root', '{data}', '--out', '{out}', '--protocol', '{protocol}',
                      '--variant-file', str(variant), '--train-seed', str(seed), '--profile', 'nano_static_pilot',
                      '--expected-epochs', '30', '--compress-predictions']))
    with Path(str(queue) + '.jsonl').open('x', encoding='utf-8') as stream:
        for task in tasks:
            stream.write(json.dumps(task) + '\n')
    print(json.dumps(dict(tasks=len(tasks))), flush=True)
    return 0


def check(path: str) -> int:
    status = json.loads(Path(path).read_text(encoding='utf-8'))
    counts = status['counts']
    print(json.dumps(status, indent=2))
    expected = len(list((Path(os.environ['ROOT']) / 'matrix_stage1/variants').glob('matrix_*.json'))) * len(SEEDS)
    done = counts['done'] == expected and all(counts[k] == 0 for k in ('failed', 'pending', 'running'))
    return 0 if done else 1


if __name__ == '__main__':
    raise SystemExit(check(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == '--check' else build())
