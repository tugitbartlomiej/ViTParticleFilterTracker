#!/bin/bash
# 36 RF-DETR Nano runs: 4 selectors x 3 budgets x 3 training seeds.
# Required: PROJECT_DIR, PREPARED_DIR, QUEUE_DIR, RESULTS_DIR, PY, PROTOCOL.
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodelist=hopper
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --job-name=nano_static36
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${PROJECT_DIR:?}" "${PREPARED_DIR:?}" "${QUEUE_DIR:?}" "${RESULTS_DIR:?}"
: "${PY:?}" "${PROTOCOL:?}"
export DATA_ARCHIVE="${PREPARED_DIR}.tar" N_WORKERS=8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONPATH
cd "$PROJECT_DIR"
test -s "$DATA_ARCHIVE"
mkdir -p "$(dirname "$QUEUE_DIR")" "$RESULTS_DIR" slurm_logs
"$PY" -B - <<'PY'
import json, os
from pathlib import Path
project = Path(os.environ['PROJECT_DIR'])
prepared = Path(os.environ['PREPARED_DIR'])
queue = Path(os.environ['QUEUE_DIR'])
results = Path(os.environ['RESULTS_DIR'])
tasks = []
for strategy in ['random', 'uniform_time', 'dino_coverage', 'dino_temporal_hybrid']:
    for budget in [500, 1500, 2500]:
        for seed in [42, 1337, 2024]:
            run_id = f'tracking_{strategy}_b{budget}_s{seed}'
            if (results/run_id).exists():
                raise FileExistsError(f'No implicit retraining: {results/run_id}')
            variant = f'variants/{run_id}.json'
            tasks.append(dict(run_id=run_id, arch='rf-detr-nano', seed=seed,
                selection_seed=42, train_seed=seed, stage='train', strategy=strategy,
                budget=budget, profile='nano_static_pilot', variant_file=variant,
                argv=['{python}', str(project/'pipeline/train/run_with_validation.py'),
                      '--run-id', run_id, '--data-root', '{data}', '--out', '{out}',
                      '--protocol', '{protocol}', '--variant-file', '{data}/'+variant,
                      '--train-seed', str(seed), '--profile', 'nano_static_pilot',
                      '--expected-epochs', '30', '--compress-predictions']))
queue_file = Path(str(queue) + '.jsonl')
with queue_file.open('x', encoding='utf-8') as stream:
    for task in tasks:
        stream.write(json.dumps(task) + '\n')
print({'tasks': len(tasks), 'queue_file': str(queue_file)}, flush=True)
PY
"$PY" eden/queue.py init "$QUEUE_DIR" --queue-file "${QUEUE_DIR}.jsonl"
"$PY" -B -m pip freeze > "${QUEUE_DIR}.environment.txt"
sha256sum "$DATA_ARCHIVE" "$PROTOCOL" > "${QUEUE_DIR}.inputs.sha256"
bash eden/run_node.slurm
"$PY" eden/queue.py status "$QUEUE_DIR" --json | "$PY" -c '
import json, sys
s=json.load(sys.stdin); print(json.dumps(s, indent=2)); c=s["counts"]
sys.exit(0 if c["done"] == 36 and all(c[k] == 0 for k in ("failed","pending","running")) else 1)
'
