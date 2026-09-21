#!/bin/bash
# Decisive test of the Langevin frame selections: 4 variants x 3 training seeds at budget 1500,
# same recipe, data archive and protocol as the static campaign (document 35), at most four GPUs.
# Baseline for comparison already exists: tracking_uniform_time_b1500_s{42,1337,2024}.
# Required: ROOT (with variants/), PROJECT_DIR, DATA_ARCHIVE, PROTOCOL, PY.
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodelist=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --job-name=kl_select12
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${ROOT:?}" "${PROJECT_DIR:?}" "${DATA_ARCHIVE:?}" "${PROTOCOL:?}" "${PY:?}"
export QUEUE_DIR="$ROOT/queue/select12" RESULTS_DIR="$ROOT/results_selection" N_WORKERS=4
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONPATH
cd "$PROJECT_DIR"
test -s "$DATA_ARCHIVE"
mkdir -p "$(dirname "$QUEUE_DIR")" "$RESULTS_DIR" slurm_logs
"$PY" -B - <<'PY'
import json, os
from pathlib import Path
root, project = Path(os.environ['ROOT']), Path(os.environ['PROJECT_DIR'])
results, queue = Path(os.environ['RESULTS_DIR']), Path(os.environ['QUEUE_DIR'])
tasks = []
for strategy in ['langevin_reducible', 'langevin_surprise', 'langevin_veto_uniform', 'langevin_veto_coverage']:
    variant = root / 'variants' / f'tracking_{strategy}_b1500_sel42.json'
    if len(json.loads(variant.read_text())['frame_ids']) != 1500:
        raise ValueError(f'{variant}: budget mismatch')
    for seed in [42, 1337, 2024]:
        run_id = f'tracking_{strategy}_b1500_s{seed}'
        if (results / run_id).exists():
            raise FileExistsError(f'No implicit retraining: {results / run_id}')
        tasks.append(dict(run_id=run_id, arch='rf-detr-nano', seed=seed, selection_seed=42, train_seed=seed,
            stage='train', strategy=strategy, budget=1500, profile='nano_static_pilot', variant_file=str(variant),
            argv=['{python}', str(project / 'pipeline/train/run_with_validation.py'), '--run-id', run_id,
                  '--data-root', '{data}', '--out', '{out}', '--protocol', '{protocol}',
                  '--variant-file', str(variant), '--train-seed', str(seed), '--profile', 'nano_static_pilot',
                  '--expected-epochs', '30', '--compress-predictions']))
with Path(str(queue) + '.jsonl').open('x', encoding='utf-8') as stream:
    for task in tasks:
        stream.write(json.dumps(task) + '\n')
print({'tasks': len(tasks)}, flush=True)
PY
"$PY" eden/queue.py init "$QUEUE_DIR" --queue-file "${QUEUE_DIR}.jsonl"
sha256sum "$DATA_ARCHIVE" "$PROTOCOL" "$ROOT"/variants/*.json > "${QUEUE_DIR}.inputs.sha256"
bash eden/run_node.slurm
"$PY" eden/queue.py status "$QUEUE_DIR" --json | "$PY" -c '
import json, sys
s=json.load(sys.stdin); print(json.dumps(s, indent=2)); c=s["counts"]
sys.exit(0 if c["done"] == 12 and all(c[k] == 0 for k in ("failed","pending","running")) else 1)
'
