#!/bin/bash
# Submit exactly one 64-frame RF-DETR Nano smoke including checkpoint reloads.
# Usage: PY=/absolute/venv/bin/python DATA_ARCHIVE=/absolute/old64.tar bash eden/submit_nano_smoke.sh
set -euo pipefail
export PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
: "${PY:?Set the clean RF-DETR Python executable}"
: "${DATA_ARCHIVE:?Set the prepared smoke archive}"
export PY DATA_ARCHIVE
export RUN_TAG="${RUN_TAG:-old64_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$PY" == /* && -x "$PY" && "$DATA_ARCHIVE" == /* && -f "$DATA_ARCHIVE" ]]
export QUEUE_DIR="$PROJECT_DIR/eden/queues/$RUN_TAG"
export TRAINING_RUN_ID="${EXISTING_RUN_ID:-$RUN_TAG}"
[[ "$TRAINING_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]]
export RESULTS_DIR="$PROJECT_DIR/results/$TRAINING_RUN_ID"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
if [[ -f "$PROJECT_DIR/../SOURCE_COMMIT" ]]; then
    export SMALLVIT_SOURCE_COMMIT="$(cat "$PROJECT_DIR/../SOURCE_COMMIT")"
fi
cd "$PROJECT_DIR"
mkdir -p slurm_logs eden/queues
"$PY" - <<'PY'
import json, os
from pathlib import Path
project = Path(os.environ['PROJECT_DIR'])
tag = os.environ['RUN_TAG']
queue = Path(os.environ['QUEUE_DIR'])
training_id = os.environ['TRAINING_RUN_ID']
eval_only = bool(os.environ.get('EXISTING_RUN_ID'))
results = Path(os.environ['RESULTS_DIR'])
if queue.exists() or (results.exists() and not eval_only):
    raise FileExistsError('Use a fresh RUN_TAG; no overwrite or implicit resume')
if eval_only and not (results / training_id / 'run_manifest.json').is_file():
    raise FileNotFoundError('Evaluation-only mode requires a completed training manifest')
task = {'run_id': tag, 'arch': 'rf-detr-nano', 'seed': 42, 'train_seed': 42,
        'profile': 'nano_smoke', 'variant_file': 'variants/smoke64.json',
        'argv': ['{python}', str(project / 'pipeline/train/smoke_rfdetr.py'),
                 '--run-id', training_id, '--data-root', '{data}', '--out', '{out}',
                 '--protocol', '{protocol}', '--variant-file', '{data}/variants/smoke64.json']}
if eval_only:
    task['argv'].append('--eval-only')
with (queue.parent / (tag + '.jsonl')).open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(task)+'\n')
PY
"$PY" eden/queue.py init "$QUEUE_DIR" --queue-file "$QUEUE_DIR.jsonl"
"$PY" -m pip freeze > "$QUEUE_DIR/environment.txt"
sha256sum "$DATA_ARCHIVE" > "$QUEUE_DIR/data_archive.sha256"
sbatch --parsable --export=ALL eden/run_smoke.slurm | tee "$QUEUE_DIR/slurm_job_id.txt"
