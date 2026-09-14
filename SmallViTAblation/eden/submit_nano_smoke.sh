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
export RESULTS_DIR="$PROJECT_DIR/results/$RUN_TAG"
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
if queue.exists() or Path(os.environ['RESULTS_DIR']).exists():
    raise FileExistsError('Use a fresh RUN_TAG; no overwrite or implicit resume')
task = {'run_id': tag, 'arch': 'rf-detr-nano', 'seed': 42, 'train_seed': 42,
        'profile': 'nano_smoke', 'variant_file': 'variants/smoke64.json',
        'argv': ['{python}', str(project / 'pipeline/train/smoke_rfdetr.py'),
                 '--run-id', '{run_id}', '--data-root', '{data}', '--out', '{out}',
                 '--protocol', '{protocol}', '--variant-file', '{data}/variants/smoke64.json']}
with (queue.parent / (tag + '.jsonl')).open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(task)+'\n')
PY
"$PY" eden/queue.py init "$QUEUE_DIR" --queue-file "$QUEUE_DIR.jsonl"
"$PY" -m pip freeze > "$QUEUE_DIR/environment.txt"
sha256sum "$DATA_ARCHIVE" > "$QUEUE_DIR/data_archive.sha256"
sbatch --parsable --export=ALL eden/run_smoke.slurm | tee "$QUEUE_DIR/slurm_job_id.txt"
