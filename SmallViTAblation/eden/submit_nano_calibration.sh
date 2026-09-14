#!/bin/bash
# Submit only three calibration runs after successful tracking smoke.
# Required: PY, DATA_ARCHIVE, PREFLIGHT_REPORT, PROJECT_DIR.
set -euo pipefail
: "${PY:?}" "${DATA_ARCHIVE:?}" "${PREFLIGHT_REPORT:?}" "${PROJECT_DIR:?}"
export PY DATA_ARCHIVE PROJECT_DIR PREFLIGHT_REPORT
export PROTOCOL="$PROJECT_DIR/configs/protocol_tracking_calibration.yaml"
export QUEUE_DIR="$PROJECT_DIR/eden/queues/tracking_calibration_$(date -u +%Y%m%dT%H%M%SZ)"
export RESULTS_DIR="$PROJECT_DIR/results/calibration"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export SMALLVIT_SOURCE_COMMIT="$(cat "$PROJECT_DIR/../SOURCE_COMMIT")"
unset PYTHONPATH
cd "$PROJECT_DIR"
mkdir -p slurm_logs eden/queues
"$PY" -B - <<'PY'
import json, os
from pathlib import Path
project = Path(os.environ['PROJECT_DIR'])
report = json.loads(Path(os.environ['PREFLIGHT_REPORT']).read_text())
if report.get('status') != 'passed' or report.get('evaluation_only_job'):
    raise RuntimeError('Requires a complete successful tracking training smoke')
queue = Path(os.environ['QUEUE_DIR'])
tasks = []
for seed in [42, 1337, 2024]:
    run_id = f'tracking_uniform_b500_s{seed}_calibration'
    if (Path(os.environ['RESULTS_DIR'])/run_id).exists():
        raise FileExistsError('Calibration output exists; no implicit retraining')
    tasks.append(dict(run_id=run_id, arch='rf-detr-nano', seed=seed,
        stage='train', strategy='uniform_time', budget=500,
        profile='nano_static_pilot', train_seed=seed,
        variant_file=f'variants/calibration_s{seed}.json',
        argv=['{python}', str(project/'pipeline/train/run_with_validation.py'),
              '--run-id', run_id, '--data-root', '{data}', '--out', '{out}',
              '--protocol', '{protocol}', '--variant-file', '{data}/variants/calibration_s'+str(seed)+'.json',
              '--train-seed', str(seed), '--profile', 'nano_static_pilot', '--expected-epochs', '30']))
with Path(str(queue)+'.jsonl').open('x') as f:
    for task in tasks:
        f.write(json.dumps(task)+'\n')
PY
"$PY" eden/queue.py init "$QUEUE_DIR" --queue-file "$QUEUE_DIR.jsonl"
"$PY" -B -m pip freeze > "$QUEUE_DIR/environment.txt"
sha256sum "$DATA_ARCHIVE" > "$QUEUE_DIR/data_archive.sha256"
sha256sum "$PROTOCOL" > "$QUEUE_DIR/protocol.sha256"
sbatch --parsable --export=ALL eden/run_nano_calibration.slurm | tee "$QUEUE_DIR/slurm_job_id.txt"
