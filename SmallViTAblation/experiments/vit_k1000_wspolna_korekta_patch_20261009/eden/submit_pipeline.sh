#!/usr/bin/env bash
set -euo pipefail
source /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wspolna_korekta_patch_20261009/eden/common.sh
"$PY" "$ROOT/eden/preflight.py" --mode production --root "$ROOT" \
  --checkpoint "$CHECKPOINT" --adamw "$CHECKPOINT_ADAMW" --coco "$COCO" \
  --archive "$ARCHIVE" --manifest "$MANIFEST" --old-evaluation "$OLD_EVALUATION"
train_id=$(sbatch --parsable "$ROOT/eden/production.slurm")
final_id=$(sbatch --parsable --dependency="afterok:$train_id" "$ROOT/eden/finalize.slurm")
"$PY" - "$SUBMISSION/PRODUCTION_SUBMITTED.json" "$train_id" "$final_id" <<'PY'
import json, os, sys, tempfile
path, train, final = sys.argv[1:]
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix='production.')
with os.fdopen(fd, 'w') as f:
    json.dump({'training_job_id': train, 'finalize_job_id': final, 'status': 'submitted'}, f)
    f.write('\n')
os.replace(tmp, path)
PY
printf '%s %s\n' "$train_id" "$final_id"
