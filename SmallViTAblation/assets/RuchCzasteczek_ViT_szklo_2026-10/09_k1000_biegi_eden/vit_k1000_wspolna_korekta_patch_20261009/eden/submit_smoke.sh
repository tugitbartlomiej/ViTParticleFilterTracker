#!/usr/bin/env bash
set -euo pipefail
source /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wspolna_korekta_patch_20261009/eden/common.sh
test ! -e "$SUBMISSION/SMOKE_SUBMITTED.json"
job_id=$(sbatch --parsable "$ROOT/eden/smoke.slurm")
"$PY" - "$SUBMISSION/SMOKE_SUBMITTED.json" "$job_id" <<'PY'
import json, os, sys, tempfile
path, job = sys.argv[1:]
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix='smoke.')
with os.fdopen(fd, 'w') as f: json.dump({'job_id': job, 'status': 'submitted'}, f); f.write('\n')
os.replace(tmp, path)
PY
printf '%s\n' "$job_id"
