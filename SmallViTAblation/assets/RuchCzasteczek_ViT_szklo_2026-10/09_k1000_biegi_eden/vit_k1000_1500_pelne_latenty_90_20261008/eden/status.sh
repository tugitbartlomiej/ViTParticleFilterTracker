#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"

echo "ROOT=$ROOT"
df -B1 "$ROOT"
squeue -u "${USER:-bpiotrowski}" -o '%.18i %.28j %.8T %.10M %.6D %R' | grep -E 'JOBID|k1k1500' || true
if [[ -f "$ROOT/submission/SMOKE_SUBMITTED.json" ]]; then
  cat "$ROOT/submission/SMOKE_SUBMITTED.json"
fi
if [[ -f "$ROOT/submission/PRODUCTION_SUBMITTED.json" ]]; then
  cat "$ROOT/submission/PRODUCTION_SUBMITTED.json"
fi
if [[ -f "$RESULTS/smoke/SMOKE_VALIDATION.json" ]]; then
  "$PY" - "$RESULTS/smoke/SMOKE_VALIDATION.json" <<'PY'
import json, pathlib, sys
j=json.loads(pathlib.Path(sys.argv[1]).read_text())
print(json.dumps({"smoke": j.get("status"), "benchmark": j.get("benchmark"), "alarm": j.get("alarm_from_independent_local_pilot_addressed")}, indent=2))
PY
fi
if [[ -d "$RESULTS/production/records" ]]; then
  echo "production_records=$(find "$RESULTS/production/records" -maxdepth 1 -type f -name '*.json' | wc -l)"
fi
if [[ -f "$RESULTS/FINAL_STATUS.json" ]]; then
  "$PY" - "$RESULTS/FINAL_STATUS.json" <<'PY'
import json, pathlib, sys
j=json.loads(pathlib.Path(sys.argv[1]).read_text())
print(json.dumps({"final_status": j.get("status"), "finished_utc": j.get("finished_utc")}, indent=2))
PY
fi
