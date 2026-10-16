#!/usr/bin/env bash
set -euo pipefail
source /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009/eden/common.sh
df -B1 "$ROOT"
squeue -u "${USER:-bpiotrowski}" -o '%.18i %.28j %.8T %.10M %R' | grep -E 'JOBID|k1k_r9' || true
for path in "$SUBMISSION/SMOKE_SUBMITTED.json" "$RESULTS/smoke/SMOKE_VALIDATION.json" "$SUBMISSION/PRODUCTION_SUBMITTED.json" "$RESULTS/merged/REPLAY_VALIDATION.json"; do
  if [[ -f "$path" ]]; then echo "--- $path"; cat "$path"; fi
done
if [[ -d "$RESULTS/production/records" ]]; then
  echo "production_records=$(find "$RESULTS/production/records" -maxdepth 1 -type f -name '*.json' | wc -l)"
fi
