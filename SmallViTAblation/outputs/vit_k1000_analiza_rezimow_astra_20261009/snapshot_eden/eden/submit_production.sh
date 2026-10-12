#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
mkdir -p "$ROOT/submission"
LOCK="$ROOT/submission/production.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "Produkcja jest już zgłaszana lub została zgłoszona" >&2
  exit 5
fi
cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT
if [[ -e "$ROOT/submission/PRODUCTION_SUBMITTED.json" ]]; then
  echo "Istnieje marker produkcji; blokada duplikatu" >&2
  exit 6
fi
"$PY" - "$RESULTS/smoke/SMOKE_VALIDATION.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.is_file() or json.loads(p.read_text())["status"] != "pass":
    raise SystemExit("Smoke nie przeszedł; produkcja zablokowana")
PY
"$HERE/preflight.sh" production

MARKER="$ROOT/submission/PRODUCTION_SUBMITTED.json"
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action begin --kind production
PRODUCTION=$(sbatch --parsable "$HERE/production.slurm")
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action add --key production_array --job-id "$PRODUCTION"
PRODUCTION_DEP=${PRODUCTION%%;*}
MERGE=$(sbatch --parsable --dependency="afterok:$PRODUCTION_DEP" "$HERE/merge.slurm")
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action add --key merge --job-id "$MERGE"
MERGE_DEP=${MERGE%%;*}
MOVEMENT=$(sbatch --parsable --dependency="afterok:$MERGE_DEP" "$HERE/movement.slurm")
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action add --key movement --job-id "$MOVEMENT"
REGIMES=$(sbatch --parsable --dependency="afterok:$MERGE_DEP" "$HERE/regimes.slurm")
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action add --key regimes --job-id "$REGIMES"
DMDC=$(sbatch --parsable --dependency="afterok:$MERGE_DEP" "$HERE/dmdc.slurm")
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action add --key dmdc --job-id "$DMDC"
REGIMES_DEP=${REGIMES%%;*}
EVALUATION=$(sbatch --parsable --dependency="afterok:$REGIMES_DEP" "$HERE/package_eval.slurm")
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action add --key evaluation --job-id "$EVALUATION"
MOVEMENT_DEP=${MOVEMENT%%;*}
DMDC_DEP=${DMDC%%;*}
EVALUATION_DEP=${EVALUATION%%;*}
FINAL=$(sbatch --parsable --dependency="afterok:$MOVEMENT_DEP:$DMDC_DEP:$EVALUATION_DEP" "$HERE/finalize.slurm")
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action add --key finalize --job-id "$FINAL"
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action finish
cat "$MARKER"
