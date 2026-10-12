#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
[[ -f "$ROOT/submission/PRODUCTION_SUBMITTED.json" ]] || { echo "Brak produkcji do wznowienia" >&2; exit 2; }
[[ ! -d "$ROOT/submission/resume.lock" ]] || { echo "Wznowienie już trwa" >&2; exit 3; }
mkdir "$ROOT/submission/resume.lock"
trap 'rmdir "$ROOT/submission/resume.lock" 2>/dev/null || true' EXIT
"$HERE/preflight.sh" production --resume
MARKER="$ROOT/submission/RESUME_$(date -u +%Y%m%dT%H%M%SZ).json"
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action begin --kind production_resume
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
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action finish --final-status resume_submitted
echo "$PRODUCTION"
