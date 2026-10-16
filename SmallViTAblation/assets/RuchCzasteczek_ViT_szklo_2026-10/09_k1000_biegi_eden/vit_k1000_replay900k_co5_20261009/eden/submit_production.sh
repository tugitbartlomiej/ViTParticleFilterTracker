#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
LOCK="$SUBMISSION/production.lock"
if ! mkdir "$LOCK" 2>/dev/null; then echo "Produkcja jest już zgłaszana" >&2; exit 5; fi
cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT
test ! -e "$SUBMISSION/PRODUCTION_SUBMITTED.json"
"$PY" - "$RESULTS/smoke/SMOKE_VALIDATION.json" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
if not p.is_file() or json.loads(p.read_text())["status"] != "pass":
    raise SystemExit("Smoke nie ma statusu pass")
PY
"$PY" "$HERE/preflight.py" --mode production --root "$ROOT" \
  --checkpoint "$CHECKPOINT" --adamw "$CHECKPOINT_ADAMW" --coco "$COCO" --archive "$ARCHIVE" \
  --manifest "$MANIFEST" --old-index "$OLD_INDEX" --old-production "$OLD_PRODUCTION" \
  --pca "$PCA" --panel "$PANEL" --smoke-indices "$SMOKE_INDICES" \
  --analysis-pca32 "$PCA32" --analysis-pca32-manifest "$PCA32_MANIFEST"
MARKER="$SUBMISSION/PRODUCTION_SUBMITTED.json"
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action begin --kind production
PRODUCTION=$(sbatch --parsable "$HERE/production.slurm")
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action add --key production_array --job-id "$PRODUCTION"
PRODUCTION_DEP=${PRODUCTION%%;*}
MERGE=$(sbatch --parsable --dependency="afterok:$PRODUCTION_DEP" "$HERE/merge.slurm")
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action add --key merge --job-id "$MERGE"
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action finish
cat "$MARKER"
