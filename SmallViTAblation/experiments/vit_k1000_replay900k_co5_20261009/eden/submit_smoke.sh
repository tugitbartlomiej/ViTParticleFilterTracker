#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
LOCK="$SUBMISSION/smoke.lock"
if ! mkdir "$LOCK" 2>/dev/null; then echo "Smoke jest już zgłaszany" >&2; exit 5; fi
cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT
test ! -e "$SUBMISSION/SMOKE_SUBMITTED.json"
"$PY" "$HERE/preflight.py" --mode smoke --root "$ROOT" \
  --checkpoint "$CHECKPOINT" --adamw "$CHECKPOINT_ADAMW" --coco "$COCO" --archive "$ARCHIVE" \
  --manifest "$MANIFEST" --old-index "$OLD_INDEX" --old-production "$OLD_PRODUCTION" \
  --pca "$PCA" --panel "$PANEL" --smoke-indices "$SMOKE_INDICES" --verify-old-artifact-hashes
cd "$CODE"
"$TEST_PY" -m pytest -q --import-mode=importlib test_replay.py
"$PY" -m py_compile ./*.py
MARKER="$SUBMISSION/SMOKE_SUBMITTED.json"
"$PY" submission_marker.py --path "$MARKER" --action begin --kind smoke
JOB=$(sbatch --parsable "$HERE/smoke.slurm")
"$PY" submission_marker.py --path "$MARKER" --action add --key smoke --job-id "$JOB"
"$PY" submission_marker.py --path "$MARKER" --action finish
echo "$JOB"
