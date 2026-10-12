#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
mkdir -p "$ROOT/submission"
LOCK="$ROOT/submission/smoke.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "Smoke jest już zgłaszany lub został zgłoszony" >&2
  exit 5
fi
cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT
if [[ -e "$ROOT/submission/SMOKE_SUBMITTED.json" ]]; then
  echo "Istnieje marker smoke; blokada duplikatu" >&2
  exit 6
fi
"$HERE/preflight.sh" smoke
cd "$CODE"
"$TEST_PY" -m pytest -q test_pelne_latenty.py
MARKER="$ROOT/submission/SMOKE_SUBMITTED.json"
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action begin --kind smoke
JOB=$(sbatch --parsable "$HERE/smoke.slurm")
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action add --key smoke --job-id "$JOB"
"$PY" "$CODE/submission_marker.py" --path "$MARKER" --action finish
echo "$JOB"
