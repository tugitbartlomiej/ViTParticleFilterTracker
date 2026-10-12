#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
PHASE=${1:?usage: preflight.sh smoke|production}
RESUME_FLAG=${2:-}
EXTRA=()
if [[ "$RESUME_FLAG" == "--resume" ]]; then
  EXTRA+=(--allow-existing-submission)
fi

"$PY" "$HERE/preflight.py" \
  --phase "$PHASE" \
  --root "$ROOT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-adamw "$CHECKPOINT_ADAMW" \
  --coco "$COCO" \
  --archive "$ARCHIVE" \
  --manifest "$MANIFEST" "${EXTRA[@]}"

# Odtworzenie manifestu strumieniuje istniejący TAR i nie wypakowuje obrazów.
mkdir -p "$ROOT/preflight/rebuilt_manifest"
"$PY" "$CODE/przygotuj_manifest.py" \
  --coco "$COCO" \
  --checkpoint "$CHECKPOINT" \
  --archive "$ARCHIVE" \
  --output "$ROOT/preflight/rebuilt_manifest" \
  --verify-existing "$MANIFEST"

"$PY" "$CODE/snapshot_provenance.py" \
  --code "$CODE" --eden "$ROOT/eden" \
  --checkpoint "$CHECKPOINT" --checkpoint-adamw "$CHECKPOINT_ADAMW" \
  --coco "$COCO" --manifest "$MANIFEST" --archive "$ARCHIVE" \
  --output "$ROOT/preflight/RUN_PROVENANCE_${PHASE}.json"
