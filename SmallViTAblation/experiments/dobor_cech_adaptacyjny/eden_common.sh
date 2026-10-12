#!/bin/bash
# Source only from the two Slurm entry points. No submission or tests on sourcing alone.
set -euo pipefail
: "${PROJECT_ROOT:?}" "${RESULTS_ROOT:?}" "${PY:?}" "${INITIAL_WEIGHTS:?}" "${SLURM_JOB_ID:?}"
: "${DATA_ARCHIVE:?}" "${CONTEXT_ARCHIVE:?}"
test -x "$PY" && test -f "$INITIAL_WEIGHTS" && test -f "$DATA_ARCHIVE" && test -f "$CONTEXT_ARCHIVE"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONPATH
cd "$PROJECT_ROOT"
PACKAGE=SmallViTAblation.experiments.dobor_cech_adaptacyjny
export INPUT_PROVENANCE="$RESULTS_ROOT/inputs_${SLURM_JOB_ID}.json"
"$PY" -B -m "$PACKAGE.wejscia" --data-archive "$DATA_ARCHIVE" --context-archive "$CONTEXT_ARCHIVE" --out "$INPUT_PROVENANCE"
WORK=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/features-${SLURM_JOB_ID}.XXXXXX")
# Guard the exact mktemp path, not a caller-supplied directory.
cleanup() {
    case "$WORK" in "${SLURM_TMPDIR:-/tmp}/features-${SLURM_JOB_ID}."*) rm -rf -- "$WORK" ;; esac
}
trap cleanup EXIT
DATA_ROOT=$("$PY" -B -m "$PACKAGE.stage_eden" "$DATA_ARCHIVE" "$WORK/data")
CONTEXT_ROOT=$("$PY" -B -m "$PACKAGE.stage_eden" "$CONTEXT_ARCHIVE" "$WORK/context")
export DATA_ROOT CONTEXT_ROOT
