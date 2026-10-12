#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --job-name=features_kg_pilot
set -euo pipefail
JOB_START=$SECONDS
: "${PROJECT_ROOT:?}" "${TEST_PASS_FILE:?}" "${ARM:?}" "${SEED:?}"
test -s "$TEST_PASS_FILE"
source "$PROJECT_ROOT/SmallViTAblation/experiments/dobor_cech_adaptacyjny/eden_common.sh"
"$PY" -B -m "$PACKAGE.gates" "$TEST_PASS_FILE"
OUT="$RESULTS_ROOT/${ARM}_s${SEED}"
test ! -e "$OUT"
"$PY" -B -m "$PACKAGE.preflight" --initial-weights "$INITIAL_WEIGHTS"
PROTOCOL="$RESULTS_ROOT/protocol_${ARM}_s${SEED}"
ARGS=(--data-root "$DATA_ROOT" --context-root "$CONTEXT_ROOT" --out "$PROTOCOL")
if [[ -n "${FEATURE_ANNOTATIONS:-}" ]]; then ARGS+=(--annotations "$FEATURE_ANNOTATIONS"); fi
"$PY" -B -m "$PACKAGE.przygotuj" "${ARGS[@]}"
"$PY" -B -m "$PACKAGE.trenuj" --protocol "$PROTOCOL" --out "$OUT" \
    --initial-weights "$INITIAL_WEIGHTS" --arm "$ARM" --seed "$SEED"
printf '{"job_elapsed_seconds": %s, "includes": "archive extraction, preparation, initialization, training, validation, selection, artifacts", "includes_slurm_wait": false}\n' \
    "$((SECONDS-JOB_START))" > "$OUT/job_timing.json"
