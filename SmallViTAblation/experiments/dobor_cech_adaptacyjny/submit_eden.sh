#!/bin/bash
# Explicit invocation only. Default action submits tests, never a campaign automatically.
set -euo pipefail
: "${PROJECT_ROOT:?Set the deployed repository root on Eden}" "${RESULTS_ROOT:?Set a fresh persistent results directory}"
export PROJECT_ROOT RESULTS_ROOT
export PY="${PY:-/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python}"
export INITIAL_WEIGHTS="${INITIAL_WEIGHTS:-/home2/faculty/bpiotrowski/.roboflow/models/rf-detr-nano.pth}"
export TEST_DEPS="${TEST_DEPS:-$(dirname "$PROJECT_ROOT")/test_dependencies}"
export DATA_ARCHIVE="${DATA_ARCHIVE:-/home2/faculty/bpiotrowski/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar}"
export CONTEXT_ARCHIVE="${CONTEXT_ARCHIVE:-/home2/faculty/bpiotrowski/smallvit_runs/2026-09-22_dobor_klatek_stan_sieci/context_train.tar}"
HERE="$PROJECT_ROOT/SmallViTAblation/experiments/dobor_cech_adaptacyjny"
mkdir -p "$RESULTS_ROOT/slurm_logs"
case "${1:-tests}" in
  tests)
    sbatch --export=ALL --output="$RESULTS_ROOT/slurm_logs/%x_%j.out" \
      --error="$RESULTS_ROOT/slurm_logs/%x_%j.err" "$HERE/run_tests.slurm"
    ;;
  pilot|pilot-after-tests)
    DEPENDENCY=()
    if [[ "${1}" == pilot-after-tests ]]; then
      TEST_JOB_ID="${2:?Provide the already submitted test job ID}"
      [[ "$TEST_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm test job ID' >&2; exit 2; }
      TEST_PASS_FILE="$RESULTS_ROOT/tests_${TEST_JOB_ID}/integration/INTEGRATION_PASSED.json"
      DEPENDENCY=(--dependency="afterok:$TEST_JOB_ID" --kill-on-invalid-dep=yes)
    else
      : "${TEST_PASS_FILE:?Point to integration/INTEGRATION_PASSED.json from the completed Eden test job}"
      test -s "$TEST_PASS_FILE"
      (cd "$PROJECT_ROOT" && "$PY" -B -m SmallViTAblation.experiments.dobor_cech_adaptacyjny.gates "$TEST_PASS_FILE")
    fi
    export TEST_PASS_FILE
    for ARM in random glister features_glister features_kg_glister features_placebo_glister; do
      for SEED in 42 1337 2024; do
        # Values arrive in the process environment; avoid Eden's broken --export=ALL,key=value path.
        env ARM="$ARM" SEED="$SEED" sbatch --export=ALL "${DEPENDENCY[@]}" \
          --output="$RESULTS_ROOT/slurm_logs/%x_%j.out" --error="$RESULTS_ROOT/slurm_logs/%x_%j.err" \
          "$HERE/run_experiment.slurm"
      done
    done
    ;;
  *) echo 'Usage: bash submit_eden.sh tests|pilot|pilot-after-tests TEST_JOB_ID' >&2; exit 2 ;;
esac
