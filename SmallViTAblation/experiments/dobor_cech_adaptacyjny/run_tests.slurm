#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --job-name=features_kg_tests
set -euo pipefail
: "${PROJECT_ROOT:?}" "${RESULTS_ROOT:?}" "${PY:?}" "${INITIAL_WEIGHTS:?}"
cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTHONPATH
PACKAGE=SmallViTAblation.experiments.dobor_cech_adaptacyjny
OUT="$RESULTS_ROOT/tests_${SLURM_JOB_ID}"
mkdir "$OUT"
"$PY" -B -m "$PACKAGE.preflight" --initial-weights "$INITIAL_WEIGHTS" > "$OUT/preflight.json"
if [[ -n "${TEST_DEPS:-}" ]]; then
    test -f "$TEST_DEPS/pytest/__init__.py"
    export PYTHONPATH="$TEST_DEPS"
fi
"$PY" -B -m pytest -q --import-mode=importlib SmallViTAblation/experiments/dobor_cech_adaptacyjny/tests \
    --junitxml="$OUT/unit_tests.xml" | tee "$OUT/unit_tests.log"
unset PYTHONPATH
source SmallViTAblation/experiments/dobor_cech_adaptacyjny/eden_common.sh
"$PY" -B -m "$PACKAGE.integracja_eden" --data-root "$DATA_ROOT" --context-root "$CONTEXT_ROOT" \
    --out "$OUT/integration" --initial-weights "$INITIAL_WEIGHTS" | tee "$OUT/integration.log"
echo "Eden contract and integration tests completed: $OUT"
