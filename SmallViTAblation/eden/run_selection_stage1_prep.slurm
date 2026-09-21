#!/bin/bash
# Preparation of stage 1 (document 39): confidence of the L0 detector on the pool (arm B4) and
# construction of all nine arms from the shared L0. One GPU; the four-GPU training job waits for it.
# It also builds and initialises the shared run queue, so that several one-GPU workers can start after it.
# Required: ROOT, PROJECT_DIR, DATA_ARCHIVE, PROTOCOL, PY, L0_CHECKPOINT, L0_VARIANT, DINO_CACHE.
#SBATCH -A transformers_vsc
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=kl_stage1_prep
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${ROOT:?}" "${PROJECT_DIR:?}" "${DATA_ARCHIVE:?}" "${PROTOCOL:?}" "${PY:?}" "${L0_CHECKPOINT:?}" "${L0_VARIANT:?}" "${DINO_CACHE:?}"
export QUEUE_DIR="$ROOT/queue/stage1" RESULTS_DIR="$ROOT/results_stage1"
MATRIX="$ROOT/matrix_stage1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
unset PYTHONPATH
PRE="/dev/shm/${USER}_${SLURM_JOB_ID:-$$}_kl_stage1_pre"
cleanup_pre() { rm -rf -- "$PRE" 2>/dev/null || true; }
trap cleanup_pre EXIT INT TERM
test -s "$DATA_ARCHIVE" && test -s "$L0_CHECKPOINT" && test -d "$DINO_CACHE"
mkdir -p "$PRE/data" "$MATRIX"

# (a) confidence of the L0 detector on the pool: one GPU, a few minutes
tar -xf "$DATA_ARCHIVE" -C "$PRE/data"
test -s "$PRE/data/variants/$L0_VARIANT"
cp "$PRE/data/variants/$L0_VARIANT" "$MATRIX/L0_$L0_VARIANT"
if [[ ! -s "$MATRIX/uncertainty_L0.json" ]]; then
    TMPDIR="$PRE" "$PY" -B "$PROJECT_DIR/pipeline/koopman_langevin/pool_confidence.py" \
        --data-root "$PRE/data" --checkpoint "$L0_CHECKPOINT" --out "$MATRIX/uncertainty_L0.json"
fi

# (b) every arm from the shared L0; labels are not read
if [[ ! -s "$MATRIX/variants/MATRIX_MANIFEST.json" ]]; then
    (cd "$ROOT" && "$PY" -B -m SmallViTAblation.pipeline.koopman_langevin.selection_matrix \
        --pool "$PRE/data/pool_manifest.jsonl" --cache "$DINO_CACHE" \
        --initial-selection "$MATRIX/L0_$L0_VARIANT" --uncertainty "$MATRIX/uncertainty_L0.json" \
        --budget 1500 --out "$MATRIX/variants")
fi
cleanup_pre
test "$(ls "$MATRIX"/variants/matrix_*.json | wc -l)" -eq 9
ls -la "$MATRIX/variants"

# shared queue of 27 runs, initialised once; workers only claim from it
mkdir -p "$(dirname "$QUEUE_DIR")" "$RESULTS_DIR"
cd "$PROJECT_DIR"
"$PY" -B "$PROJECT_DIR/eden/build_stage1_queue.py"
"$PY" eden/queue.py init "$QUEUE_DIR" --queue-file "${QUEUE_DIR}.jsonl"
sha256sum "$DATA_ARCHIVE" "$PROTOCOL" "$MATRIX"/variants/*.json "$MATRIX/uncertainty_L0.json" > "${QUEUE_DIR}.inputs.sha256"
"$PY" eden/queue.py status "$QUEUE_DIR"
