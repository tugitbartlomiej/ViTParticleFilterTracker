#!/bin/bash
# Stage 1 of the controlled frame-selection experiment (document 39): 9 arms x 3 training seeds,
# budget 1500 from one shared L0 of 500 frames, static campaign recipe, at most four GPUs.
# Trains the 27 runs. The arms are built beforehand by run_selection_stage1_prep.slurm (afterok).
# Required: ROOT, PROJECT_DIR, DATA_ARCHIVE, PROTOCOL, PY.
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodelist=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --job-name=kl_stage1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${ROOT:?}" "${PROJECT_DIR:?}" "${DATA_ARCHIVE:?}" "${PROTOCOL:?}" "${PY:?}"
MATRIX="$ROOT/matrix_stage1"
export QUEUE_DIR="$ROOT/queue/stage1" RESULTS_DIR="$ROOT/results_stage1" N_WORKERS=4
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONPATH
test -s "$MATRIX/variants/MATRIX_MANIFEST.json" && test "$(ls "$MATRIX"/variants/matrix_*.json | wc -l)" -eq 9
mkdir -p "$(dirname "$QUEUE_DIR")" "$RESULTS_DIR" "$PROJECT_DIR/slurm_logs"

# (c) 27 training runs through the campaign queue
cd "$PROJECT_DIR"
"$PY" -B "$PROJECT_DIR/eden/build_stage1_queue.py"
"$PY" eden/queue.py init "$QUEUE_DIR" --queue-file "${QUEUE_DIR}.jsonl"
sha256sum "$DATA_ARCHIVE" "$PROTOCOL" "$MATRIX"/variants/*.json "$MATRIX/uncertainty_L0.json" > "${QUEUE_DIR}.inputs.sha256"
bash eden/run_node.slurm
"$PY" eden/queue.py status "$QUEUE_DIR" --json > "${QUEUE_DIR}.final_status.json"
"$PY" -B "$PROJECT_DIR/eden/build_stage1_queue.py" --check "${QUEUE_DIR}.final_status.json"
