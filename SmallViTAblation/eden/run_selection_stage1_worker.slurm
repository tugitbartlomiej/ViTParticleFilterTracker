#!/bin/bash
# One-GPU worker of stage 1 (document 39). Several of these run in parallel, on different nodes,
# and claim runs from the shared queue that run_selection_stage1_prep.slurm initialised.
# Partition is given on the sbatch command line (-p short for the A100 nodes, -p hopper for H100).
# Required: ROOT, PROJECT_DIR, DATA_ARCHIVE, PROTOCOL, PY.
#SBATCH -A transformers_vsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --job-name=kl_s1_worker
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${ROOT:?}" "${PROJECT_DIR:?}" "${DATA_ARCHIVE:?}" "${PROTOCOL:?}" "${PY:?}"
export QUEUE_DIR="$ROOT/queue/stage1" RESULTS_DIR="$ROOT/results_stage1" N_WORKERS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONPATH
test -d "$QUEUE_DIR" && test -s "$DATA_ARCHIVE"
cd "$PROJECT_DIR"
bash eden/run_node.slurm
