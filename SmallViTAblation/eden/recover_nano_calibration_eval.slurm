#!/bin/bash
# Evaluation-only recovery for three completed Nano calibration runs.
# Required: PROJECT_DIR, RUN_ROOT, DATA_ARCHIVE, PY, PROTOCOL.
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --job-name=nano_eval_recovery
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${PROJECT_DIR:?}" "${RUN_ROOT:?}" "${DATA_ARCHIVE:?}" "${PY:?}" "${PROTOCOL:?}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
cd "$PROJECT_DIR"
mkdir -p slurm_logs

STAGE="/dev/shm/${USER}_${SLURM_JOB_ID}_nano_eval_recovery"
cleanup() { rm -rf -- "$STAGE"; }
trap cleanup EXIT
mkdir -p "$STAGE"
tar -xf "$DATA_ARCHIVE" -C "$STAGE"
DATA_ROOT="$STAGE"
test -f "$DATA_ROOT/annotations_val.json"

pids=()
seeds=(42 1337 2024)
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  run_id="tracking_uniform_b500_s${seed}_calibration"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "$PY" -B pipeline/eval/recover_nano_calibration.py \
      --run-root "$RUN_ROOT" --data-root "$DATA_ROOT" \
      --protocol "$PROTOCOL" --run-id "$run_id" \
      >"slurm_logs/${run_id}_recovery_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
