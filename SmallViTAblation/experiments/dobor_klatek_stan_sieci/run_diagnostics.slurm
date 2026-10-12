#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --job-name=token_checkpoints
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
: "${ROOT:?}" "${PROJECT_DIR:?}" "${DATA_ARCHIVE:?}" "${PY:?}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONPATH
WORK="/dev/shm/${USER}_${SLURM_JOB_ID}_token_diagnostics"
cleanup() { case "$WORK" in /dev/shm/*_token_diagnostics) rm -rf -- "$WORK" ;; *) exit 2 ;; esac; }
trap cleanup EXIT
trap 'exit 130' INT TERM
test ! -e "$WORK"
mkdir -p "$WORK"
tar -xf "$DATA_ARCHIVE" -C "$WORK"
cd "$ROOT"
"$PY" -B -m SmallViTAblation.experiments.dobor_klatek_stan_sieci.diagnostyka_checkpointow \
 --project-dir "$PROJECT_DIR" --data-root "$WORK" --selection-root "$ROOT/selection" \
 --pilot-root "$ROOT/pilot" --out "$ROOT/diagnostyka_checkpointow"
"$PY" -B -m SmallViTAblation.experiments.dobor_klatek_stan_sieci.uruchom \
 --output-root "$ROOT/pilot" --report > "$ROOT/WYNIKI_PILOT.json"
