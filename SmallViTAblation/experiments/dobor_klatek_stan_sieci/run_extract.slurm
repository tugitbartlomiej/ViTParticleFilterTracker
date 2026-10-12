#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=token_extract
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
: "${ROOT:?}" "${CONTEXT_ARCHIVE:?}" "${PY:?}" "${L0_CHECKPOINT:?}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONPATH
WORK="/dev/shm/${USER}_${SLURM_JOB_ID}_token_extract"
cleanup() { case "$WORK" in /dev/shm/*_token_extract) rm -rf -- "$WORK" ;; *) exit 2 ;; esac; }
trap cleanup EXIT
trap 'exit 130' INT TERM
test ! -e "$WORK"
mkdir -p "$WORK" "$ROOT/cache"
tar -xf "$CONTEXT_ARCHIVE" -C "$WORK"
cd "$ROOT/source_extract"
"$PY" -B -m SmallViTAblation.experiments.dobor_klatek_stan_sieci.token_ekstrakcja \
 --data-root "$WORK" --checkpoint "$L0_CHECKPOINT" --out "$ROOT/cache/L0_tokens" --batch-size 32
