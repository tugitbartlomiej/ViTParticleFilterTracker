#!/bin/bash
#SBATCH --job-name=slepa
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/logi/%x_%j.out
# Ślepa klasyfikacja klasy dynamiki: wzorce Langevina 2D o znanej klasie kontra tokeny (kod/slepa_klasyfikacja.py).
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_koloidalne"
PY="$B/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/.venv/bin/python"
SCRATCH="/dev/shm/${USER}_slepa_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH" "$EXP/wyniki/slepa"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TMPDIR="$SCRATCH" MPLCONFIGDIR="$SCRATCH" XDG_CACHE_HOME="$SCRATCH"
cd "$EXP/kod"
echo "start $(date), węzeł $(hostname)"
"$PY" -u -B slepa_klasyfikacja.py "$EXP/wyniki/ruch60_rp128_50tr_50val" "$B/smallvit_runs/2026-10-10_szklo_pokretla/wyniki" "$EXP/wyniki/slepa" 32
echo "KONIEC $(date)"
