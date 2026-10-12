#!/bin/bash
#SBATCH --job-name=krysztal_wiz
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/logi/%x_%j.out
# Wizualizacja kryształu (kod/krysztal_wizualizacja.py): oś aplikacji + panel PNG (CPU).
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_koloidalne"
PY="$B/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/.venv/bin/python"
SCRATCH="/dev/shm/${USER}_kw_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONUNBUFFERED=1 TMPDIR="$SCRATCH" MPLCONFIGDIR="$SCRATCH" XDG_CACHE_HOME="$SCRATCH"
cd "$EXP/kod"
"$PY" -u -B krysztal_wizualizacja.py "$EXP/wyniki/ruch60_rp128_50tr_50val" "$EXP/wyniki/pomiar_pelny60_3000tr_1000val" "$EXP/wyniki" \
      "$EXP/wyniki/os_szklo_krysztal_60ep" "$EXP/wyniki/krysztal_panel"
echo "KONIEC $(date)"
