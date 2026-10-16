#!/bin/bash
#SBATCH --job-name=podprzestrzenie
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/logi/%x_%j.out
# Podprzestrzeń stała (pozycyjna) i aktywna dla tokenów, kulek i wiązek (kod/podprzestrzenie.py).
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_koloidalne"
PY="$B/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/.venv/bin/python"
SCRATCH="/dev/shm/${USER}_pp_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONUNBUFFERED=1 TMPDIR="$SCRATCH" MPLCONFIGDIR="$SCRATCH" XDG_CACHE_HOME="$SCRATCH"
cd "$EXP/kod"
"$PY" -u -B podprzestrzenie.py "$EXP/wyniki/ruch60_rp128_50tr_50val" "$EXP/wyniki/wiazki_koloidy" "$EXP/wyniki/podprzestrzenie"
echo "KONIEC $(date)"
