#!/bin/bash
#SBATCH --job-name=pozycja
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/logi/%x_%j.out
# Czy centrum drgań tokenu wyznacza położenie patcha z kroku 0 (kod/pozycja_patchy.py), CPU.
#   sbatch pozycja_patchy.slurm
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_koloidalne"
PY="$B/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/.venv/bin/python"
DANE="$EXP/wyniki/ruch60_rp128_50tr_50val"
OUT="$EXP/wyniki/pozycja_patchy"
SCRATCH="/dev/shm/${USER}_pozycja_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH" "$OUT" "$EXP/logi"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH" MPLCONFIGDIR="$SCRATCH" XDG_CACHE_HOME="$SCRATCH"
[ -f "$DANE/META.json" ] || { echo "BRAK $DANE/META.json"; exit 2; }
cd "$EXP/kod"
echo "start $(date), węzeł $(hostname)"
"$PY" -u -B pozycja_patchy.py "$DANE" "$OUT"
echo "KONIEC $(date)"
