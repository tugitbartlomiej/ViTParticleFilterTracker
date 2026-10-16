#!/bin/bash
#SBATCH --job-name=lindemann
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/logi/%x_%j.out
# Kryterium Zahna–Mareta (kod/lindemann_wzgledny.py): eksport 60 epok + dwa gęste okna RP128 (CPU).
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_koloidalne"
POK="$B/smallvit_runs/2026-10-10_szklo_pokretla/wyniki"
PY="$B/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/.venv/bin/python"
OUT="$EXP/wyniki/lindemann"
SCRATCH="/dev/shm/${USER}_lind_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH" "$OUT" "$EXP/logi"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONUNBUFFERED=1 TMPDIR="$SCRATCH" MPLCONFIGDIR="$SCRATCH" XDG_CACHE_HOME="$SCRATCH"
cd "$EXP/kod"
echo "start $(date), węzeł $(hostname)"
"$PY" -u -B lindemann_wzgledny.py gesty "$POK/lr_5e-4_rp/gesty" 300 2188 12 "$OUT" lr_5e-4_rp_300_2188
"$PY" -u -B lindemann_wzgledny.py gesty "$POK/dlugi_lr_5e-4_rp/gesty" 12034 13128 12 "$OUT" dlugi_lr_5e-4_rp_12034_13128
"$PY" -u -B lindemann_wzgledny.py ruch60 "$EXP/wyniki/ruch60_rp128_50tr_50val" "$OUT"
echo "KONIEC $(date)"
