#!/bin/bash
#SBATCH --job-name=markow
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/logi/%x_%j.out
# Test skali kroku (kod/markow_test_eden.py) na gęstym zapisie RP128 biegów pokręteł: trzy okna po kolei (CPU, bez GPU).
#   sbatch markow_test.slurm
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_koloidalne"
POK="$B/smallvit_runs/2026-10-10_szklo_pokretla/wyniki"
PY="$B/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/.venv/bin/python"
OUT="$EXP/wyniki/markow"
SCRATCH="/dev/shm/${USER}_markow_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH" "$OUT" "$EXP/logi"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH" MPLCONFIGDIR="$SCRATCH" XDG_CACHE_HOME="$SCRATCH"
"$PY" -c "import deeptime, pydmd, scipy, matplotlib; print('deeptime', deeptime.__version__)"
cd "$EXP/kod"
echo "start $(date), węzeł $(hostname)"
"$PY" -u -B markow_test_eden.py "$POK/lr_5e-4_rp/gesty"       300  1094 12 "$OUT" lr_5e-4_rp_300_1094
"$PY" -u -B markow_test_eden.py "$POK/lr_5e-4_rp/gesty"       300  2188 12 "$OUT" lr_5e-4_rp_300_2188
"$PY" -u -B markow_test_eden.py "$POK/dlugi_lr_5e-4_rp/gesty" 12034 13128 12 "$OUT" dlugi_lr_5e-4_rp_12034_13128
echo "KONIEC $(date)"
