#!/bin/bash
#SBATCH --job-name=kontrola_siatki
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/logi/%x_%j.out
# Kontrole kryształu na gęstym zapisie RP128 jednego biegu pokręteł (CPU): siatka patchy, kryterium Zahna–Mareta, fonony.
#   sbatch [--dependency=afterok:<id treningu>] kontrole_siatki.slurm <bieg> [od do]
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
BIEG=${1:?bieg}; OD=${2:-300}; DO=${3:-2188}
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_koloidalne"
G="$B/smallvit_runs/2026-10-10_szklo_pokretla/wyniki/$BIEG/gesty"
PY="$B/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/.venv/bin/python"
OUT="$EXP/wyniki/kontrole_siatki"
SCRATCH="/dev/shm/${USER}_ks_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH" "$OUT" "$EXP/logi"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONUNBUFFERED=1 TMPDIR="$SCRATCH" MPLCONFIGDIR="$SCRATCH" XDG_CACHE_HOME="$SCRATCH"
ls "$G"/pca32_000.npz >/dev/null || { echo "BRAK gesty $G"; exit 2; }
cd "$EXP/kod"
echo "kontrole $BIEG okno $OD-$DO, start $(date), węzeł $(hostname)"
"$PY" -u -B siatka_gesty.py "$G" "$OUT" "$BIEG"
"$PY" -u -B lindemann_wzgledny.py gesty "$G" "$OD" "$DO" 12 "$OUT" "$BIEG"
"$PY" -u -B fonony.py gesty "$G" "$OD" "$DO" 12 "$OUT" "$BIEG"
echo "KONIEC $(date)"
