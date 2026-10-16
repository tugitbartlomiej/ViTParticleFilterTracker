#!/bin/bash
#SBATCH --job-name=naped_v2
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/logi/%x_%j.out
# Wspólny napęd z przyrostów wag, wersja 2: poprawione wyrównanie czasowe Δθ z ruchem tokenów + prognoza poza oknem.
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_koloidalne"
E35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
PYT="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
OUT="$EXP/wyniki/naped_v2"
SCRATCH="/dev/shm/${USER}_naped_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH" "$OUT"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH" MPLCONFIGDIR="$SCRATCH" XDG_CACHE_HOME="$SCRATCH"
cd "$EXP/kod"
echo "start $(date), węzeł $(hostname)"
"$PYT" -u -B wspolny_naped_wagi.py "$E35/wyniki/pelny/wagi" "$EXP/wyniki/ruch60_rp128_50tr_50val" "$OUT"
echo "KONIEC $(date)"
