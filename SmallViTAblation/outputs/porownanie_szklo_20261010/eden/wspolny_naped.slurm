#!/bin/bash
#SBATCH --job-name=naped
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/logi/%x_%j.out
# Odjęcie wspólnego napędu (CPU): skala kroku z czynników pola innych zdjęć (kod/wspolny_naped_krok.py, venv z deeptime) na dwóch
# oknach gęstych, potem skala checkpointów z prawdziwych przyrostów wag (kod/wspolny_naped_wagi.py, venv z torch).
#   sbatch wspolny_naped.slurm
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_koloidalne"
POK="$B/smallvit_runs/2026-10-10_szklo_pokretla/wyniki"
E35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
PYD="$B/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/.venv/bin/python"
PYT="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
OUT="$EXP/wyniki/naped"
SCRATCH="/dev/shm/${USER}_naped_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH" "$OUT" "$EXP/logi"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH" MPLCONFIGDIR="$SCRATCH" XDG_CACHE_HOME="$SCRATCH"
cd "$EXP/kod"
echo "start $(date), węzeł $(hostname)"
"$PYD" -u -B wspolny_naped_krok.py "$POK/lr_5e-4_rp/gesty"       300  2188 12 "$OUT" lr_5e-4_rp_300_2188
"$PYD" -u -B wspolny_naped_krok.py "$POK/dlugi_lr_5e-4_rp/gesty" 12034 13128 12 "$OUT" dlugi_lr_5e-4_rp_12034_13128
"$PYT" -u -B wspolny_naped_wagi.py "$E35/wyniki/pelny/wagi" "$EXP/wyniki/ruch60_rp128_50tr_50val" "$OUT"
echo "KONIEC $(date)"
