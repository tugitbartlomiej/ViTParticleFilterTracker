#!/bin/bash
#SBATCH --job-name=dyfuzja_gen
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@60
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-29_dyfuzja_narzedzie/logi/%x_%j.out
# Generowanie na Edenie (dyfuzja_narzedzie/PLAN.md): wycinki od szumu prowadzone siłą tożsamości małego ViT z pętli
# wersji 3 (wagi z rozgrzewki k02.pt danego ziarna, wektor `nowy` z WEKTORY.npz).
#   sbatch generuj_eden.slurm <ziarno 47-56> <zmiana siły> [n] [waga] [model dyfuzji]
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
S=${1:?ziarno}; ZMIANA=${2:?zmiana siły}; N=${3:-16}; WAGA=${4:-0.3}
B=/home2/faculty/bpiotrowski
MY="$B/smallvit_runs/2026-09-29_dyfuzja_narzedzie"
MODEL=${5:-$MY/wyniki/trening/model_ema.pt}
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
V1="$B/smallvit_runs/2026-09-28_maly_vit_od_zera"
VIT="$B/smallvit_runs/2026-09-28_maly_vit_sila/wyniki/petla_s$S/wspolny/k02.pt"
PROBE="$V1/wejscia/probe"
for f in "$MODEL" "$VIT" "$MY/wejscia/WEKTORY.npz" "$PROBE/images/00_original.png" "$PROBE/masks/00_original.png" \
         "$V1/deps/timm/__init__.py" "$MY/kod/generuj.py"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
SCRATCH=$(mktemp -d "/dev/shm/dyfgen-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/dyfgen-*) rm -rf -- "$SCRATCH";; esac; }
trap cleanup EXIT INT TERM
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 KMP_DUPLICATE_LIB_OK=TRUE
export TMPDIR="$SCRATCH" TORCH_HOME="$SCRATCH/torch" MPLCONFIGDIR="$SCRATCH/mpl" XDG_CACHE_HOME="$SCRATCH/xdg"
export PYTHONPATH="$V1/deps"                 # timm dla definicji małego ViT (brak w venv Edenu)
OUT="$MY/wyniki/gen_s${S}_z${ZMIANA}_w${WAGA}_${SLURM_JOB_ID}"
cd "$MY/kod"
"$PY" -u -B generuj.py --dyfuzja "$MODEL" --vit "$VIT" --wektory "$MY/wejscia/WEKTORY.npz" --klucz "s${S}_nowy" \
  --kotwica "$PROBE/images/00_original.png" --kotwica-maska "$PROBE/masks/00_original.png" --zmiana "$ZMIANA" \
  -n "$N" --paczka 8 --waga "$WAGA" --out "$OUT"
echo "KONIEC $(date) -> $OUT"
