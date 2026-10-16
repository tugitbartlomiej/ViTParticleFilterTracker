#!/bin/bash
#SBATCH --job-name=pomiar_vit
#SBATCH --account=transformers_vsc
#SBATCH --partition=short,hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/logi/%x_%j.out
# Pomiar naturalny trajektorii tokenów wielu zdjęć po checkpointach biegu 35 000 (format osi czasu aplikacji),
# potem analiza uczące/walidacyjne (analiza_train_val.py) na Edenie, żeby nie ściągać pełnych klatek.
#   sbatch pomiar_eden.slurm 3000 1000                 # do końca 21. epoki (krok 22 974), ~340 checkpointów
#   sbatch pomiar_eden.slurm 3000 1000 pelny60 65640   # wszystkie 806 wag do końca 60. epoki
#   sbatch pomiar_eden.slurm 200 100 proba             # próba kodu
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
NTR=${1:-3000}; NVAL=${2:-1000}; TRYB=${3:-pelny}; MAXK=${4:-22974}
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_koloidalne"
EXP35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
ZIPY="$B/smallvit_data/cataract_lmm/3_Object_Tracking/annotations"
OUT="$EXP/wyniki/pomiar_${TRYB}_${NTR}tr_${NVAL}val"
for f in "$EXP35/kod/trening35000.py" "$EXP35/deps/timm/__init__.py" "$EXP35/wejscia/MANIFEST_35000.jsonl" "$EXP35/wyniki/pelny/wagi/k00000.pt" \
         "$ARCHIVE" "$ZIPY/TR_0001_S1_P03.zip" "$EXP/kod/pomiar_eden.py" "$EXP/kod/analiza_train_val.py" "$EXP/kod/odciski.py"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
SCRATCH="/dev/shm/${USER}_pomiar_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$SCRATCH/tmp" "$EXP/logi" "$EXP/wyniki"
trap 'rm -rf -- "$SCRATCH"; du -sh "$OUT" 2>/dev/null; lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1' EXIT
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$EXP35/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache"
export PYTHONPATH="$EXP35/deps"
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/tmp" TORCH_HOME="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/tmp"
echo "pomiar $NTR+$NVAL ($TRYB, do kroku $MAXK), węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"        # klatki walidacji (ten sam podział co bieg)
cp "$EXP/kod/pomiar_eden.py" "$EXP35/kod/pomiar_eden.py"
cd "$EXP35/kod"
EXTRA=""; [ "$TRYB" = proba ] && EXTRA="--pca32_n 10"
"$PY" -u -B pomiar_eden.py --wagi "$EXP35/wyniki/pelny/wagi" --manifest "$EXP35/wejscia/MANIFEST_35000.jsonl" --zipy "$ZIPY" \
      --out "$OUT" --n_train "$NTR" --n_val "$NVAL" --max_krok "$MAXK" $EXTRA
echo "POMIAR GOTOWY $(date)"
# analiza uczące/walidacyjne na Edenie: TRAIN_VAL_<tryb>.json + train_val_<tryb>.png obok wyników pomiaru
cd "$EXP/kod"
"$PY" -u -B analiza_train_val.py "$OUT" "_$TRYB" "$OUT" || echo "ANALIZA NIEUDANA (pomiar zostaje)"
echo "KONIEC $(date)"
