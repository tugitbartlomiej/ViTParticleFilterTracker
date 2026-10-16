#!/bin/bash
#SBATCH --job-name=vit35k_21ep
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@300
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok/logi/%x_%j.out
# Mały ViT od zera na 35 000 klatkach, 21 epok, wagi po każdej epoce i gęste kroki (README.md eksperymentu).
#   próba kodu (512 klatek, 2 epoki):  sbatch bieg_eden.slurm proba
#   bieg właściwy:                     sbatch bieg_eden.slurm pelny
#   wznowienie albo dalsze epoki:      sbatch bieg_eden.slurm wznow [łączna liczba epok, domyślnie 21]
#   gęsty zapis ruchu sond i sił:      sbatch bieg_eden.slurm gesty_proba  |  sbatch bieg_eden.slurm gesty [epoki]
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
TRYB=${1:?proba, pelny, wznow, gesty_proba albo gesty}
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
ZIPY="$B/smallvit_data/cataract_lmm/3_Object_Tracking/annotations"
case "$TRYB" in proba) OUT="$EXP/wyniki/proba"; ARG="--proba";; pelny) OUT="$EXP/wyniki/pelny"; ARG="";;
  wznow) OUT="$EXP/wyniki/pelny"; ARG="--wznow --epoki ${2:-21}";;
  gesty_proba) OUT="$EXP/wyniki/gesty_proba"; ARG="--proba --obserwuj";;
  gesty) OUT="$EXP/wyniki/gesty"; ARG="--obserwuj --sondy $EXP/wejscia/SONDY_100.jsonl --epoki ${2:-21}";;
  *) echo "zły tryb $TRYB"; exit 2;; esac
for f in "$EXP/deps/timm/__init__.py" "$EXP/wejscia/pretrain/valid/_annotations.coco.json" \
         "$EXP/kod/trening35000.py" "$EXP/wejscia/MANIFEST_35000.jsonl" "$ARCHIVE" "$ZIPY/TR_0001_S1_P03.zip"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
[ "$TRYB" = pelny ] && [ -e "$OUT/wagi/k00000.pt" ] && { echo "ISTNIEJE $OUT/wagi - użyj trybu wznow"; exit 4; }
[ "$TRYB" = gesty ] && [ -e "$OUT/gesty/BAZA_PCA32.npz" ] && { echo "ISTNIEJE $OUT/gesty - nie nadpisuję"; exit 4; }

SCRATCH="/dev/shm/${USER}_vit35k_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"; touch "$SCRATCH/.start"
PID=""
sprzataj() {
  set +e
  [ -n "$PID" ] && kill "$PID" 2>/dev/null && wait "$PID" 2>/dev/null
  echo "pliki w /tmp powstałe w czasie zadania (puste = czysto):"
  find /tmp -maxdepth 1 -user "$(id -u)" -newer "$SCRATCH/.start" 2>/dev/null
  case "$SCRATCH" in /dev/shm/${USER}_vit35k_*) rm -rf -- "$SCRATCH";; esac
  echo "moje wpisy w /dev/shm po sprzątaniu (puste = czysto):"
  find /dev/shm -maxdepth 1 -user "$(id -u)" 2>/dev/null
  du -sh "$OUT" "$OUT"/* 2>/dev/null
  lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1
}
trap sprzataj EXIT
trap 'echo "SYGNAŁ - przerywam (stan epok zostaje w pelny/)"; exit 143' INT TERM
uruchom() { "$@" & PID=$!; wait "$PID"; local r=$?; PID=""; return $r; }

export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$EXP/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache"
export PYTHONPATH="$EXP/deps"   # timm 0.9.16, kopia z 2026-10-08_k1000_podmiana150_sd/deps (brak w venv Edenu)
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/mpl" TORCH_HOME="$SCRATCH/torch" XDG_CACHE_HOME="$SCRATCH/xdg"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$TMPDIR" "$OUT"
echo "tryb $TRYB, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
echo "df /dev/shm: $(df -h /dev/shm | tail -1)"
lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1 || true
tar -xf "$ARCHIVE" -C "$SCRATCH/data"        # klatki walidacji (ten sam podział co bieg K1000)
cd "$EXP/kod"
uruchom "$PY" -u -B trening35000.py --wyniki "$OUT" --zipy "$ZIPY" --manifest "$EXP/wejscia/MANIFEST_35000.jsonl" $ARG
echo "KONIEC $TRYB $(date)"
