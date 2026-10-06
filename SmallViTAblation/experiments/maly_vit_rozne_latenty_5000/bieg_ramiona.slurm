#!/bin/bash
#SBATCH --job-name=vit_latenty
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@180
#SBATCH --array=0-6%1
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_maly_vit_rozne_latenty_5000/logi/%x_%A_%a.out
# Małe ViT o różnej wielkości latentu, ten sam przepis i zbiór 5000 klatek (PLAN.md tego katalogu).
# Zadanie tablicowe: indeks 0..6 = ramię (kolejność RAMIONA niżej), %1 = jedno ramię naraz.
#   próba kodu (20 kroków każde ramię):  sbatch bieg_ramiona.slurm proba
#   bieg właściwy (5000 kroków):          sbatch bieg_ramiona.slurm pelny
#   jedno ramię:                          sbatch --array=3 bieg_ramiona.slurm pelny      (3 = tiny)
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
TRYB=${1:?proba albo pelny}
RAMIONA=(femto pico nano tiny small tiny_tl8 nano_tl8)
RAMIE=${RAMIONA[$SLURM_ARRAY_TASK_ID]}
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-06_maly_vit_rozne_latenty_5000"
STARA="$B/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia"
OD_ZERA="$B/smallvit_runs/2026-09-28_maly_vit_od_zera"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
OUT="$EXP/wyniki/${TRYB}_${RAMIE}"
for f in "$OD_ZERA/deps/timm/__init__.py" "$OD_ZERA/wejscia/pretrain/train/_annotations.coco.json" \
         "$OD_ZERA/wejscia/pretrain/valid/_annotations.coco.json" "$OD_ZERA/wejscia/index/manifest.jsonl" \
         "$OD_ZERA/wejscia/wykluczone.json" "$STARA/wejscia/obrazy.npy" "$EXP/kod/bieg.py" "$EXP/kod/model.py" \
         "$EXP/kod/dane.py" "$EXP/kod/trening.py" "$ARCHIVE" "$PY"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
[ -e "$OUT/MODEL_${RAMIE}.pt" ] && { echo "ISTNIEJE $OUT/MODEL_${RAMIE}.pt - nie nadpisuję"; exit 4; }

SCRATCH="/dev/shm/${USER}_vlatenty_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"; touch "$SCRATCH/.start"
PID=""
sprzataj() {
  set +e
  [ -n "$PID" ] && kill "$PID" 2>/dev/null && wait "$PID" 2>/dev/null
  echo "pliki w /tmp powstałe w czasie zadania (puste = czysto):"
  find /tmp -maxdepth 1 -user "$(id -u)" -newer "$SCRATCH/.start" 2>/dev/null
  case "$SCRATCH" in /dev/shm/${USER}_vlatenty_*) rm -rf -- "$SCRATCH";; esac
  echo "moje wpisy w /dev/shm po sprzątaniu (puste = czysto):"
  find /dev/shm -maxdepth 1 -user "$(id -u)" 2>/dev/null
  echo "df /dev/shm: $(df -h /dev/shm | tail -1)"
}
trap sprzataj EXIT
trap 'echo "SYGNAŁ - przerywam"; exit 143' INT TERM
uruchom() { "$@" & PID=$!; wait "$PID"; local r=$?; PID=""; return $r; }

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$OD_ZERA/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache"
export CZASTKI_WEJSCIA="$STARA/wejscia"
export PYTHONPATH="$OD_ZERA/deps"   # timm 0.9.16 (brak w venv Edenu)
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/mpl" TORCH_HOME="$SCRATCH/torch" XDG_CACHE_HOME="$SCRATCH/xdg"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$TMPDIR" "$OUT"
echo "tryb $TRYB, ramię $RAMIE (zadanie $SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID), węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1 || true
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
[ -f "$SCRATCH/data/pool_manifest.jsonl" ] && [ -d "$SCRATCH/data/images" ] || { echo "złe archiwum"; exit 3; }

cd "$EXP/kod"
if [ "$TRYB" = proba ]; then
  uruchom "$PY" -u -B bieg.py --ramie "$RAMIE" --wyniki "$OUT" --proba
else
  uruchom "$PY" -u -B bieg.py --ramie "$RAMIE" --wyniki "$OUT" --krokow 5000 --zapis-wag 500 --co-ocena 500
fi
du -sh "$OUT" "$OUT"/* 2>/dev/null || true
lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1 || true
echo "KONIEC $TRYB $RAMIE $(date)"
