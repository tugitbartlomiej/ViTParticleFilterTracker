#!/bin/bash
#SBATCH --job-name=k1000_pod150z
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@180
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_k1000_podmiana150_sd/logi/%x_%j.out
# Rozszerzenie (PLAN.md, 2026-10-08): ramiona C0 / A150 / R150 (150 losowych klatek) × ziarna 47, 48, 49,
# 5 epok po 47 kroków, od K1000 z AdamW. Wyniki: wyniki/ziarna/<ramie>_s<ziarno>/.
#   sbatch --export=NIL bieg_ziarna.slurm ziarna
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
USER=${USER:-$(id -un)}; export USER
TRYB=${1:?ziarna}
[ "$TRYB" = ziarna ] || { echo "tylko tryb ziarna"; exit 2; }
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-08_k1000_podmiana150_sd"
OD_ZERA="$B/smallvit_runs/2026-09-28_maly_vit_od_zera"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
WEJ="$EXP/wejscia/wejscia_podmiana"
LOSW="$EXP/wejscia/wejscia_losowe"
OUT="$EXP/wyniki/$TRYB"
for f in "$EXP/deps/timm/__init__.py" "$EXP/wejscia/pretrain/train/_annotations.coco.json" \
         "$EXP/wejscia/pretrain/valid/_annotations.coco.json" \
         "$EXP/kod/bieg.py" "$WEJ/NOWE.json" "$WEJ/CEL.json" "$WEJ/PODMIANA.json" "$WEJ/MODEL_VIT_K1000.pt" "$LOSW/LOSOWE.json" \
         "$EXP/wejscia/k1000_pelny.pt" "$ARCHIVE"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done

SCRATCH="/dev/shm/${USER}_pod150_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"; touch "$SCRATCH/.start"
PID=""
sprzataj() {
  set +e
  [ -n "$PID" ] && kill "$PID" 2>/dev/null && wait "$PID" 2>/dev/null
  echo "pliki w /tmp powstałe w czasie zadania (puste = czysto):"
  find /tmp -maxdepth 1 -user "$(id -u)" -newer "$SCRATCH/.start" 2>/dev/null
  case "$SCRATCH" in /dev/shm/${USER}_pod150_*) rm -rf -- "$SCRATCH";; esac
  echo "moje wpisy w /dev/shm po sprzątaniu (puste = czysto):"
  find /dev/shm -maxdepth 1 -user "$(id -u)" 2>/dev/null
  echo "df /dev/shm: $(df -h /dev/shm | tail -1)"
}
trap sprzataj EXIT
trap 'echo "SYGNAŁ - przerywam"; exit 143' INT TERM
uruchom() { "$@" & PID=$!; wait "$PID"; local r=$?; PID=""; return $r; }

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Adnotacje COCO bazy i walidacji: własne kopie (SHA256 zgodne z od_zera/eden/WEJSCIA_SHA256.json); kopia valid
# w od_zera/wejscia zniknęła przy sprzątaniu 2026-10-08.
export MALY_VIT_WEJSCIA="$EXP/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache"
export PODMIANA_WEJSCIA="$WEJ" LOSOWE_WEJSCIA="$LOSW" K1000_PELNY="$EXP/wejscia/k1000_pelny.pt"
export PYTHONPATH="$EXP/deps"   # timm 0.9.16 (brak w venv Edenu; kopia w od_zera/deps wyczyszczona 2026-10-08, więc własna kopia z F:\Python)
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/mpl" TORCH_HOME="$SCRATCH/torch" XDG_CACHE_HOME="$SCRATCH/xdg"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$TMPDIR" "$OUT"
echo "tryb $TRYB, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
echo "df /dev/shm: $(df -h /dev/shm | tail -1)"
lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1 || true
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
[ -f "$SCRATCH/data/pool_manifest.jsonl" ] && [ -d "$SCRATCH/data/images" ] || { echo "złe archiwum"; exit 3; }

cd "$EXP/kod"
for Z in 47 48 49; do
  for R in C0 A150 R150; do
    uruchom "$PY" -u -B bieg.py --wyniki "$OUT" --ramie "$R" --ziarno "$Z"
  done
done
du -sh "$OUT" "$OUT"/* 2>/dev/null || true
lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1 || true
echo "KONIEC $TRYB $(date)"
