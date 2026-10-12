#!/bin/bash
#SBATCH --job-name=czastki_gesto
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@180
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/logi/%x_%j.out
# Gęsta historia surowych tokenów małego ViT (README.md tego katalogu). Uczenie 1000 kroków od zera, ziarno 47,
# tokeny 12 pełnych klatek po każdym kroku, potem analiza kierunków ruchu (co 10, co 1, kroki 60 i 120).
#   próba kodu (20 kroków):  sbatch bieg_eden.slurm proba
#   bieg właściwy:           sbatch bieg_eden.slurm pelny
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
TRYB=${1:?proba albo pelny}
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia"
OD_ZERA="$B/smallvit_runs/2026-09-28_maly_vit_od_zera"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
OUT="$EXP/wyniki/$TRYB"
for f in "$OD_ZERA/deps/timm/__init__.py" "$OD_ZERA/wejscia/pretrain/train/_annotations.coco.json" \
         "$OD_ZERA/wejscia/pretrain/valid/_annotations.coco.json" "$OD_ZERA/wejscia/probe/manifest.json" \
         "$EXP/kod/bieg.py" "$EXP/kod/analiza.py" "$EXP/wejscia/obrazy.npy" "$EXP/wejscia/maski.npz" "$ARCHIVE"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
[ -e "$OUT/dane/LATENT_GESTY.npy" ] && { echo "ISTNIEJE $OUT/dane/LATENT_GESTY.npy - nie nadpisuję"; exit 4; }

SCRATCH="/dev/shm/${USER}_czastki_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"; touch "$SCRATCH/.start"
PID=""
sprzataj() {
  set +e
  [ -n "$PID" ] && kill "$PID" 2>/dev/null && wait "$PID" 2>/dev/null
  # TMPDIR procesu wskazuje do $SCRATCH, więc w /tmp nic nie powinno powstać. Niczego w /tmp nie kasujemy:
  # na tym samym węźle mogą biec inne zadania tego konta i ich pliki też są „moje i młodsze niż start”.
  echo "pliki w /tmp powstałe w czasie zadania (puste = czysto):"
  find /tmp -maxdepth 1 -user "$(id -u)" -newer "$SCRATCH/.start" 2>/dev/null
  case "$SCRATCH" in /dev/shm/${USER}_czastki_*) rm -rf -- "$SCRATCH";; esac
  echo "moje wpisy w /dev/shm po sprzątaniu (puste = czysto):"
  find /dev/shm -maxdepth 1 -user "$(id -u)" 2>/dev/null
  echo "df /dev/shm: $(df -h /dev/shm | tail -1)"
}
trap sprzataj EXIT
trap 'echo "SYGNAŁ - przerywam"; exit 143' INT TERM
uruchom() { "$@" & PID=$!; wait "$PID"; local r=$?; PID=""; return $r; }

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$OD_ZERA/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache"
export CZASTKI_WEJSCIA="$EXP/wejscia"
export PYTHONPATH="$OD_ZERA/deps"   # timm 0.9.16 (brak w venv Edenu)
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/mpl" TORCH_HOME="$SCRATCH/torch" XDG_CACHE_HOME="$SCRATCH/xdg"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$TMPDIR" "$OUT"
echo "tryb $TRYB, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
echo "df /dev/shm: $(df -h /dev/shm | tail -1)"
lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1 || true
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
[ -f "$SCRATCH/data/pool_manifest.jsonl" ] && [ -d "$SCRATCH/data/images" ] || { echo "złe archiwum"; exit 3; }

cd "$EXP/kod"
if [ "$TRYB" = proba ]; then
  uruchom "$PY" -u -B bieg.py --wyniki "$OUT" --proba
  uruchom "$PY" -u -B analiza.py --dane "$OUT/dane" --wyniki "$OUT/analiza_co1" --co 1
else
  uruchom "$PY" -u -B bieg.py --wyniki "$OUT"
  uruchom "$PY" -u -B analiza.py --dane "$OUT/dane" --wyniki "$OUT/analiza_co10" --co 10
  uruchom "$PY" -u -B analiza.py --dane "$OUT/dane" --wyniki "$OUT/analiza_kroki_60_120" --kroki 60,120
  uruchom "$PY" -u -B analiza.py --dane "$OUT/dane" --wyniki "$OUT/analiza_co1" --co 1
fi
du -sh "$OUT" "$OUT"/* 2>/dev/null || true
lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1 || true
echo "KONIEC $TRYB $(date)"
