#!/bin/bash
#SBATCH --job-name=gesta_paczka
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@180
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-03_maly_vit_gesta_paczka/logi/%x_%j.out
# Gęsta historia tokenów małego ViT przy WIĘKSZEJ paczce (PLAN.md tego katalogu). Ta sama recepta i ziarno 47 co
# bieg 1818732, zmienia się tylko rozmiar paczki. Zapis co krok tylko 60 wybranych tokenów na obraz (wszystkie
# narzędzia plus losowe tło), żeby bieg zajmował ~2,2 GB zamiast 22 GB.
#   próba kodu (20 kroków, paczka 320):  sbatch bieg_paczka.slurm proba 320
#   bieg właściwy:                        sbatch bieg_paczka.slurm pelny 320   (albo 128)
#   trzeci argument = beta1 AdamW (domyślnie 0.9), np.: sbatch bieg_paczka.slurm pelny 32 0.99
#   (wyniki biegów 1820333/1820334 leżą w wyniki/pelny_p320 i pelny_p128 bez przyrostka _b0.9: wcześniejsza wersja)
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
TRYB=${1:?proba albo pelny}
PACZKA=${2:?rozmiar paczki, np. 320}
BETA1=${3:-0.9}
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-03_maly_vit_gesta_paczka"
STARA="$B/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia"
OD_ZERA="$B/smallvit_runs/2026-09-28_maly_vit_od_zera"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
OUT="$EXP/wyniki/${TRYB}_p${PACZKA}_b${BETA1}"
for f in "$OD_ZERA/deps/timm/__init__.py" "$OD_ZERA/wejscia/pretrain/train/_annotations.coco.json" \
         "$OD_ZERA/wejscia/pretrain/valid/_annotations.coco.json" "$OD_ZERA/wejscia/probe/manifest.json" \
         "$EXP/kod/bieg.py" "$STARA/wejscia/obrazy.npy" "$STARA/wejscia/maski.npz" "$STARA/wejscia/images.json" "$ARCHIVE"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
[ -e "$OUT/dane/LATENT_GESTY.npy" ] && { echo "ISTNIEJE $OUT/dane/LATENT_GESTY.npy - nie nadpisuję"; exit 4; }

SCRATCH="/dev/shm/${USER}_gpaczka_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"; touch "$SCRATCH/.start"
PID=""
sprzataj() {
  set +e
  [ -n "$PID" ] && kill "$PID" 2>/dev/null && wait "$PID" 2>/dev/null
  echo "pliki w /tmp powstałe w czasie zadania (puste = czysto):"
  find /tmp -maxdepth 1 -user "$(id -u)" -newer "$SCRATCH/.start" 2>/dev/null
  case "$SCRATCH" in /dev/shm/${USER}_gpaczka_*) rm -rf -- "$SCRATCH";; esac
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
echo "tryb $TRYB, paczka $PACZKA, beta1 $BETA1, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1 || true
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
[ -f "$SCRATCH/data/pool_manifest.jsonl" ] && [ -d "$SCRATCH/data/images" ] || { echo "złe archiwum"; exit 3; }

cd "$EXP/kod"
if [ "$TRYB" = proba ]; then
  uruchom "$PY" -u -B bieg.py --wyniki "$OUT" --proba --paczka "$PACZKA" --beta1 "$BETA1" --tokeny-na-obraz 60
else
  uruchom "$PY" -u -B bieg.py --wyniki "$OUT" --paczka "$PACZKA" --beta1 "$BETA1" --tokeny-na-obraz 60 --zapis-wag 100 --zapis-pelny 500
fi
du -sh "$OUT" "$OUT"/* 2>/dev/null || true
lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1 || true
echo "KONIEC $TRYB p$PACZKA $(date)"
