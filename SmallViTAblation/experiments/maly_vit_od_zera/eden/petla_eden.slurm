#!/bin/bash
#SBATCH --job-name=malyvit
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-28_maly_vit_od_zera/logi/%x_%A_%a.out
# Pętla małego ViT od zera (maly_vit_od_zera/PLAN.md): jedno ziarno na zadanie tablicy, ziarno = $1 + indeks tablicy.
# Kod: kopia z maly_vit_od_zera/eden/kod (ścieżki z MALY_VIT_*), wejścia: eden/wejscia (sumy w WEJSCIA_SHA256.json).
# Próba kodu (krótkie obroty):  sbatch --array=0-0 petla_eden.slurm 1 proba
# Kampania (ziarna 47..56):     sbatch --array=0-9 petla_eden.slurm 47
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
BAZA=${1:?pierwsze ziarno}; TRYB=${2:-pelny}
S=$((BAZA + SLURM_ARRAY_TASK_ID))
B=/home2/faculty/bpiotrowski
MY="$B/smallvit_runs/2026-09-28_maly_vit_od_zera"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
for f in "$MY/deps/timm/__init__.py" "$MY/kod/petla.py" "$MY/wejscia/index/features.npy" "$MY/wejscia/fastgan/checkpoint_050000.pth" \
         "$MY/wejscia/fastgan/models.py" "$MY/wejscia/dinov2_vits14_pretrain.pth" "$MY/wejscia/probe/manifest.json" "$ARCHIVE"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
SCRATCH=$(mktemp -d "/dev/shm/malyvit-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/malyvit-*) rm -rf -- "$SCRATCH";; *) exit 91;; esac; }
trap cleanup EXIT INT TERM
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$MY/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache" MALY_VIT_WYNIKI="$MY/wyniki"
export PYTHONPATH="$MY/deps"   # timm 0.9.16 (brak w venv Edenu; kopia z lokalnego środowiska)
export MPLCONFIGDIR="$SCRATCH/mpl" TORCH_HOME="$SCRATCH/torch" XDG_CACHE_HOME="$SCRATCH/xdg"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$MY/wyniki"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
[ -f "$SCRATCH/data/pool_manifest.jsonl" ] && [ -d "$SCRATCH/data/images" ] || { echo "złe archiwum"; exit 3; }
echo "ziarno $S, tryb $TRYB, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
ARG=""; [ "$TRYB" = proba ] && ARG="--proba"
cd "$MY/kod"
"$PY" -u -B petla.py --ziarno "$S" $ARG
echo "KONIEC ziarno $S $(date)"
