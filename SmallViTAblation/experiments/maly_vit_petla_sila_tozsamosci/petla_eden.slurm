#!/bin/bash
#SBATCH --job-name=malyvit3
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-28_maly_vit_sila/logi/%x_%A_%a.out
# Pętla małego ViT, wersja 3 (maly_vit_petla_sila_tozsamosci/PLAN.md): prognoza siły tożsamości, FastGAN wzdłuż wektora.
# Jedno ziarno na zadanie tablicy, ziarno = $1 + indeks tablicy. Wejścia i timm z katalogu wersji 1, rozgrzewka
# (kroki 0–1000) z wersji 2 — twarde dowiązania, bez kopiowania bajtów.
# Próba kodu (krótkie obroty w ramionach):  sbatch --array=0-0 petla_eden.slurm 47 proba
# Kampania (ziarna 47..56):                  sbatch --array=0-9 petla_eden.slurm 47
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
BAZA=${1:?pierwsze ziarno}; TRYB=${2:-pelny}
S=$((BAZA + SLURM_ARRAY_TASK_ID))
B=/home2/faculty/bpiotrowski
MY="$B/smallvit_runs/2026-09-28_maly_vit_sila"
V1="$B/smallvit_runs/2026-09-28_maly_vit_od_zera"
V2W="$B/smallvit_runs/2026-09-28_maly_vit_wektor/wyniki/petla_s$S/wspolny"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
for f in "$V1/deps/timm/__init__.py" "$MY/kod/petla.py" "$MY/kod/sila.py" "$V1/wejscia/index/features.npy" \
         "$V1/wejscia/fastgan/checkpoint_050000.pth" "$V1/wejscia/fastgan/models.py" "$V1/wejscia/dinov2_vits14_pretrain.pth" \
         "$V1/wejscia/probe/manifest.json" "$ARCHIVE" "$V2W/k02.pt" "$V2W/k01/LATENT.npz" "$V2W/k02/LATENT.npz" "$V2W/OCENY.json"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
D="$MY/wyniki/petla_s$S"; [ "$TRYB" = proba ] && D="$MY/wyniki/proba_s$S"
if [ ! -f "$D/wspolny/k02.pt" ]; then mkdir -p "$D" && cp -al "$V2W" "$D/"; fi
SCRATCH=$(mktemp -d "/dev/shm/malyvit3-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/malyvit3-*) rm -rf -- "$SCRATCH";; *) exit 91;; esac; }
trap cleanup EXIT INT TERM
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$V1/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache" MALY_VIT_WYNIKI="$MY/wyniki"
export PYTHONPATH="$V1/deps"   # timm 0.9.16 (brak w venv Edenu; kopia z lokalnego środowiska)
export MPLCONFIGDIR="$SCRATCH/mpl" TORCH_HOME="$SCRATCH/torch" XDG_CACHE_HOME="$SCRATCH/xdg"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
[ -f "$SCRATCH/data/pool_manifest.jsonl" ] && [ -d "$SCRATCH/data/images" ] || { echo "złe archiwum"; exit 3; }
echo "ziarno $S, tryb $TRYB, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
ARG=""; [ "$TRYB" = proba ] && ARG="--proba"
cd "$MY/kod"
"$PY" -u -B petla.py --ziarno "$S" $ARG
echo "KONIEC ziarno $S $(date)"
