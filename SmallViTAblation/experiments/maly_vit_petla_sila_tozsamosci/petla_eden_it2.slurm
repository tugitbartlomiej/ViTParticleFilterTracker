#!/bin/bash
#SBATCH --job-name=malyvit3b
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-29_maly_vit_sila_it2/logi/%x_%A_%a.out
# Pętla małego ViT, wersja 3, iteracja 2 (maly_vit_petla_sila_tozsamosci/PLAN.md, „Iteracja 2”): 14 obrotów, lista
# podobnych klatek 180, ziarna 47–66. Ziarno = $1 + indeks tablicy. Rozgrzewka z wersji 2, gdy istnieje (ziarna 47–56,
# twarde dowiązania); dla pozostałych liczona w zadaniu tym samym kodem.
# Próba kodu (nowe ziarno, więc sprawdza też liczenie rozgrzewki):  sbatch --array=0-0 petla_eden_it2.slurm 57 proba
# Kampania:                                                          sbatch --array=0-19 petla_eden_it2.slurm 47
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
BAZA=${1:?pierwsze ziarno}; TRYB=${2:-pelny}
S=$((BAZA + SLURM_ARRAY_TASK_ID))
B=/home2/faculty/bpiotrowski
MY="$B/smallvit_runs/2026-09-29_maly_vit_sila_it2"
V1="$B/smallvit_runs/2026-09-28_maly_vit_od_zera"
V2W="$B/smallvit_runs/2026-09-28_maly_vit_wektor/wyniki/petla_s$S/wspolny"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
for f in "$V1/deps/timm/__init__.py" "$MY/kod/petla.py" "$MY/kod/sila.py" "$V1/wejscia/index/features.npy" \
         "$V1/wejscia/fastgan/checkpoint_050000.pth" "$V1/wejscia/fastgan/models.py" "$V1/wejscia/dinov2_vits14_pretrain.pth" \
         "$V1/wejscia/probe/manifest.json" "$ARCHIVE"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
D="$MY/wyniki/petla_s$S"; [ "$TRYB" = proba ] && D="$MY/wyniki/proba_s$S"
if [ ! -f "$D/wspolny/k02.pt" ] && [ -f "$V2W/k02.pt" ] && [ -f "$V2W/k01/LATENT.npz" ] && [ -f "$V2W/k02/LATENT.npz" ] && [ -f "$V2W/OCENY.json" ]; then
  mkdir -p "$D" && cp -al "$V2W" "$D/" && echo "rozgrzewka z wersji 2: $V2W"
else
  echo "rozgrzewka: $([ -f "$D/wspolny/k02.pt" ] && echo istnieje || echo 'liczona w zadaniu')"
fi
SCRATCH=$(mktemp -d "/dev/shm/malyvit3b-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/malyvit3b-*) rm -rf -- "$SCRATCH";; *) exit 91;; esac; }
trap cleanup EXIT INT TERM
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$V1/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache" MALY_VIT_WYNIKI="$MY/wyniki"
export PYTHONPATH="$V1/deps"   # timm 0.9.16 (brak w venv Edenu; kopia z lokalnego środowiska)
export MPLCONFIGDIR="$SCRATCH/mpl" TORCH_HOME="$SCRATCH/torch" XDG_CACHE_HOME="$SCRATCH/xdg"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$MY/wyniki"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
[ -f "$SCRATCH/data/pool_manifest.jsonl" ] && [ -d "$SCRATCH/data/images" ] || { echo "złe archiwum"; exit 3; }
echo "ziarno $S, tryb $TRYB, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
ARG=""; [ "$TRYB" = proba ] && ARG="--proba"
cd "$MY/kod"
"$PY" -u -B petla.py --ziarno "$S" --obroty 14 --top-k 180 $ARG
echo "KONIEC ziarno $S $(date)"
