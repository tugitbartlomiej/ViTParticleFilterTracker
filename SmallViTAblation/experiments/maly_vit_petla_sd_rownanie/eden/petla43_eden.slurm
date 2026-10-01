#!/bin/bash
#SBATCH --job-name=petla43
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_petla43_sd_prowadzenie/logi/%x_%A_%a.out
# Pętla małego ViT, wersja 4.3 (docs/26_PIPELINE_TOZSAMOSC_PROGNOZA_SD.md): bliźniak wersji 3 z Langevinem z jądrem
# pamięci i Stable Diffusion prowadzonym latentem tożsamości. 14 obrotów × 500 kroków, 134 klatki na obrót, listy 180.
# Ziarno = $1 + indeks tablicy. Rozgrzewka (21 epok = 1000 kroków) z wersji 2 o tym samym ziarnie (ziarna 47–56,
# twarde dowiązania), jak w wersji 3; inne ziarna liczą ją w zadaniu.
# Próba kodu:  sbatch --array=0-0 petla43_eden.slurm 47 proba
# Kampania:    sbatch --array=0-9%5 petla43_eden.slurm 47   (wyniki w petla43b_s*; petla43_s47 to przerwany pierwszy bieg)
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
BAZA=${1:?pierwsze ziarno}; TRYB=${2:-pelny}
S=$((BAZA + SLURM_ARRAY_TASK_ID))
B=/home2/faculty/bpiotrowski
MY="$B/smallvit_runs/2026-10-01_petla43_sd_prowadzenie"
V1="$B/smallvit_runs/2026-09-28_maly_vit_od_zera"
V2W="$B/smallvit_runs/2026-09-28_maly_vit_wektor/wyniki/petla_s$S/wspolny"
SD="$B/smallvit_runs/2026-09-30_dyfuzja_sd_filmy"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
ADN="$B/smallvit_data/cataract_lmm/3_Object_Tracking/annotations"
for f in "$V1/deps/timm/__init__.py" "$SD/deps/diffusers/__init__.py" "$MY/kod/petla4.py" "$MY/kod/prowadzenie.py" \
         "$MY/kod/prognoza_gle.py" "$MY/wejscia/pusty_tekst.pt" "$V1/wejscia/index/features.npy" \
         "$V1/wejscia/dinov2_vits14_pretrain.pth" "$V1/wejscia/probe/manifest.json" "$V1/wejscia/wykluczone.json" \
         "$SD/wyniki/trening/ema_020000/config.json" "$SD/wyniki/trening/ema_020000/diffusion_pytorch_model.safetensors" \
         "$SD/modele/sd-vae-ft-mse/config.json" "$ADN/TR_0001_S1_P03.zip" "$ARCHIVE"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
D="$MY/wyniki/petla43b_s$S"; [ "$TRYB" = proba ] && D="$MY/wyniki/proba43_s$S"
if [ ! -f "$D/wspolny/k02.pt" ] && [ -f "$V2W/k02.pt" ] && [ -f "$V2W/k01/LATENT.npz" ] && [ -f "$V2W/k02/LATENT.npz" ] && [ -f "$V2W/OCENY.json" ]; then
  mkdir -p "$D" && cp -al "$V2W" "$D/" && echo "rozgrzewka z wersji 2: $V2W"
else
  echo "rozgrzewka: $([ -f "$D/wspolny/k02.pt" ] && echo istnieje || echo 'liczona w zadaniu')"
fi
SCRATCH=$(mktemp -d "/dev/shm/petla43-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/petla43-*) rm -rf -- "$SCRATCH";; *) exit 91;; esac; }
trap cleanup EXIT INT TERM
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$V1/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache" MALY_VIT_WYNIKI="$MY/wyniki"
export MALY_VIT_ADNOTACJE="$ADN" MALY_VIT_SD_UNET="$SD/wyniki/trening/ema_020000" MALY_VIT_SD_VAE="$SD/modele/sd-vae-ft-mse"
export MALY_VIT_SD_TEKST="$MY/wejscia/pusty_tekst.pt"
export PYTHONPATH="$V1/deps:$SD/deps"          # timm 0.9.16 i diffusers 0.39.0 (brak w venv Edenu)
export HF_HUB_OFFLINE=1 MPLCONFIGDIR="$SCRATCH/mpl" TORCH_HOME="$SCRATCH/torch" XDG_CACHE_HOME="$SCRATCH/xdg" HF_HOME="$SCRATCH/hf"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$MY/wyniki"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
[ -f "$SCRATCH/data/pool_manifest.jsonl" ] && [ -d "$SCRATCH/data/images" ] || { echo "złe archiwum"; exit 3; }
echo "ziarno $S, tryb $TRYB, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
ARG="--obroty 14 --top-k 180"; [ "$TRYB" = proba ] && ARG="--proba"
cd "$MY/kod"
"$PY" -u -B petla4.py --ziarno "$S" $ARG
echo "KONIEC ziarno $S $(date)"
