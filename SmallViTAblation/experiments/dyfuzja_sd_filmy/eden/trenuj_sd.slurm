#!/bin/bash
#SBATCH --job-name=sd_trening
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=23:50:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@900
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-30_dyfuzja_sd_filmy/logi/%x_%j.out
# Dostrajanie SD 1.5 na latentach klatek (kod/trenuj_sd.py), torchrun na wszystkich kartach zadania. Latenty kopiowane do
# /dev/shm węzła (szybki losowy odczyt), kasowane pułapką. 15 min przed końcem czasu skrypt tworzy STOP: Python kończy
# krok, zapisuje ostatni.pt i wychodzi; kolejne zgłoszenie z tymi samymi argumentami wznawia.
# Paczka globalna ~256: akumulacja = ceil(256 / (karty × na_kartę)).
#   sbatch [--gres=gpu:N] trenuj_sd.slurm <kroki> <na_kartę> [proba] [checkpointing]
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
KROKI=${1:?kroki}; NA=${2:?obrazów na kartę}; TRYB=${3:-pelny}; CKPT=${4:-}
B=/home2/faculty/bpiotrowski
MY=$B/smallvit_runs/2026-09-30_dyfuzja_sd_filmy
PY=$B/venvs/rfdetr_nano_clean_1_10_1/bin/python
if [ "$TRYB" = proba ]; then
  LAT=$B/smallvit_data/cataract_lmm/sd576_train_proba2; VAL=$B/smallvit_data/cataract_lmm/sd576_val_co10_proba2
  OUT=$MY/wyniki/proba; EXTRA="--zapis-co 100 --ema-co 100 --podglad-co 100 --log-co 10 --rozgrzewka 50"
else
  LAT=$B/smallvit_data/cataract_lmm/sd576_train; VAL=$B/smallvit_data/cataract_lmm/sd576_val_co10
  OUT=$MY/wyniki/trening; EXTRA=""
fi
for f in "$MY/kod/trenuj_sd.py" "$MY/kod/wspolne.py" "$MY/modele/sd15/unet/diffusion_pytorch_model.fp16.safetensors" \
         "$MY/modele/sd15/text_encoder/model.fp16.safetensors" "$MY/modele/sd-vae-ft-mse/config.json" "$MY/deps/diffusers/__init__.py"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
[ -d "$LAT/indeks" ] && [ -d "$VAL/indeks" ] || { echo "brak latentów $LAT lub $VAL"; exit 2; }
SCRATCH=$(mktemp -d "/dev/shm/sdtr-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/sdtr-*) rm -rf -- "$SCRATCH";; esac; }
trap cleanup EXIT INT
mkdir -p "$OUT"
przerwij() { echo "SIGTERM $(date): tworzę STOP"; touch "$OUT/STOP"; }
trap przerwij TERM
export TMPDIR="$SCRATCH" HF_HOME="$SCRATCH/hf" TORCH_HOME="$SCRATCH/torch" TRITON_CACHE_DIR="$SCRATCH/triton" \
       MPLCONFIGDIR="$SCRATCH/mpl" PYTHONPATH="$MY/deps" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN
N=$(nvidia-smi -L | wc -l)
AKUM=$(( (256 + N * NA - 1) / (N * NA) )); PACZKA=$(( AKUM * N * NA ))
echo "tryb $TRYB, $N × $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1), węzeł $(hostname), start $(date)"
echo "na kartę $NA, akumulacja $AKUM, paczka globalna $PACZKA, kroki $KROKI ${CKPT:+, checkpointing}"
lfs quota -p 1299 /mnt/evafs | tail -1
t0=$(date +%s)
cp -r "$LAT" "$SCRATCH/lat_train"; cp -r "$VAL" "$SCRATCH/lat_val"
echo "latenty w /dev/shm: $(du -sh $SCRATCH/lat_train | cut -f1) w $(( $(date +%s) - t0 )) s"
cd "$MY/kod"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$N" trenuj_sd.py --dane "$SCRATCH/lat_train" \
  --podglad-dane "$SCRATCH/lat_val" --sd "$MY/modele/sd15" --vae "$MY/modele/sd-vae-ft-mse" --out "$OUT" \
  --kroki "$KROKI" --paczka "$PACZKA" --na-karte "$NA" --workery 3 ${CKPT:+--checkpointing} $EXTRA &
PID=$!
RC=0
while kill -0 "$PID" 2>/dev/null; do                 # wait przerwany przez SIGTERM wraca wcześniej: czekamy dalej
  if wait "$PID"; then RC=0; else RC=$?; fi
done
echo "koniec $(date), kod $RC"; cat "$OUT/STATUS.json" || true
lfs quota -p 1299 /mnt/evafs | tail -1
exit $RC
