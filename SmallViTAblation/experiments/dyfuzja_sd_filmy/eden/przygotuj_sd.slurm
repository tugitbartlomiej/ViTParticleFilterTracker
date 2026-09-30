#!/bin/bash
#SBATCH --job-name=sd_latenty
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-30_dyfuzja_sd_filmy/logi/%x_%j.out
# Klatki filmów Cataract-LMM → latenty VAE SD i maski (kod/przygotuj_latenty.py), jeden proces na kartę, filmy
# podzielone między procesy. Wynik w ~/smallvit_data/cataract_lmm/sd576_<podział>[_co<N>] (poza katalogiem biegu, bo
# to wejście dla wielu uczeń). Film z gotowym indeksem jest pomijany, więc ponowne zgłoszenie dokańcza pracę.
#   sbatch [--gres=gpu:N] przygotuj_sd.slurm <train|val> [co_która_klatka=1] [limit_filmów=0]
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
PODZIAL=${1:?train albo val}; CO=${2:-1}; LIMIT=${3:-0}
B=/home2/faculty/bpiotrowski
MY=$B/smallvit_runs/2026-09-30_dyfuzja_sd_filmy
PY=$B/venvs/rfdetr_nano_clean_1_10_1/bin/python
ARCH=$B/smallvit_data/cataract_lmm/3_Object_Tracking/annotations
NAZWA=sd576_$PODZIAL; [ "$CO" != 1 ] && NAZWA=${NAZWA}_co$CO; [ "$LIMIT" != 0 ] && NAZWA=${NAZWA}_proba$LIMIT
OUT=$B/smallvit_data/cataract_lmm/$NAZWA
for f in "$MY/kod/przygotuj_latenty.py" "$MY/kod/wspolne.py" "$MY/wejscia/pool_manifest.jsonl" \
         "$MY/modele/sd-vae-ft-mse/config.json" "$MY/deps/diffusers/__init__.py"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
[ "$(ls $ARCH/*.zip | wc -l)" -ge 170 ] || { echo "archiwów filmów mniej niż 170"; exit 2; }
SCRATCH=$(mktemp -d "/dev/shm/sdlat-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/sdlat-*) rm -rf -- "$SCRATCH";; esac; }
trap cleanup EXIT INT TERM
export TMPDIR="$SCRATCH" HF_HOME="$SCRATCH/hf" TORCH_HOME="$SCRATCH/torch" TRITON_CACHE_DIR="$SCRATCH/triton" \
       MPLCONFIGDIR="$SCRATCH/mpl" PYTHONPATH="$MY/deps" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
N=$(nvidia-smi -L | wc -l)
echo "podział $PODZIAL, co $CO, limit $LIMIT, $N kart $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), węzeł $(hostname), start $(date)"
lfs quota -p 1299 /mnt/evafs | tail -1
cd "$MY/kod"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$N" przygotuj_latenty.py --archiwa "$ARCH" \
  --manifest "$MY/wejscia/pool_manifest.jsonl" --out "$OUT" --podzial "$PODZIAL" --co "$CO" \
  --vae "$MY/modele/sd-vae-ft-mse" --watki 8 --limit-filmow "$LIMIT"
echo "filmów gotowych: $(ls $OUT/indeks/*.json | grep -vc tmp), rozmiar $(du -sh $OUT | cut -f1), koniec $(date)"
lfs quota -p 1299 /mnt/evafs | tail -1
