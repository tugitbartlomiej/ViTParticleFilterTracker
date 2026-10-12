#!/bin/bash
#SBATCH --job-name=sd_vit600_online
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=23:30:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy/logi/%x_%j.out
set -euo pipefail
B=/home2/faculty/bpiotrowski
MY=$B/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy
OLD=$B/smallvit_runs/2026-09-30_dyfuzja_sd_filmy
PY=$B/venvs/rfdetr_nano_clean_1_10_1/bin/python
SCRATCH=$(mktemp -d "/dev/shm/sdvit-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/sdvit-*) rm -rf -- "$SCRATCH";; esac; }
trap cleanup EXIT INT
trap 'touch "$MY/wyniki/trening/STOP"' TERM
export TMPDIR="$SCRATCH" HF_HOME="$SCRATCH/hf" TRITON_CACHE_DIR="$SCRATCH/triton"
export PYTHONPATH="$MY/deps:$OLD/deps" OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$MY/wyniki/trening" "$MY/wyniki/smoke"
cd "$MY/kod"
nvidia-smi
# No datasets, images, videos or activation caches are copied/extracted.
COMMON=(--data-manifest "$MY/wejscia/FILMY.json" --sd-root "$OLD/wyniki/trening/ema_020000" --vae-root "$OLD/modele/sd-vae-ft-mse" --vit-checkpoint "$MY/wejscia/MODEL_VIT_K1000.pt" --batch-size 2 --workers 2)
"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 run.py "${COMMON[@]}" --output "$MY/wyniki/trening" --steps 20000 --resume auto &
PID=$!
RC=0
while kill -0 "$PID" 2>/dev/null; do
    if wait "$PID"; then RC=0; else RC=$?; fi
done
exit "$RC"
