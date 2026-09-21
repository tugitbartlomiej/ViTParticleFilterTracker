#!/bin/bash
# One RF-DETR Nano run of the static campaign recipe that KEEPS weights every N epochs,
# then reads the latent of every kept checkpoint on the dense probe clips.
# Required: ROOT, PROJECT_DIR (source with --keep-epoch-checkpoints), DATA_TAR, PROTOCOL, PY,
#           VARIANT (file name inside <data>/variants), RUN_ID, TRAIN_SEED. Optional: EVERY (default 3).
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodelist=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=kl_epochs
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${ROOT:?}" "${PROJECT_DIR:?}" "${DATA_TAR:?}" "${PROTOCOL:?}" "${PY:?}" "${VARIANT:?}" "${RUN_ID:?}" "${TRAIN_SEED:?}"
EVERY="${EVERY:-3}"
if [[ -d /raid && -w /raid ]]; then SCRATCH="/raid/${USER}_${SLURM_JOB_ID:-$$}_kl_epochs"
else SCRATCH="/dev/shm/${USER}_${SLURM_JOB_ID:-$$}_kl_epochs"; fi
cleanup() { rm -rf -- "$SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
mkdir -p "$SCRATCH/data" "$ROOT/results" "$ROOT/latents_epochs"
export TMPDIR="$SCRATCH" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor" MPLCONFIGDIR="$SCRATCH/mpl"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
unset PYTHONPATH
test -s "$DATA_TAR" && test -s "$ROOT/package/tracks.json"
test ! -e "$ROOT/results/$RUN_ID"
tar -xf "$DATA_TAR" -C "$SCRATCH/data"
DATA="$(dirname "$(find "$SCRATCH/data" -maxdepth 2 -name pool_manifest.jsonl | head -1)")"
test -s "$DATA/variants/$VARIANT"
cd "$PROJECT_DIR"
"$PY" -B pipeline/train/run_with_validation.py --run-id "$RUN_ID" --data-root "$DATA" --out "$ROOT/results" \
    --protocol "$PROTOCOL" --variant-file "$DATA/variants/$VARIANT" --train-seed "$TRAIN_SEED" \
    --profile nano_static_pilot --expected-epochs 30 --compress-predictions --keep-epoch-checkpoints "$EVERY"
COCO="$HOME/.roboflow/models/rf-detr-nano.pth"
"$PY" -B "$ROOT/code/detector_latent.py" --package "$ROOT/package" --checkpoint "$COCO" --tag ep000_coco --out "$ROOT/latents_epochs"
for ckpt in "$ROOT/results/$RUN_ID"/epoch_checkpoints/epoch_*.pth; do
    tag="$(basename "$ckpt" .pth)"
    "$PY" -B "$ROOT/code/detector_latent.py" --package "$ROOT/package" --checkpoint "$ckpt" --tag "$tag" --out "$ROOT/latents_epochs"
done
ls -la "$ROOT/latents_epochs"
