#!/bin/bash
# Detector-latent probe: read the RF-DETR latent of several checkpoints on dense clips.
# One GPU, short. Writes only small NPZ files to $OUT_DIR; nothing stays on the node.
# Required: ROOT (run root with package/, checkpoints/, code/), PY.
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodelist=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=kl_latent
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${ROOT:?}" "${PY:?}"
OUT_DIR="${OUT_DIR:-$ROOT/latents}"
SCRATCH="/dev/shm/${USER}_${SLURM_JOB_ID:-$$}_kl_latent"
cleanup() { rm -rf -- "$SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
mkdir -p "$SCRATCH" "$OUT_DIR"
export TMPDIR="$SCRATCH" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor" MPLCONFIGDIR="$SCRATCH/mpl"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
unset PYTHONPATH
test -s "$ROOT/package/tracks.json"
COCO="$HOME/.roboflow/models/rf-detr-nano.pth"
test -s "$COCO"
"$PY" -B "$ROOT/code/detector_latent.py" --package "$ROOT/package" --checkpoint "$COCO" --tag coco --out "$OUT_DIR"
for ckpt in "$ROOT"/checkpoints/*.pth; do
    tag="$(basename "$ckpt" .pth)"
    [[ -s "$OUT_DIR/LATENT_${tag}.npz" ]] && continue
    "$PY" -B "$ROOT/code/detector_latent.py" --package "$ROOT/package" --checkpoint "$ckpt" --tag "$tag" --out "$OUT_DIR"
done
ls -la "$OUT_DIR"
