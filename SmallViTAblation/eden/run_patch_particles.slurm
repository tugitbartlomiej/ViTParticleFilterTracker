#!/bin/bash
# Patch-token particle recorder (grant theory convention) on several RF-DETR checkpoints.
# One GPU, minutes. Writes CZASTKI_<tag>/ under $OUT_DIR; nothing stays on the node.
# Required: ROOT, PY. Optional: OUT_DIR (default $ROOT/czastki).
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodelist=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=kl_czastki
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${ROOT:?}" "${PY:?}"
OUT_DIR="${OUT_DIR:-$ROOT/czastki}"
SCRATCH="/dev/shm/${USER}_${SLURM_JOB_ID:-$$}_kl_czastki"
cleanup() { rm -rf -- "$SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
mkdir -p "$SCRATCH" "$OUT_DIR"
export TMPDIR="$SCRATCH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
unset PYTHONPATH
test -s "$ROOT/package/tracks.json"
COCO="$HOME/.roboflow/models/rf-detr-nano.pth"
EP="$ROOT/results/epochs_uniform_time_b1500_s42/epoch_checkpoints"
run() { [[ -s "$OUT_DIR/CZASTKI_$2/meta.json" ]] && return 0; "$PY" -B -P "$ROOT/code/patch_particles.py" --package "$ROOT/package" --checkpoint "$1" --tag "$2" --out "$OUT_DIR"; }
run "$COCO" coco
run "$ROOT/checkpoints/ut_b0500_best.pth" b0500
run "$ROOT/checkpoints/ut_b2500_best.pth" b2500
run "$EP/epoch_005.pth" ep005
run "$EP/epoch_029.pth" ep029
du -sh "$OUT_DIR"/CZASTKI_*
