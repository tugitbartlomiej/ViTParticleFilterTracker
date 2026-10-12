#!/bin/bash
#SBATCH --job-name=vit8z256
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_8z256/logs/%x_%j.out
#SBATCH --error=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_8z256/logs/%x_%j.err
set -euo pipefail
ROOT=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_8z256
SOURCE=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_1z256
PY=/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python
DEPS=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-28_maly_vit_od_zera/deps
STEPS=${1:-500}
SUBDIR=${2:-results}
export PYTHONPATH="$DEPS"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p "$ROOT/logs" "$ROOT/$SUBDIR"
cd "$ROOT/code"
for VARIANT in 0 1; do
  "$PY" -u -B train_compare.py \
    --variant "$VARIANT" \
    --steps "$STEPS" \
    --data "$SOURCE/input_pack" \
    --checkpoint "$SOURCE/assets/MALY_VIT_K8000_s47_losowe.pth" \
    --output "$ROOT/$SUBDIR"
done
"$PY" -u -B aggregate.py "$ROOT/$SUBDIR"
