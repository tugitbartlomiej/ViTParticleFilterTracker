#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=ablation_eval
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/ablation_eval_%j.log
#SBATCH --error=logs/ablation_eval_%j.log
#SBATCH --open-mode=append

# =============================================================================
# Ablation Study — Evaluation Job
#
# Evaluates all ablation variants (V1-V7) on:
#   1. Same-distribution test set (from video-level split)
#   2. Cross-dataset external benchmark (Roboflow)
#
# Requires: all best models saved in ~/DETR/ablation_best/v{2..7}/
#           V1 uses existing checkpoint (epoch 210)
#
# Usage:
#   sbatch run_ablation_eval.slurm
# =============================================================================

set -euo pipefail

echo '============================================================'
echo '  ABLATION STUDY — EVALUATION'
echo '============================================================'
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start: $(date)"
echo "Node: ${SLURM_NODELIST:-local}"

HOME_DIR=/mnt/evafs/faculty/home/bpiotrowski
FULL_DATASET_TAR=$HOME_DIR/datasets_20250606.tar.gz

# === SETUP ===
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-$$}_eval}
mkdir -p "$TMPDIR"

source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# === EXTRACT IMAGES ===
echo ""
echo "=== Extracting 91k images to TMPDIR ==="
mkdir -p "$TMPDIR/images"
tar -xzf "$FULL_DATASET_TAR" \
    --strip-components=3 \
    -C "$TMPDIR/images" \
    "datasets/DETR_augmented_dataset_20250218/images/" \
    2>/dev/null || {
    echo "Strip extraction failed, trying full extract..."
    tar -xzf "$FULL_DATASET_TAR" -C "$TMPDIR/"
    IMG_FOUND=$(find "$TMPDIR" -type d -name "images" | head -1)
    if [[ -n "$IMG_FOUND" ]] && [[ "$IMG_FOUND" != "$TMPDIR/images" ]]; then
        mv "$IMG_FOUND"/* "$TMPDIR/images/" 2>/dev/null || true
    fi
}

IMAGE_COUNT=$(find "$TMPDIR/images" -maxdepth 1 -name "*.jpg" | wc -l)
echo "Images extracted: $IMAGE_COUNT"

cd $HOME_DIR/DETR
mkdir -p logs

# === SAME-DISTRIBUTION EVALUATION ===
echo ""
echo "=== Same-Distribution Evaluation ==="
python evaluate_ablation.py \
    --variants-dir $HOME_DIR/DETR/ablation_best \
    --v1-checkpoint $HOME_DIR/DETR/Checkpoints/20k_finetune_v2_fixed/checkpoint_epoch_210.pth \
    --test-annotations $HOME_DIR/DETR/ablation/v2/annotations_test.json \
    --images-dir "$TMPDIR/images" \
    --output $HOME_DIR/DETR/ablation/results_same_dist.json \
    --coco-eval \
    --variants v1,v2,v3,v4,v5,v6,v7

# === CROSS-DATASET EVALUATION ===
EXTERNAL_DIR=$HOME_DIR/DETR/external_benchmark
if [[ -d "$EXTERNAL_DIR/valid" ]] && [[ -f "$EXTERNAL_DIR/valid/_annotations.coco.json" ]]; then
    echo ""
    echo "=== Cross-Dataset Evaluation ==="
    python evaluate_ablation.py \
        --variants-dir $HOME_DIR/DETR/ablation_best \
        --v1-checkpoint $HOME_DIR/DETR/Checkpoints/20k_finetune_v2_fixed/checkpoint_epoch_210.pth \
        --test-annotations $HOME_DIR/DETR/ablation/v2/annotations_test.json \
        --images-dir "$TMPDIR/images" \
        --cross-annotations "$EXTERNAL_DIR/valid/_annotations.coco.json" \
        --cross-images-dir "$EXTERNAL_DIR/valid/" \
        --output $HOME_DIR/DETR/ablation/results_cross_dataset.json \
        --coco-eval \
        --variants v1,v2,v3,v4,v5,v6,v7
else
    echo "WARNING: External benchmark not found at $EXTERNAL_DIR/valid/ — skipping cross-dataset eval"
fi

# === CLEANUP ===
echo "Cleaning TMPDIR..."
rm -rf "$TMPDIR"

echo ""
echo "============================================================"
echo "  EVALUATION COMPLETED"
echo "  Finished: $(date)"
echo "  Results: ~/DETR/ablation/results_*.json"
echo "============================================================"
