#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p debug
#SBATCH --gres=gpu:3
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=ablation_%a
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/ablation_%a_%j.log
#SBATCH --error=logs/ablation_%a_%j.log
#SBATCH --open-mode=append

# =============================================================================
# DETR Ablation Study — Training on GPU node
#
# All 91k images extracted from datasets_20250606.tar.gz to node-local TMPDIR.
# Annotation JSONs (per-variant) pre-generated on ~/DETR/ablation/v{1..5}/
#
# Usage:
#   sbatch --array=1-5 run_ablation_variant.slurm       # all 5 parallel
#   sbatch --array=1-5%2 run_ablation_variant.slurm     # max 2 concurrent
#   VARIANT=v3 sbatch run_ablation_variant.slurm        # single variant
# =============================================================================

set -euo pipefail

# === VARIANT RESOLUTION ===
VARIANTS=(v1 v2 v3 v4 v5 v6 v7)
VARIANT_NAMES=("FULL" "NO_FOURIER" "NO_EL2N" "QUALITY_ONLY" "RANDOM" "DIVERSITY_ONLY" "EL2N_ONLY")

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    IDX=$((SLURM_ARRAY_TASK_ID - 1))
    VARIANT="${VARIANTS[$IDX]}"
    VARIANT_NAME="${VARIANT_NAMES[$IDX]}"
elif [[ -n "${VARIANT:-}" ]]; then
    VARIANT_NAME="CUSTOM"
    for i in "${!VARIANTS[@]}"; do
        if [[ "${VARIANTS[$i]}" == "$VARIANT" ]]; then
            VARIANT_NAME="${VARIANT_NAMES[$i]}"
            break
        fi
    done
else
    echo "ERROR: Set SLURM_ARRAY_TASK_ID (--array) or VARIANT env var"
    exit 1
fi

echo '============================================================'
echo "  ABLATION STUDY: $VARIANT ($VARIANT_NAME)"
echo '============================================================'
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Start: $(date)"
echo "Node: ${SLURM_NODELIST:-local}"

# === PATHS ===
HOME_DIR=/mnt/evafs/faculty/home/bpiotrowski
ABLATION_DIR=$HOME_DIR/DETR/ablation
VARIANT_DIR=$ABLATION_DIR/$VARIANT
EPOCH_170_CKPT=$HOME_DIR/DETR/Checkpoints/checkpoint_epoch_170.pth
FULL_DATASET_TAR=$HOME_DIR/datasets_20250606.tar.gz

# Output dirs (variant-specific, on shared filesystem)
CHECKPOINT_DIR=$HOME_DIR/DETR/ablation_checkpoints/$VARIANT
OUTPUT_DIR=$HOME_DIR/DETR/ablation_output/$VARIANT
BEST_MODEL_DIR=$HOME_DIR/DETR/ablation_best/$VARIANT

# === VALIDATION ===
for f in "$VARIANT_DIR/annotations_train.json" "$VARIANT_DIR/annotations_val.json" "$EPOCH_170_CKPT" "$FULL_DATASET_TAR"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

# === SETUP TMPDIR (node-local SSD, ~1-2 TB free) ===
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-$$}_${VARIANT}}
mkdir -p "$TMPDIR"
echo "TMPDIR: $TMPDIR"
echo "TMPDIR free space: $(df -h "$TMPDIR" | tail -1 | awk '{print $4}')"

# === CONDA ===
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# === EXTRACT FULL 91K DATASET TO NODE-LOCAL STORAGE ===
echo ""
echo "=== EXTRACTING 91K DATASET TO TMPDIR ==="
echo "Source: $FULL_DATASET_TAR (39 GB compressed)"
echo "Target: $TMPDIR/images/"
echo "This will take ~5-10 minutes..."

EXTRACT_START=$(date +%s)

# Extract images directory from tar.gz
# Structure inside: datasets_20250606/datasets/DETR_augmented_dataset_20250218/images/*.jpg
mkdir -p "$TMPDIR/images"
tar -xzf "$FULL_DATASET_TAR" \
    --strip-components=3 \
    -C "$TMPDIR/images" \
    "datasets/DETR_augmented_dataset_20250218/images/" \
    2>/dev/null || {
    echo "Strip-components extraction failed, trying alternative..."
    # Alternative: extract everything then move
    tar -xzf "$FULL_DATASET_TAR" -C "$TMPDIR/"
    # Find the images directory
    IMG_FOUND=$(find "$TMPDIR" -type d -name "images" | head -1)
    if [[ -n "$IMG_FOUND" ]]; then
        if [[ "$IMG_FOUND" != "$TMPDIR/images" ]]; then
            mv "$IMG_FOUND"/* "$TMPDIR/images/" 2>/dev/null || true
        fi
    fi
}

EXTRACT_END=$(date +%s)
EXTRACT_TIME=$((EXTRACT_END - EXTRACT_START))

IMAGES_DIR="$TMPDIR/images"
IMAGE_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -name "*.jpg" | wc -l)
echo ""
echo "=== EXTRACTION COMPLETE ==="
echo "  Time: ${EXTRACT_TIME}s"
echo "  Images: $IMAGE_COUNT"
echo "  Disk used: $(du -sh $IMAGES_DIR | awk '{print $1}')"

if [[ "$IMAGE_COUNT" -lt 10000 ]]; then
    echo "WARNING: Expected ~91k images, got $IMAGE_COUNT"
    echo "Contents of TMPDIR:"
    find "$TMPDIR" -type d | head -20
    find "$TMPDIR" -name "*.jpg" | head -5
    # Try to continue anyway — might be enough for this variant
fi

# === SETUP OUTPUT DIRECTORIES ===
mkdir -p "$CHECKPOINT_DIR" "$OUTPUT_DIR" "$BEST_MODEL_DIR" logs

# Copy epoch 170 checkpoint as starting point (if not already there from previous run)
if [[ ! -f "$CHECKPOINT_DIR/checkpoint_epoch_170.pth" ]]; then
    echo "Copying epoch 170 checkpoint to $CHECKPOINT_DIR..."
    cp "$EPOCH_170_CKPT" "$CHECKPOINT_DIR/checkpoint_epoch_170.pth"
fi

# === DDP CONFIGURATION ===
# Detect GPU count from SLURM
NGPUS=${SLURM_GPUS_ON_NODE:-3}
# Parse comma-separated GPU list if needed
if echo "$NGPUS" | grep -q ','; then
    NGPUS=$(echo "$NGPUS" | tr ',' '\n' | wc -l)
fi
if [[ "$NGPUS" -lt 1 ]] || [[ "$NGPUS" -gt 12 ]]; then
    NGPUS=3
fi

export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + (${SLURM_JOB_ID:-$$} % 1000) + ${SLURM_ARRAY_TASK_ID:-0}))
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# === CHECKPOINT SYNC (background, every 10 min) ===
ARCHIVE_DIR=$HOME_DIR/DETR/Checkpoints/ablation/$VARIANT
mkdir -p "$ARCHIVE_DIR"

sync_checkpoints() {
    rsync -a --update "$CHECKPOINT_DIR/" "$ARCHIVE_DIR/" 2>&1 | grep -c "checkpoint" || true
    if [[ -d "$OUTPUT_DIR/logs" ]]; then
        rsync -a --update "$OUTPUT_DIR/logs/" "$ARCHIVE_DIR/tensorboard/" 2>/dev/null || true
    fi
}
trap 'echo "[$(date)] EXIT trap ($VARIANT)"; sync_checkpoints' EXIT INT TERM
(while true; do sleep 600; sync_checkpoints; done) &
SYNC_PID=$!

# === TRAINING ===
echo ""
echo "=== TRAINING CONFIGURATION ==="
echo "  Variant:     $VARIANT ($VARIANT_NAME)"
echo "  Node:        $(hostname)"
echo "  GPUs:        ${NGPUS}x"
echo "  Batch/GPU:   4 (effective: $((NGPUS * 4)))"
echo "  Epochs:      170 -> 300 (early stopping patience=30)"
echo "  Images dir:  $IMAGES_DIR ($IMAGE_COUNT images)"
echo "  Train ann:   $VARIANT_DIR/annotations_train.json"
echo "  Val ann:     $VARIANT_DIR/annotations_val.json"
echo "  Checkpoint:  $CHECKPOINT_DIR"
echo ""

cd /mnt/evafs/faculty/home/bpiotrowski/DETR

torchrun \
    --nproc_per_node=$NGPUS \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    detr_train_ablation.py \
    --images_dir "$IMAGES_DIR" \
    --annotations_path "$VARIANT_DIR/annotations_train.json" \
    --annotations_val_path "$VARIANT_DIR/annotations_val.json" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --best_model_dir "$BEST_MODEL_DIR" \
    --epochs ${EPOCHS:-300} \
    --batch_size 4 \
    --num_workers 4 \
    --num_queries 100 \
    --save_interval 5 \
    --patience 30 \
    --augment \
    --model_checkpoint facebook/detr-resnet-50 \
    --lr 5e-5 \
    --lr_backbone 5e-6 \
    --weight_decay 1e-4 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler cosine \
    --lr_min 1e-7 \
    --resume_training

TRAIN_EXIT=$?
echo ""
echo "Training exit code: $TRAIN_EXIT"

# === CLEANUP ===
kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_checkpoints

# Clean TMPDIR (node-local)
echo "Cleaning TMPDIR ($TMPDIR)..."
rm -rf "$TMPDIR"

echo ""
echo "============================================================"
echo "  $VARIANT ($VARIANT_NAME) COMPLETED"
echo "  Finished: $(date)"
echo "  Checkpoints: $ARCHIVE_DIR/"
echo "  Best model:  $BEST_MODEL_DIR/"
echo "============================================================"
