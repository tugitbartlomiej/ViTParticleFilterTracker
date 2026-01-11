#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --gres=gpu:3
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=detr_20k_4gpu
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_20k_4gpu_%j.log
#SBATCH --error=logs/detr_20k_4gpu_%j.log
#SBATCH --open-mode=append

set -eu
echo "============================================================"
echo "  DETR FINE-TUNING - RESUME ep295 TO 500 (4 GPU, ANY NODE)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"

# === SETUP TMPDIR ===
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR: $TMPDIR"

# === CONDA ACTIVATION ===
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# === EXTRACT 20K DATASET ===
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/20kSelectedImages.tar
echo ""
echo "=== EXTRACTING 20K DATASET ==="
mkdir -p "$TMPDIR/20k_dataset/images"
tar -xf "$ARCHIVE" -C "$TMPDIR/20k_dataset"
cp /mnt/evafs/faculty/home/bpiotrowski/merged_20k_annotations_fixed.json "$TMPDIR/20k_dataset/"

cd "$TMPDIR/20k_dataset"
if [[ -f "merged_20k_annotations_fixed.json" ]]; then
    mkdir -p images_tmp
    find . -maxdepth 1 -name "*.jpg" -exec mv {} images_tmp/ \; 2>/dev/null || true
    mv images_tmp/* images/ 2>/dev/null || true
    rmdir images_tmp 2>/dev/null || true
fi

ANNOT_JSON="$TMPDIR/20k_dataset/merged_20k_annotations_fixed.json"
IMAGES_DIR="$TMPDIR/20k_dataset/images"
if [[ \! -d "$IMAGES_DIR" ]] || [[ -z "$(ls -A $IMAGES_DIR 2>/dev/null)" ]]; then
    IMAGES_DIR="$TMPDIR/20k_dataset"
fi

IMAGE_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -name "*.jpg" | wc -l)
echo "=== DATASET: $IMAGE_COUNT images ==="

# === DDP CONFIGURATION ===
export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# === DIRECTORIES ===
ARCHIVE_DIR=/mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints/20k_finetune_v2_fixed
CHECKPOINT_DIR=/mnt/evafs/faculty/home/bpiotrowski/DETR/ckpt_20k_finetune_v2_fixed
OUTPUT_DIR=/mnt/evafs/faculty/home/bpiotrowski/DETR/train_out_20k_finetune
BEST_MODEL_DIR=/mnt/evafs/faculty/home/bpiotrowski/DETR/best_20k_finetune

mkdir -p "$ARCHIVE_DIR" "$CHECKPOINT_DIR" "$OUTPUT_DIR" "$BEST_MODEL_DIR" logs
mkdir -p "$OUTPUT_DIR/logs"  # TensorBoard logs directory
mkdir -p "$ARCHIVE_DIR/tensorboard_logs"

# Verify checkpoint
if [[ \! -f "$CHECKPOINT_DIR/checkpoint_epoch_295.pth" ]]; then
    echo "ERROR: checkpoint_epoch_295.pth not found\!"
    exit 1
fi
echo "Resume from: checkpoint_epoch_295.pth"

# === CHECKPOINT & TENSORBOARD SYNC ===
sync_all() {
    echo "[$(date)] Syncing checkpoints and TensorBoard logs..."
    # Sync checkpoints
    rsync -av --update "$CHECKPOINT_DIR/" "$ARCHIVE_DIR/" 2>&1 | grep -E "(checkpoint|sent|total)" || true
    # Sync TensorBoard logs
    if [[ -d "$OUTPUT_DIR/logs" ]]; then
        rsync -av --update "$OUTPUT_DIR/logs/" "$ARCHIVE_DIR/tensorboard_logs/" 2>&1 | grep -E "(events|sent|total)" || true
    fi
    echo "[$(date)] Sync done"
}

trap 'echo "EXIT trap"; sync_all' EXIT INT TERM

# Background sync every 10 minutes
(while true; do sleep 600; sync_all; done) &
SYNC_PID=$\!
echo "Background sync started (PID: $SYNC_PID)"

echo ""
echo "=== TRAINING CONFIG ==="
echo "  GPUs: 3"
echo "  Resume: epoch 295 -> 500"
echo "  Batch/GPU: 4 (effective: 12)"
echo "  LR: 5e-5, Backbone: 5e-6"
echo "  TensorBoard: $OUTPUT_DIR/logs/"
echo "  Archive: $ARCHIVE_DIR/tensorboard_logs/"

cd /mnt/evafs/faculty/home/bpiotrowski/DETR

torchrun     --nproc_per_node=3     --master_addr="$MASTER_ADDR"     --master_port="$MASTER_PORT"     --rdzv_backend=c10d     --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"     detr_train_optimized.py     --images_dir "$IMAGES_DIR"     --annotations_path "$ANNOT_JSON"     --output_dir "$OUTPUT_DIR"     --checkpoint_dir "$CHECKPOINT_DIR"     --best_model_dir "$BEST_MODEL_DIR"     --epochs 500     --batch_size 4     --num_workers 4     --num_queries 100     --save_interval 5     --patience 50     --augment     --model_checkpoint facebook/detr-resnet-50     --lr 5e-5     --lr_backbone 5e-6     --weight_decay 1e-4     --gradient_accumulation_steps 1     --lr_scheduler cosine     --lr_min 1e-7     --resume_training

EXIT_CODE=$?

kill $SYNC_PID 2>/dev/null || true
sync_all

echo ""
echo "=== FINISHED (exit: $EXIT_CODE) at $(date) ==="
echo "Checkpoints: $ARCHIVE_DIR/"
echo "TensorBoard: $ARCHIVE_DIR/tensorboard_logs/"

exit $EXIT_CODE
