#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodelist=hopper
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=250G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=detr_ddp_4gpu_hopper_ep140
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_ddp_4gpu_hopper_resume140_%j.log
#SBATCH --error=logs/detr_ddp_4gpu_hopper_resume140_%j.log
#SBATCH --open-mode=append

set -euo pipefail
echo '=== DETR OPTIMIZED DDP 4xGPU HOPPER | RESUME EPOCH 140 | NO MIXED PRECISION ==='
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURM_NODELIST"

TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR: $TMPDIR"

source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606.tar.gz
echo "Extracting dataset to $TMPDIR..."
tar -xzf "$ARCHIVE" -C "$TMPDIR"

ANNOT_JSON=$(find "$TMPDIR" -type f -name 'augmented_coco_20250417_030014.json' | head -n 1 || true)
IMAGES_DIR="$(dirname "$ANNOT_JSON")/images"

if [[ ! -f "$ANNOT_JSON" ]]; then
    echo "ERROR: JSON file not found"
    exit 1
fi
echo "Dataset ready: $ANNOT_JSON"

export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN

# Checkpoint archival setup - RESUME FROM EPOCH 140
ARCHIVE_DIR=/mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints
CHECKPOINT_DIR=./ckpt_ddp_2gpu_dgx1_500ep
mkdir -p "$ARCHIVE_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "=== CHECKPOINT CONFIGURATION ==="
echo "  Working dir: $CHECKPOINT_DIR"
echo "  Archive dir: $ARCHIVE_DIR"
echo "  Resume: ENABLED (from checkpoint_epoch_140.pth)"
echo "  Mixed Precision: DISABLED (fix for NaN issue)"

# Copy latest checkpoint from archive if not in working dir
if [[ ! -f "$CHECKPOINT_DIR/checkpoint_epoch_140.pth" ]]; then
    echo "Copying checkpoint_epoch_140.pth from archive..."
    rsync -av "$ARCHIVE_DIR/checkpoint_epoch_140.pth" "$CHECKPOINT_DIR/"
fi

# Function to sync checkpoints to archive
sync_checkpoints() {
    echo "[$(date)] Syncing checkpoints to archive..."
    rsync -av --update "$CHECKPOINT_DIR/" "$ARCHIVE_DIR/" 2>&1 | grep -E "(checkpoint|sent|total)" || true
    echo "[$(date)] Sync completed"
}

# Trap to ensure final sync on exit (success or failure)
trap 'echo "EXIT trap triggered"; sync_checkpoints' EXIT INT TERM

# Background loop to periodically sync checkpoints every 10 minutes
(
    while true; do
        sleep 600
        sync_checkpoints
    done
) &
SYNC_PID=$!
echo "Background checkpoint sync started (PID: $SYNC_PID)"

echo ""
echo "=== STARTING TRAINING ==="
echo "GPUs: 4x A100 (Hopper)"
echo "Epochs: 500 (resuming from 140)"
echo "Batch size per GPU: 16"
echo "Effective batch: 64 (4 GPU x 16)"
echo "Mixed Precision: DISABLED"
echo ""

torchrun \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29500 \
    detr_train_optimized.py \
    --images_dir "$IMAGES_DIR" \
    --annotations_path "$ANNOT_JSON" \
    --output_dir ./train_out_ddp_4gpu_hopper_resume140 \
    --checkpoint_dir ./ckpt_ddp_2gpu_dgx1_500ep \
    --best_model_dir ./best_ddp_4gpu_hopper_resume140 \
    --epochs 500 \
    --batch_size 16 \
    --num_workers 4 \
    --num_queries 100 \
    --save_interval 10 \
    --patience 50 \
    --augment \
    --model_checkpoint facebook/detr-resnet-50 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 1e-4 \
    --gradient_accumulation_steps 1 \
    --resume_training

TRAIN_EXIT_CODE=$?
echo ""
echo "=== TRAINING FINISHED ==="
echo "Exit code: $TRAIN_EXIT_CODE"

# Stop background sync
kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true

# Final checkpoint sync
echo "Performing final checkpoint sync..."
sync_checkpoints

echo "Job finished at: $(date)"
echo "All checkpoints archived to: $ARCHIVE_DIR"

exit $TRAIN_EXIT_CODE
