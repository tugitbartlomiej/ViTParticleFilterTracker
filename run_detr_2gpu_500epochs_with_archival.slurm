#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-1
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=detr_ddp_2gpu_500ep
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_ddp_2gpu_500ep_%j.log
#SBATCH --error=logs/detr_ddp_2gpu_500ep_%j.log
#SBATCH --open-mode=append

set -euo pipefail
echo '=== DETR OPTIMIZED DDP 2xGPU on DGX-1 | 500 EPOCHS | CHECKPOINT ARCHIVAL ==='
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR: $TMPDIR"

source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606.tar.gz
echo "Extracting dataset..."
tar -xzf "$ARCHIVE" -C "$TMPDIR"

ANNOT_JSON=$(find "$TMPDIR" -type f -name 'augmented_coco_20250417_030014.json' | head -n 1 || true)
IMAGES_DIR="$(dirname "$ANNOT_JSON")/images"

if [[ ! -f "$ANNOT_JSON" ]]; then
    echo "ERROR: JSON file not found"
    exit 1
fi

export WORLD_SIZE=2
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN

# Checkpoint archival setup
ARCHIVE_DIR=/mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints
CHECKPOINT_DIR=./ckpt_ddp_2gpu_dgx1_500ep
mkdir -p "$ARCHIVE_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "Checkpoint archival enabled:"
echo "  Working dir: $CHECKPOINT_DIR"
echo "  Archive dir: $ARCHIVE_DIR"

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

echo "Starting training on DGX-1 with 2 GPUs for 500 epochs"
echo "Resume training: ENABLED (will load latest checkpoint if available)"

torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29500 \
    detr_train_optimized.py \
    --images_dir "$IMAGES_DIR" \
    --annotations_path "$ANNOT_JSON" \
    --output_dir ./train_out_ddp_2gpu_dgx1_500ep \
    --checkpoint_dir ./ckpt_ddp_2gpu_dgx1_500ep \
    --best_model_dir ./best_ddp_2gpu_dgx1_500ep \
    --epochs 500 \
    --batch_size 16 \
    --num_workers 6 \
    --num_queries 100 \
    --save_interval 10 \
    --patience 50 \
    --augment \
    --model_checkpoint facebook/detr-resnet-50 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 1e-4 \
    --gradient_accumulation_steps 1 \
    --use_amp \
    --resume_training

TRAIN_EXIT_CODE=$?
echo "Training completed with exit code: $TRAIN_EXIT_CODE"

# Stop background sync
kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true

# Final checkpoint sync
echo "Performing final checkpoint sync..."
sync_checkpoints

echo "Job finished at: $(date)"
echo "All checkpoints archived to: $ARCHIVE_DIR"

exit $TRAIN_EXIT_CODE
