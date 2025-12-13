#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --gres=gpu:2
#SBATCH --mem=250G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=detr_20k_finetune_hopper
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_20k_finetune_hopper_%j.log
#SBATCH --error=logs/detr_20k_finetune_hopper_%j.log
#SBATCH --open-mode=append

set -euo pipefail
echo '============================================================'
echo '  DETR FINE-TUNING ON 20K SELECTED DATASET (HOPPER, 2x H100)'
echo '============================================================'
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURM_NODELIST"

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
echo "Archive: $ARCHIVE"
echo "Extracting to: $TMPDIR/20k_dataset"

mkdir -p "$TMPDIR/20k_dataset/images"
tar -xf "$ARCHIVE" -C "$TMPDIR/20k_dataset"
cp /mnt/evafs/faculty/home/bpiotrowski/merged_20k_annotations.json "$TMPDIR/20k_dataset/"

cd "$TMPDIR/20k_dataset"
if [[ -f "merged_20k_annotations.json" ]]; then
    mkdir -p images_tmp
    mv *.jpg images_tmp/ 2>/dev/null || true
    mv images_tmp/* images/ 2>/dev/null; rmdir images_tmp
fi

ANNOT_JSON="$TMPDIR/20k_dataset/merged_20k_annotations.json"
IMAGES_DIR="$TMPDIR/20k_dataset/images"
if [[ ! -d "$IMAGES_DIR" ]] || [[ -z "$(ls -A $IMAGES_DIR 2>/dev/null)" ]]; then
    IMAGES_DIR="$TMPDIR/20k_dataset"
fi

if [[ ! -f "$ANNOT_JSON" ]]; then
    echo "ERROR: Annotations JSON not found at: $ANNOT_JSON"
    ls -la "$TMPDIR/20k_dataset" | head -20
    exit 1
fi

IMAGE_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -name \"*.jpg\" | wc -l)
echo ""
echo \"=== DATASET VERIFICATION ===\"
echo \"  Annotations: $ANNOT_JSON\"
echo \"  Images dir: $IMAGES_DIR\"
echo \"  Image count: $IMAGE_COUNT\"
if [[ \"$IMAGE_COUNT\" -lt 1000 ]]; then
    echo \"ERROR: Too few images found ($IMAGE_COUNT). Expected ~20000.\"
    exit 1
fi

# === DDP CONFIGURATION ===
export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0

# === NCCL SETTINGS ===
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,NET
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

# === CHECKPOINT CONFIGURATION ===
ARCHIVE_DIR=/mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints
CHECKPOINT_DIR=./ckpt_20k_finetune
OUTPUT_DIR=./train_out_20k_finetune
BEST_MODEL_DIR=./best_20k_finetune

mkdir -p \"$ARCHIVE_DIR/20k_finetune\"
mkdir -p \"$CHECKPOINT_DIR\" \"$OUTPUT_DIR\" \"$BEST_MODEL_DIR\" logs

sync_checkpoints() {
    echo \"[$(date)] Syncing checkpoints to archive...\"
    rsync -av --update \"$CHECKPOINT_DIR/\" \"$ARCHIVE_DIR/20k_finetune/\" 2>&1 | grep -E \"(checkpoint|final_checkpoint|sent|total)\" || true
    echo \"[$(date)] Sync completed\"
}
trap 'echo \"EXIT trap triggered\"; sync_checkpoints' EXIT INT TERM
(
    while true; do
        sleep 600
        sync_checkpoints
    done
) &
SYNC_PID=$!
echo \"Background checkpoint sync started (PID: $SYNC_PID)\"

echo ""
echo \"=== TRAINING CONFIGURATION ===\"
echo \"  GPUs: 2x H100\"
echo \"  Batch per GPU: 4 (effective batch 8)\"
echo \"  Mixed Precision: DISABLED (stability)\"

cd /mnt/evafs/faculty/home/bpiotrowski/DETR

torchrun \
    --nproc_per_node=2 \
    --master_addr=\"$MASTER_ADDR\" \
    --master_port=\"$MASTER_PORT\" \
    --rdzv_backend=c10d \
    --rdzv_endpoint=\"$MASTER_ADDR:$MASTER_PORT\" \
    detr_train_optimized.py \
    --images_dir \"$IMAGES_DIR\" \
    --annotations_path \"$ANNOT_JSON\" \
    --output_dir \"$OUTPUT_DIR\" \
    --checkpoint_dir \"$CHECKPOINT_DIR\" \
    --best_model_dir \"$BEST_MODEL_DIR\" \
    --epochs 300 \
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

TRAIN_EXIT_CODE=$?

echo \"\"
echo \"=== TRAINING FINISHED ===\"
echo \"Exit code: $TRAIN_EXIT_CODE\"

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true

echo \"Performing final checkpoint sync...\"
sync_checkpoints

echo \"\"
echo \"============================================================\"
echo \"  JOB COMPLETED\"
echo \"  Finished at: $(date)\"
echo \"  Checkpoints archived to: $ARCHIVE_DIR/20k_finetune/\"
