#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=detr_20k_scratch
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_20k_scratch_%j.log
#SBATCH --error=logs/detr_20k_scratch_%j.log
#SBATCH --open-mode=append

# =============================================================================
# DETR — trening OD ZERA (epoch 0 -> 400) na kurowanym zbiorze 20k
#
# ROZNICA vs run_ablation_variant.slurm:
#   - BRAK kopiowania checkpoint_epoch_170.pth
#   - BRAK --resume_training  => detr_train_ablation.py startuje start_epoch=0
#   - epochs 400 (zamiast 170->300), LR jak w base (1e-4 / 1e-5)
#   - NIE kasujemy posrednich checkpointow (potrzebne do trajektorii ep100/200/300/400)
#
# Init wag: facebook/detr-resnet-50 (COCO-pretrained) — taki sam jak base DETR.
# Dane: pool 91k (datasets_20250606.tar.gz) + anotacje wariantu (domyslnie v1 FULL = 20k).
#
# Usage:
#   sbatch run_detr_20k_from_scratch.slurm                # v1 (FULL curated 20k)
#   VARIANT=v5 sbatch run_detr_20k_from_scratch.slurm     # inny wariant 20k
#   EPOCHS=300 sbatch run_detr_20k_from_scratch.slurm     # inny budzet epok
# =============================================================================

set -euo pipefail

# === VARIANT (domyslnie v1 = FULL curated 20k) ===
VARIANT="${VARIANT:-v1}"
VARIANTS=(v1 v2 v3 v4 v5 v6 v7)
VARIANT_NAMES=("FULL" "NO_FOURIER" "NO_EL2N" "QUALITY_ONLY" "RANDOM" "DIVERSITY_ONLY" "EL2N_ONLY")
VARIANT_NAME="CUSTOM"
for i in "${!VARIANTS[@]}"; do
    if [[ "${VARIANTS[$i]}" == "$VARIANT" ]]; then VARIANT_NAME="${VARIANT_NAMES[$i]}"; break; fi
done

EPOCHS="${EPOCHS:-400}"

echo '============================================================'
echo "  DETR FROM SCRATCH on 20k: $VARIANT ($VARIANT_NAME) | 0 -> $EPOCHS"
echo '============================================================'
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start:  $(date)"
echo "Node:   ${SLURM_NODELIST:-local}"

# === PATHS ===
HOME_DIR=/mnt/evafs/faculty/home/bpiotrowski
ABLATION_DIR=$HOME_DIR/DETR/ablation
VARIANT_DIR=$ABLATION_DIR/$VARIANT
FULL_DATASET_TAR=$HOME_DIR/datasets_20250606.tar.gz

# Output dirs (osobne od ablacji, zeby nie nadpisac)
CHECKPOINT_DIR=$HOME_DIR/DETR/scratch_20k_checkpoints/$VARIANT
OUTPUT_DIR=$HOME_DIR/DETR/scratch_20k_output/$VARIANT
BEST_MODEL_DIR=$HOME_DIR/DETR/scratch_20k_best/$VARIANT
ARCHIVE_DIR=$HOME_DIR/DETR/Checkpoints/scratch_20k/$VARIANT

# === VALIDATION (UWAGA: BRAK wymogu checkpoint_epoch_170 — trenujemy od zera) ===
for f in "$VARIANT_DIR/annotations_train.json" "$VARIANT_DIR/annotations_val.json" "$FULL_DATASET_TAR"; do
    if [[ ! -f "$f" ]]; then echo "ERROR: brak pliku: $f"; exit 1; fi
done

# === TMPDIR (node-local) ===
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    TMPDIR="$SLURM_TMPDIR"
elif [[ -d "/raid" ]] && [[ "$(df /raid --output=avail 2>/dev/null | tail -1)" -gt 52428800 ]]; then
    TMPDIR="/raid/${USER}_${SLURM_JOB_ID:-$$}_scratch20k_${VARIANT}"
elif [[ "$(df /dev/shm --output=avail 2>/dev/null | tail -1)" -gt 52428800 ]]; then
    TMPDIR="/dev/shm/${USER}_${SLURM_JOB_ID:-$$}_scratch20k_${VARIANT}"
else
    TMPDIR="/tmp/${USER}_${SLURM_JOB_ID:-$$}_scratch20k_${VARIANT}"
fi
mkdir -p "$TMPDIR"
echo "TMPDIR: $TMPDIR (free: $(df -h "$TMPDIR" | tail -1 | awk '{print $4}'))"

# === CONDA ===
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# === EXTRACT POOL 91k -> node-local (anotacje 20k wybieraja podzbior) ===
echo ""
echo "=== EXTRACTING 91K POOL TO TMPDIR (~5-10 min) ==="
mkdir -p "$TMPDIR/images"
tar -xzf "$FULL_DATASET_TAR" \
    --strip-components=3 \
    -C "$TMPDIR/images" \
    "datasets/DETR_augmented_dataset_20250218/images/" \
    2>/dev/null || {
    echo "Strip-components extraction failed, fallback..."
    tar -xzf "$FULL_DATASET_TAR" -C "$TMPDIR/"
    IMG_FOUND=$(find "$TMPDIR" -type d -name "images" | head -1)
    if [[ -n "$IMG_FOUND" && "$IMG_FOUND" != "$TMPDIR/images" ]]; then
        mv "$IMG_FOUND"/* "$TMPDIR/images/" 2>/dev/null || true
    fi
}
IMAGES_DIR="$TMPDIR/images"
IMAGE_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -name "*.jpg" | wc -l)
echo "Pool images: $IMAGE_COUNT (disk: $(du -sh $IMAGES_DIR | awk '{print $1}'))"
if [[ "$IMAGE_COUNT" -lt 10000 ]]; then
    echo "WARNING: oczekiwano ~91k, jest $IMAGE_COUNT"
fi

# === OUTPUT DIRS ===
mkdir -p "$CHECKPOINT_DIR" "$OUTPUT_DIR" "$BEST_MODEL_DIR" "$ARCHIVE_DIR" logs
# UWAGA: NIE kopiujemy zadnego checkpointu — start od epoki 0.

# === DDP ===
NGPUS=${SLURM_GPUS_ON_NODE:-4}
if echo "$NGPUS" | grep -q ','; then NGPUS=$(echo "$NGPUS" | tr ',' '\n' | wc -l); fi
if [[ "$NGPUS" -lt 1 ]] || [[ "$NGPUS" -gt 12 ]]; then NGPUS=4; fi

export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + (${SLURM_JOB_ID:-$$} % 1000)))
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# === CHECKPOINT SYNC (co 10 min, archiwum — BEZ kasowania posrednich) ===
sync_checkpoints() {
    rsync -a --update "$CHECKPOINT_DIR/" "$ARCHIVE_DIR/" 2>&1 | grep -c "checkpoint" || true
    if [[ -d "$OUTPUT_DIR/logs" ]]; then
        rsync -a --update "$OUTPUT_DIR/logs/" "$ARCHIVE_DIR/tensorboard/" 2>/dev/null || true
    fi
}
trap 'echo "[$(date)] EXIT trap (scratch20k $VARIANT)"; sync_checkpoints; echo "Cleaning TMPDIR ($TMPDIR)..."; rm -rf "$TMPDIR"' EXIT INT TERM
(while true; do sleep 600; sync_checkpoints; done) &
SYNC_PID=$!

# === TRAINING ===
echo ""
echo "=== TRAINING CONFIGURATION ==="
echo "  Variant:    $VARIANT ($VARIANT_NAME)  [curated 20k]"
echo "  Init:       facebook/detr-resnet-50 (COCO)  | NO resume, start_epoch=0"
echo "  GPUs:       ${NGPUS}x  | batch/GPU 8 (eff $((NGPUS * 8)))"
echo "  Epochs:     0 -> $EPOCHS  (patience 60)"
echo "  LR:         1e-4 / backbone 1e-5  cosine  warmup 5"
echo "  Images:     $IMAGES_DIR ($IMAGE_COUNT pool; 20k wybrane przez anotacje)"
echo ""

cd "$HOME_DIR/DETR"

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
    --epochs "$EPOCHS" \
    --batch_size 8 \
    --num_workers 4 \
    --num_queries 100 \
    --save_interval 10 \
    --patience 60 \
    --augment \
    --model_checkpoint facebook/detr-resnet-50 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 1e-4 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler cosine \
    --lr_min 1e-7 \
    --warmup_epochs 5 \
    --use_amp

TRAIN_EXIT=$?
echo ""
echo "Training exit code: $TRAIN_EXIT"

# === CLEANUP (zachowujemy WSZYSTKIE checkpointy w archiwum) ===
kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_checkpoints

echo "Cleaning TMPDIR ($TMPDIR)..."
rm -rf "$TMPDIR"

echo ""
echo "============================================================"
echo "  DETR FROM SCRATCH 20k ($VARIANT) COMPLETED  $(date)"
echo "  Checkpoints (wszystkie): $ARCHIVE_DIR/"
echo "  Best model:              $BEST_MODEL_DIR/"
echo "============================================================"
exit $TRAIN_EXIT
