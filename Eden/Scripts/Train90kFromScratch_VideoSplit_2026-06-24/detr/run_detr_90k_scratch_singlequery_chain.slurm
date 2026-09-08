#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=detr_90k_scratch_q1
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_90k_scratch_q1_%j.log
#SBATCH --error=logs/detr_90k_scratch_q1_%j.log
#SBATCH --open-mode=append

# =============================================================================
# DETR - trening OD ZERA (epoch 0 -> EPOCHS) na PELNYM poolu 91k pod VIDEO-LEVEL
# splitem. WERSJA LANCUCHOWA (chain): kazde ogniwo wznawia z ostatniego checkpointu.
#
# CEL: leakage-free odpowiednik UNCURATED baseline z artykulu (pelny pool), ale
#      z poprawnym podzialem PER-WIDEO (35/8/7) zamiast losowego frame-level.
#      train = WSZYSTKIE klatki z 35 wideo treningowych (~71k, BEZ kuracji 20k).
#
# DANE: ~/DETR/fullpool_videosplit/annotations_{train,val}.json  <- z prepare_fullpool_videosplit.py
#       (ten sam video_split.json 35/8/7 seed 42 co ablacje V1-V7 -> spojny split)
#
# ROZNICE vs run_detr_20k_scratch_chain.slurm:
#   - BRAK wariantow V1-V7      -> jeden pelny pool (train nie jest kurowany do 20k)
#   - DANE: fullpool_videosplit -> annotations_train.json ma ~71k, nie 20k
#   - reszta mechaniki identyczna (GUARD, resume, pruning, TMPDIR)
#
# Init wag: facebook/detr-resnet-50 (COCO) — taki sam jak base DETR. start_epoch=0.
#
# Uzycie (przez wrapper submit_detr_chain.sh, ktory laczy ogniwa zaleznosciami):
#   ./submit_detr_chain.sh 4            # 4 ogniwa, 0->EPOCHS (default 200)
#   EPOCHS=170 ./submit_detr_chain.sh 3
# =============================================================================

set -euo pipefail

EPOCHS="${EPOCHS:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
MILESTONE="${MILESTONE:-50}"   # checkpointy bedace wielokrotnoscia MILESTONE zachowywane na stale

echo '============================================================'
echo "  DETR FROM SCRATCH (CHAIN) on FULL 91k pool | VIDEO-LEVEL split | 0 -> $EPOCHS"
echo '============================================================'
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start:  $(date)"
echo "Node:   ${SLURM_NODELIST:-local}"

# === PATHS ===
HOME_DIR=/mnt/evafs/faculty/home/bpiotrowski
DATA_DIR=$HOME_DIR/DETR/fullpool_videosplit            # <- prepare_fullpool_videosplit.py
FULL_DATASET_TAR=$HOME_DIR/datasets_20250606.tar.gz

# Output dirs (osobne od ablacji/20k, zeby nie nadpisac)
CHECKPOINT_DIR=$HOME_DIR/DETR/fullpool_vs_scratch_q1_checkpoints
OUTPUT_DIR=$HOME_DIR/DETR/fullpool_vs_scratch_q1_output
BEST_MODEL_DIR=$HOME_DIR/DETR/fullpool_vs_scratch_q1_best

mkdir -p "$CHECKPOINT_DIR" "$OUTPUT_DIR" "$BEST_MODEL_DIR" logs

# =============================================================================
# GUARD: jesli trening juz ukonczony -> exit 0 PRZED kosztownym rozpakowaniem.
# =============================================================================
latest_epoch() {
    find "$CHECKPOINT_DIR" -maxdepth 1 -name 'checkpoint_epoch_*.pth' 2>/dev/null \
        | sed -E 's/.*checkpoint_epoch_([0-9]+)\.pth/\1/' | sort -n | tail -1
}
LAT=$(latest_epoch || true)
if [[ -f "$CHECKPOINT_DIR/final_checkpoint.pth" ]]; then
    echo "[GUARD] final_checkpoint.pth istnieje -> trening ukonczony. Ogniwo konczy sie (exit 0)."
    exit 0
fi
if [[ -n "$LAT" ]] && [[ "$LAT" -ge "$EPOCHS" ]]; then
    echo "[GUARD] ostatni checkpoint = ep$LAT >= $EPOCHS -> nic do roboty (exit 0)."
    exit 0
fi
echo "[GUARD] ostatni checkpoint: ${LAT:-BRAK} (cel: $EPOCHS) -> kontynuuje."

# === VALIDATION ===
for f in "$DATA_DIR/annotations_train.json" "$DATA_DIR/annotations_val.json" "$FULL_DATASET_TAR"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: brak pliku: $f"
        echo "       (najpierw uruchom prepare_fullpool_videosplit.py -> $DATA_DIR)"
        exit 1
    fi
done

# === TMPDIR (node-local) — wybor przez TEST ZAPISYWALNOSCI + miejsca ===
JOBSUB="${USER}_${SLURM_JOB_ID:-$$}_fullpool_vs"
NEED_KB=52428800   # ~50 GB
TMPDIR=""
for parent in "${SLURM_TMPDIR:-}" /raid /tmp /dev/shm; do
    [[ -z "$parent" ]] && continue
    cand="$parent/$JOBSUB"
    if mkdir -p "$cand" 2>/dev/null; then
        avail=$(df --output=avail "$cand" 2>/dev/null | tail -1 | tr -d ' ')
        if [[ -n "$avail" && "$avail" -gt "$NEED_KB" ]]; then
            TMPDIR="$cand"; break
        fi
        rmdir "$cand" 2>/dev/null || true
    fi
done
if [[ -z "$TMPDIR" ]]; then
    echo "ERROR: brak zapisywalnego scratch z >50GB (probowano: SLURM_TMPDIR /raid /tmp /dev/shm)"; exit 1
fi

trap 'echo "[$(date)] CLEANUP: rm -rf $TMPDIR"; rm -rf "$TMPDIR" 2>/dev/null || true' EXIT INT TERM
echo "TMPDIR: $TMPDIR (free: $(df -h "$TMPDIR" | tail -1 | awk '{print $4}'))"

# === CONDA ===
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# === EXTRACT POOL 91k -> node-local (anotacje wybieraja train-video frames) ===
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

# === PRUNING checkpointow (oszczednosc home: 475 MB/ckpt) ===
prune_checkpoints() {
    local eps recent e
    eps=$(find "$CHECKPOINT_DIR" -maxdepth 1 -name 'checkpoint_epoch_*.pth' 2>/dev/null \
          | sed -E 's/.*checkpoint_epoch_([0-9]+)\.pth/\1/' | sort -n)
    [[ -z "$eps" ]] && return 0
    recent=$(echo "$eps" | tail -5)
    for e in $eps; do
        (( e % MILESTONE == 0 )) && continue
        echo "$recent" | grep -qx "$e" && continue
        rm -f "$CHECKPOINT_DIR/checkpoint_epoch_${e}.pth"
    done
    return 0
}

trap 'echo "[$(date)] EXIT trap (fullpool_vs)"; prune_checkpoints; echo "Cleaning TMPDIR ($TMPDIR)..."; rm -rf "$TMPDIR"' EXIT INT TERM
(while true; do sleep 600; prune_checkpoints; done) &
PRUNE_PID=$!

# === TRAINING ===
echo ""
echo "=== TRAINING CONFIGURATION ==="
echo "  Dataset:    FULL pool 91k, VIDEO-LEVEL split (train = train-video frames, UNCURATED)"
echo "  Data:       $DATA_DIR (annotations_train/val.json)"
echo "  Init:       facebook/detr-resnet-50 (COCO)  | resume_training ON (ep0 jesli pusto)"
echo "  Resume od:  ${LAT:-ep0 (od zera)}"
echo "  GPUs:       ${NGPUS}x  | batch/GPU 8 (eff $((NGPUS * 8)))"
echo "  Epochs:     0 -> $EPOCHS  (patience 60, save co $SAVE_INTERVAL)"
echo "  LR:         1e-4 / backbone 1e-5  cosine  warmup 5"
echo "  Images:     $IMAGES_DIR ($IMAGE_COUNT pool; train-videos wybrane przez anotacje)"
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
    --annotations_path "$DATA_DIR/annotations_train.json" \
    --annotations_val_path "$DATA_DIR/annotations_val.json" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --best_model_dir "$BEST_MODEL_DIR" \
    --epochs "$EPOCHS" \
    --batch_size 8 \
    --num_workers 4 \
    --num_queries 1 \
    --save_interval "$SAVE_INTERVAL" \
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
    --use_amp \
    --resume_training

TRAIN_EXIT=$?
echo ""
echo "Training exit code: $TRAIN_EXIT"

# === CLEANUP ===
kill $PRUNE_PID 2>/dev/null || true
wait $PRUNE_PID 2>/dev/null || true
prune_checkpoints

echo "Cleaning TMPDIR ($TMPDIR)..."
rm -rf "$TMPDIR"

LAT_END=$(latest_epoch || true)
echo ""
echo "============================================================"
echo "  DETR FROM SCRATCH FULLPOOL-VS -- ogniwo zakonczone  $(date)"
echo "  Ostatni checkpoint: ep${LAT_END:-?} / $EPOCHS"
echo "  Checkpoints:        $CHECKPOINT_DIR/"
echo "  Best model:         $BEST_MODEL_DIR/"
echo "  (Kolejne ogniwo (jesli w kolejce) wznowi automatycznie z ep${LAT_END:-?}.)"
echo "============================================================"
exit $TRAIN_EXIT
