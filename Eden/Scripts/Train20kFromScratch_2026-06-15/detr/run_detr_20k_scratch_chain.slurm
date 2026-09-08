#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=detr_20k_scratch
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_20k_scratch_%j.log
#SBATCH --error=logs/detr_20k_scratch_%j.log
#SBATCH --open-mode=append

# =============================================================================
# DETR - trening OD ZERA (epoch 0 -> 400) na kurowanym video-level zbiorze 20k
# WERSJA LANCUCHOWA (chain): kazde ogniwo wznawia z ostatniego checkpointu.
#
# DANE: ablation/v1 (FULL, video-level split 35/8/7 wideo, seed 42) -- ten sam
#       poprawnie wykurowany zbior co ablacje V1-V7. NIE re-kurujemy, NIE wgrywamy.
#
# ROZNICE vs run_detr_20k_from_scratch.slurm:
#   - partycja: long (zamiast hopper)  -> limit 5 dni, mieszcza sie 4-dniowe ogniwa
#   - --resume_training WLACZONE        -> ogniwo 1 (pusty katalog) startuje od ep0
#                                          z initem COCO; ogniwa 2,3.. wznawiaja
#   - GUARD na poczatku                 -> jesli trening juz ukonczony, exit 0
#                                          PRZED rozpakowaniem danych (tani no-op)
#   - BRAK duplikatu ARCHIVE_DIR        -> checkpoint = 475 MB; trzymamy 1 kopie
#                                          na home + pruning (home ma ~37 GB wolne!)
#   - pruning checkpointow              -> kamienie milowe co 50 + 5 najnowszych
#
# Uzycie (przez wrapper, ktory laczy ogniwa zaleznosciami):
#   ./submit_detr_chain.sh 3            # 3 ogniwa v1, 0->400
#   VARIANT=v1 EPOCHS=400 ./submit_detr_chain.sh 3
# =============================================================================

set -euo pipefail

# === VARIANT (domyslnie v1 = FULL curated 20k, video-level) ===
VARIANT="${VARIANT:-v1}"
VARIANTS=(v1 v2 v3 v4 v5 v6 v7)
VARIANT_NAMES=("FULL" "NO_FOURIER" "NO_EL2N" "QUALITY_ONLY" "RANDOM" "DIVERSITY_ONLY" "EL2N_ONLY")
VARIANT_NAME="CUSTOM"
for i in "${!VARIANTS[@]}"; do
    if [[ "${VARIANTS[$i]}" == "$VARIANT" ]]; then VARIANT_NAME="${VARIANT_NAMES[$i]}"; break; fi
done

EPOCHS="${EPOCHS:-400}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
MILESTONE="${MILESTONE:-50}"   # checkpointy bedace wielokrotnoscia MILESTONE sa zachowywane na stale

echo '============================================================'
echo "  DETR FROM SCRATCH (CHAIN) on 20k: $VARIANT ($VARIANT_NAME) | 0 -> $EPOCHS"
echo '============================================================'
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start:  $(date)"
echo "Node:   ${SLURM_NODELIST:-local}"

# === PATHS ===
HOME_DIR=/mnt/evafs/faculty/home/bpiotrowski
ABLATION_DIR=$HOME_DIR/DETR/ablation
VARIANT_DIR=$ABLATION_DIR/$VARIANT
FULL_DATASET_TAR=$HOME_DIR/datasets_20250606.tar.gz

# Output dirs (osobne od ablacji fine-tune, zeby nie nadpisac)
CHECKPOINT_DIR=$HOME_DIR/DETR/scratch_20k_checkpoints/$VARIANT
OUTPUT_DIR=$HOME_DIR/DETR/scratch_20k_output/$VARIANT
BEST_MODEL_DIR=$HOME_DIR/DETR/scratch_20k_best/$VARIANT

mkdir -p "$CHECKPOINT_DIR" "$OUTPUT_DIR" "$BEST_MODEL_DIR" logs

# =============================================================================
# GUARD: jesli trening juz ukonczony -> exit 0 PRZED kosztownym rozpakowaniem.
#   - final_checkpoint.pth  => trening zakonczony (rowniez przez early-stop)
#   - max(checkpoint_epoch_N) >= EPOCHS => osiagnieto budzet epok
# =============================================================================
latest_epoch() {
    # find (nie ls-glob!) -> exit 0 nawet gdy katalog pusty (bez set -e crash)
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

# === VALIDATION (BRAK wymogu checkpoint_epoch_170 - trenujemy od zera) ===
for f in "$VARIANT_DIR/annotations_train.json" "$VARIANT_DIR/annotations_val.json" "$FULL_DATASET_TAR"; do
    if [[ ! -f "$f" ]]; then echo "ERROR: brak pliku: $f"; exit 1; fi
done

# === TMPDIR (node-local) - wybor przez TEST ZAPISYWALNOSCI + miejsca ===
# /raid jest pisalny TYLKO na hopper/hopper-2; na dgx pisalny jest /tmp (md0, 1.8T).
# Probujemy kandydatow po kolei; pierwszy ZAPISYWALNY z >50GB wygrywa.
JOBSUB="${USER}_${SLURM_JOB_ID:-$$}_scratch20k_${VARIANT}"
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

# === CLEANUP TRAP - USTAWIONY OD RAZU (kazda sciezka wyjscia kasuje TMPDIR!) ===
# Gwarantuje, ze node-local scratch znika nawet przy bledzie w trakcie ekstrakcji
# (ponizej, po zdefiniowaniu prune_checkpoints, trap jest rozszerzany o pruning).
trap 'echo "[$(date)] CLEANUP: rm -rf $TMPDIR"; rm -rf "$TMPDIR" 2>/dev/null || true' EXIT INT TERM
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
#   Zachowaj: wielokrotnosci MILESTONE (np. 50,100,..,400) + 5 najnowszych.
#   find_latest_checkpoint zawsze widzi najnowszy -> resume bezpieczny.
prune_checkpoints() {
    local eps recent e
    eps=$(find "$CHECKPOINT_DIR" -maxdepth 1 -name 'checkpoint_epoch_*.pth' 2>/dev/null \
          | sed -E 's/.*checkpoint_epoch_([0-9]+)\.pth/\1/' | sort -n)
    [[ -z "$eps" ]] && return 0
    recent=$(echo "$eps" | tail -5)
    for e in $eps; do
        (( e % MILESTONE == 0 )) && continue              # kamien milowy -> zostaje
        echo "$recent" | grep -qx "$e" && continue         # najnowszy -> zostaje
        rm -f "$CHECKPOINT_DIR/checkpoint_epoch_${e}.pth"
    done
    return 0
}

trap 'echo "[$(date)] EXIT trap (scratch20k chain $VARIANT)"; prune_checkpoints; echo "Cleaning TMPDIR ($TMPDIR)..."; rm -rf "$TMPDIR"' EXIT INT TERM
(while true; do sleep 600; prune_checkpoints; done) &
PRUNE_PID=$!

# === TRAINING ===
echo ""
echo "=== TRAINING CONFIGURATION ==="
echo "  Variant:    $VARIANT ($VARIANT_NAME)  [curated 20k, video-level]"
echo "  Data:       $VARIANT_DIR (annotations_train/val.json)"
echo "  Init:       facebook/detr-resnet-50 (COCO)  | resume_training ON (ep0 jesli pusto)"
echo "  Resume od:  ${LAT:-ep0 (od zera)}"
echo "  GPUs:       ${NGPUS}x  | batch/GPU 8 (eff $((NGPUS * 8)))"
echo "  Epochs:     0 -> $EPOCHS  (patience 60, save co $SAVE_INTERVAL)"
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
echo "  DETR FROM SCRATCH 20k CHAIN ($VARIANT) -- ogniwo zakonczone  $(date)"
echo "  Ostatni checkpoint: ep${LAT_END:-?} / $EPOCHS"
echo "  Checkpoints:        $CHECKPOINT_DIR/"
echo "  Best model:         $BEST_MODEL_DIR/"
echo "  (Kolejne ogniwo (jesli w kolejce) wznowi automatycznie z ep${LAT_END:-?}.)"
echo "============================================================"
exit $TRAIN_EXIT
