#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=yolo_90k_vs_scratch
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune
#SBATCH --output=logs/yolo_90k_vs_scratch_%j.log
#SBATCH --error=logs/yolo_90k_vs_scratch_%j.log
#SBATCH --open-mode=append

# =============================================================================
# YOLOv8 - trening OD ZERA (epoch 0 -> EPOCHS) na PELNYM poolu 91k pod VIDEO-LEVEL
# splitem (35/8/7 wideo, seed 42 — TEN SAM co DETR i ablacje).
#
# PARTYCJA: long / dgx (A100), limit 5 dni. YOLO jest szybki — caly trening (~150-200
#   epok na 71k) miesci sie w JEDNYM 5-dniowym jobie (szac. ~1-2 dni na A100x4),
#   wiec NIE potrzebuje lancucha. Jesli mimo to nie zdazy: last.pt jest w
#   $PROJECT_ROOT/train_out_scratch_90k_vs/exp/weights/ (na home) -> wznow recznie
#   natywnym Ultralytics resume (YOLO('last.pt').train(resume=True)).
#
# POTOK (TMP -> rozpakowanie -> video-split -> konwersja YOLO):
#   1. rozpakuj pool 91k z tar -> $TMPDIR/pool_images (node-local; na dgx /raid
#      niezapisywalny -> /tmp md0 1.8T lub /dev/shm),
#   2. konwersja COCO->YOLO przez coco_to_yolo_videosplit.py, ktora ROUTUJE obrazy do
#      train/val wg GOTOWYCH annotations_train/val.json (video-split JUZ zaszyty -> ZERO
#      random shuffle) i przelicza bbox COCO[x,y,w,h]px -> YOLO[xc,yc,w,h] znormalizowane.
#      Konwersja przetestowana: yolo/test_coco_to_yolo_videosplit.py (27 asercji PASS).
#
# Init wag: yolov8l.pt (COCO) — BEZ --pretrained_path => nowy trening od ep0.
#
# WYMAGANE NAJPIERW: prepare_fullpool_videosplit.py -> ~/DETR/fullpool_videosplit/
#
# Usage:
#   sbatch run_yolo_90k_from_scratch_videosplit.slurm
#   EPOCHS=190 sbatch run_yolo_90k_from_scratch_videosplit.slurm
# =============================================================================

set -euo pipefail

EPOCHS="${EPOCHS:-200}"
MODEL_SIZE="${MODEL_SIZE:-l}"

echo "============================================================"
echo "  YOLO FROM SCRATCH on FULL 91k | VIDEO-LEVEL split | long/dgx A100 | yolov8${MODEL_SIZE}.pt (COCO) | 0 -> $EPOCHS"
echo "  DATE: $(date)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   ${SLURM_NODELIST:-local}"
echo ""

# === PROJECT DIRECTORIES ===
PROJECT_ROOT=/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune
HOME_DIR=/mnt/evafs/faculty/home/bpiotrowski
SPLIT_DIR=$HOME_DIR/DETR/fullpool_videosplit          # <- prepare_fullpool_videosplit.py
FULL_DATASET_TAR=$HOME_DIR/datasets_20250606.tar.gz
CONVERT_SCRIPT="${CONVERT_SCRIPT:-$HOME_DIR/DETR/Scripts/Train90kFromScratch_VideoSplit_2026-06-24/yolo/coco_to_yolo_videosplit.py}"
mkdir -p "$PROJECT_ROOT/logs"

# === TMPDIR (na dgx /raid NIE jest zapisywalny -> /tmp (md0 1.8T) lub /dev/shm) ===
JOBSUB="${USER}_${SLURM_JOB_ID:-$$}_yolo90k_vs"
NEED_KB=52428800   # ~50 GB
TMPDIR=""
for cand in "${SLURM_TMPDIR:-}/$JOBSUB" "/tmp/$JOBSUB" "/dev/shm/$JOBSUB"; do
    [[ "$cand" == "/$JOBSUB" ]] && continue   # pusty SLURM_TMPDIR
    if mkdir -p "$cand" 2>/dev/null; then
        avail=$(df --output=avail "$cand" 2>/dev/null | tail -1 | tr -d ' ')
        if [[ -n "$avail" && "$avail" -gt "$NEED_KB" ]]; then TMPDIR="$cand"; break; fi
        rmdir "$cand" 2>/dev/null || true
    fi
done
if [[ -z "$TMPDIR" ]]; then echo "ERROR: brak scratch >50GB (SLURM_TMPDIR /tmp /dev/shm)"; exit 1; fi
trap 'echo "[CLEANUP] rm -rf $TMPDIR"; rm -rf "$TMPDIR" 2>/dev/null || true' EXIT INT TERM
echo "TMPDIR: $TMPDIR (free: $(df -h "$TMPDIR" | tail -1 | awk '{print $4}'))"

# === CONDA ===
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# === INPUT FILES ===
for f in "$SPLIT_DIR/annotations_train.json" "$SPLIT_DIR/annotations_val.json" "$FULL_DATASET_TAR" "$CONVERT_SCRIPT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: brak pliku: $f"
        echo "       (split: uruchom prepare_fullpool_videosplit.py; skrypt konwersji: wgraj folder na Eden)"
        exit 1
    fi
done

WRAPPER="$PROJECT_ROOT/yolo_train_20k_finetune_v2.py"
if [[ ! -f "$WRAPPER" ]]; then
    echo "ERROR: brak $WRAPPER (skopiuj z Eden/Scripts/Train20kDataset_YOLO_31.12.25/)"
    exit 1
fi

# === EXTRACT 91K POOL -> node-local ===
echo ""
echo "=== EXTRACTING 91K POOL TO TMPDIR (~5-10 min) ==="
mkdir -p "$TMPDIR/pool_images"
tar -xzf "$FULL_DATASET_TAR" \
    --strip-components=3 \
    -C "$TMPDIR/pool_images" \
    "datasets/DETR_augmented_dataset_20250218/images/" \
    2>/dev/null || {
    echo "Strip-components extraction failed, fallback..."
    tar -xzf "$FULL_DATASET_TAR" -C "$TMPDIR/"
    IMG_FOUND=$(find "$TMPDIR" -type d -name "images" | head -1)
    if [[ -n "$IMG_FOUND" && "$IMG_FOUND" != "$TMPDIR/pool_images" ]]; then
        mv "$IMG_FOUND"/* "$TMPDIR/pool_images/" 2>/dev/null || true
    fi
}
POOL_IMAGES="$TMPDIR/pool_images"
POOL_COUNT=$(find "$POOL_IMAGES" -maxdepth 1 -name "*.jpg" | wc -l)
echo "Pool images: $POOL_COUNT"
if [[ "$POOL_COUNT" -lt 10000 ]]; then echo "ERROR: za malo obrazow ($POOL_COUNT)"; exit 1; fi

# === CONVERT COCO -> YOLO  (VIDEO-LEVEL split; przetestowany skrypt) ===
echo ""
echo "=== CONVERTING COCO TO YOLO (VIDEO-LEVEL, BEZ random shuffle) ==="
YOLO_DATASET_DIR="$TMPDIR/yolo_dataset"
python3 "$CONVERT_SCRIPT" \
    --train-coco "$SPLIT_DIR/annotations_train.json" \
    --val-coco "$SPLIT_DIR/annotations_val.json" \
    --pool-images "$POOL_IMAGES" \
    --output "$YOLO_DATASET_DIR" \
    --link-mode symlink

DATASET_YAML="$YOLO_DATASET_DIR/dataset.yaml"
if [[ ! -f "$DATASET_YAML" ]]; then echo "ERROR: brak dataset.yaml"; exit 1; fi
echo "YOLO dataset ready"; cat "$DATASET_YAML"

# === DEVICE ===
NGPUS=${SLURM_GPUS_ON_NODE:-4}
if echo "$NGPUS" | grep -q ','; then NGPUS=$(echo "$NGPUS" | tr ',' '\n' | wc -l); fi
if [[ "$NGPUS" -lt 1 ]] || [[ "$NGPUS" -gt 12 ]]; then NGPUS=4; fi
DEVICE=$(seq -s, 0 $((NGPUS - 1)))

# === TRAINING ===
echo ""
echo "============================================================"
echo "  TRAINING CONFIGURATION (FROM SCRATCH, FULL 91k, VIDEO-LEVEL, long/dgx A100)"
echo "  GPUs:        ${NGPUS}x ($DEVICE)"
echo "  Model:       yolov8${MODEL_SIZE} (COCO init, BEZ pretrained_path)"
echo "  Epochs:      0 -> $EPOCHS  (patience 100)"
echo "  Batch:       32  imgsz 640"
echo "  LR:          lr0=0.01  lrf=0.01  (full-training, NIE fine-tune 0.0001)"
echo "============================================================"
echo ""

cd "$PROJECT_ROOT"

# UWAGA: celowo BEZ --pretrained_path => wrapper: YOLO('yolov8${MODEL_SIZE}.pt'), nowy trening.
python3 yolo_train_20k_finetune_v2.py \
    --dataset_yaml_path "$DATASET_YAML" \
    --project_dir "$PROJECT_ROOT/train_out_scratch_90k_vs" \
    --checkpoint_dir "$PROJECT_ROOT/ckpt_scratch_90k_vs" \
    --best_model_dir "$PROJECT_ROOT/best_scratch_90k_vs" \
    --model_size "$MODEL_SIZE" \
    --epochs "$EPOCHS" \
    --batch_size 32 \
    --imgsz 640 \
    --learning_rate 0.01 \
    --lrf 0.01 \
    --workers 12 \
    --device "$DEVICE" \
    --save_period 10 \
    --patience 100 \
    --freeze_backbone 0 \
    --verbose

EXIT_CODE=$?
echo ""
echo "============================================================"
echo "  FINISHED (exit: $EXIT_CODE)  $(date)"
echo "============================================================"

# === KONCOWE CHECKPOINTY ===
if [[ -d "$PROJECT_ROOT/train_out_scratch_90k_vs/exp/weights" ]]; then
    FINAL_DIR="$PROJECT_ROOT/final_scratch_90k_vs"
    mkdir -p "$FINAL_DIR"
    cp "$PROJECT_ROOT/train_out_scratch_90k_vs/exp/weights/"*.pt "$FINAL_DIR/" 2>/dev/null || true
    echo "Final checkpoints -> $FINAL_DIR"
fi

exit $EXIT_CODE
