#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=yolo_abl_%a
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo/ablation
#SBATCH --output=logs/yolo_abl_%a_%j.log
#SBATCH --error=logs/yolo_abl_%a_%j.log
#SBATCH --open-mode=append

# =============================================================================
# YOLO Ablation Study — Training on Hopper (4xH100 lub 7xH100 jezeli dostepne)
#
# Match produkcyjnego YOLO 20k fine-tune (Eden/Scripts/Train20kDataset_YOLO_31.12.25):
#   cos_lr=False, optimizer=SGD, lr0=0.0001, batch=32, epochs=160, imgsz=640
#
# Variants:
#   ye_a: YOLO-EL2N -> 20k -> YOLOv8 fine-tune
#   ye_b: YOLO-EL2N -> 20k -> DETR fine-tune (uzyj detr_train_ablation.py zamiast tego)
#   ye_c: DETR-EL2N -> 20k -> YOLOv8 fine-tune
#
# Tylko ye_a i ye_c uzywaja tego skryptu (oba target=YOLO).
#
# Usage:
#   sbatch --array=1-2 run_yolo_ablation_variant.slurm     # ye_a + ye_c
#   VARIANT=ye_a sbatch run_yolo_ablation_variant.slurm    # single
# =============================================================================

set -euo pipefail

# === VARIANT RESOLUTION ===
VARIANTS=(ye_a ye_c)

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    IDX=$((SLURM_ARRAY_TASK_ID - 1))
    VARIANT="${VARIANTS[$IDX]}"
elif [[ -z "${VARIANT:-}" ]]; then
    echo "ERROR: musisz ustawic VARIANT lub uzyc --array" >&2
    exit 1
fi

echo "================================================================"
echo "YOLO Ablation: VARIANT=$VARIANT  JOB=${SLURM_JOB_ID:-local}"
echo "================================================================"

# === PATHS (TODO: dostosuj do faktycznych sciezek na Edenie) ===
ABLATION_ROOT=/mnt/evafs/faculty/home/bpiotrowski/Yolo/ablation
WEIGHTS_EP170=/mnt/evafs/faculty/home/bpiotrowski/Yolo/checkpoints/yolov8m_ep170.pt
DATASET_YAML=${ABLATION_ROOT}/${VARIANT}/dataset.yaml
OUTPUT_DIR=${ABLATION_ROOT}/${VARIANT}/runs

mkdir -p "$OUTPUT_DIR" "$ABLATION_ROOT/logs"

# === CONDA / VENV ===
# TODO: source activate yolo_env (lub odpowiednik na Edenie)

# === TRAINING ===
python /mnt/evafs/faculty/home/bpiotrowski/Yolo/ablation/scripts/yolo_train_ablation.py \
    --variant "$VARIANT" \
    --weights "$WEIGHTS_EP170" \
    --data "$DATASET_YAML" \
    --epochs 160 \
    --batch 32 \
    --lr0 0.0001 \
    --imgsz 640 \
    --device 0,1,2,3 \
    --output "$OUTPUT_DIR"

echo "Done variant=$VARIANT at $(date -Iseconds)"
