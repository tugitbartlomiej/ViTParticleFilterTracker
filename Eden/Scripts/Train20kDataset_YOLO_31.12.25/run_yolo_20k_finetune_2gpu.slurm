#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=yolo_20k_ft
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/YOLO_20k
#SBATCH --output=logs/yolo_20k_finetune_%j.log
#SBATCH --error=logs/yolo_20k_finetune_%j.log
#SBATCH --open-mode=append

set -euo pipefail
echo '============================================================'
echo '  YOLO FINE-TUNING ON 20K SELECTED DATASET'
echo '  Node: DGX-4 (2x A100 80GB)'
echo '  Resume from epoch170.pt'
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
COCO_JSON=/mnt/evafs/faculty/home/bpiotrowski/merged_20k_annotations.json

echo ""
echo "=== EXTRACTING 20K DATASET ==="
echo "Archive: $ARCHIVE"
echo "Extracting to: $TMPDIR/20k_dataset"

mkdir -p "$TMPDIR/20k_dataset/images"
tar -xf "$ARCHIVE" -C "$TMPDIR/20k_dataset"

# Move images to images/ subfolder if extracted flat
cd "$TMPDIR/20k_dataset"
if [[ -d "images" ]]; then
    echo "Images already in images/ subfolder"
else
    mkdir -p images
    mv *.jpg images/ 2>/dev/null || true
fi

IMAGES_DIR="$TMPDIR/20k_dataset/images"
IMAGE_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -name "*.jpg" | wc -l)

echo ""
echo "=== DATASET VERIFICATION ==="
echo "  Images dir: $IMAGES_DIR"
echo "  Image count: $IMAGE_COUNT"

if [[ "$IMAGE_COUNT" -lt 1000 ]]; then
    echo "ERROR: Too few images found ($IMAGE_COUNT). Expected ~20000."
    exit 1
fi

# === CONVERT COCO TO YOLO FORMAT ===
echo ""
echo "=== CONVERTING COCO TO YOLO FORMAT ==="
YOLO_DATASET_DIR="$TMPDIR/20k_dataset_yolo"

python3 -c "
import json
import os
from pathlib import Path
import yaml

coco_json = '$COCO_JSON'
images_dir = Path('$IMAGES_DIR')
output_dir = Path('$YOLO_DATASET_DIR')

print(f'Loading COCO annotations: {coco_json}')
with open(coco_json, 'r') as f:
    coco_data = json.load(f)

# Create directories
labels_dir = output_dir / 'labels' / 'train'
images_out_dir = output_dir / 'images' / 'train'
labels_dir.mkdir(parents=True, exist_ok=True)
images_out_dir.mkdir(parents=True, exist_ok=True)

# Build mappings
image_id_to_info = {img['id']: img for img in coco_data['images']}
image_id_to_anns = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id not in image_id_to_anns:
        image_id_to_anns[img_id] = []
    image_id_to_anns[img_id].append(ann)

# Category mapping (single class: tooltip -> 0)
categories = coco_data.get('categories', [{'id': 1, 'name': 'tooltip'}])
cat_id_to_yolo = {cat['id']: idx for idx, cat in enumerate(categories)}
class_names = [cat['name'] for cat in categories]

print(f'Categories: {class_names}')
print(f'Images: {len(coco_data[\"images\"])}')
print(f'Annotations: {len(coco_data[\"annotations\"])}')

# Convert
converted = 0
for img_info in coco_data['images']:
    img_id = img_info['id']
    img_w, img_h = img_info['width'], img_info['height']
    file_name = img_info['file_name']

    # Create label file
    label_file = labels_dir / (Path(file_name).stem + '.txt')

    with open(label_file, 'w') as f:
        for ann in image_id_to_anns.get(img_id, []):
            x, y, w, h = ann['bbox']
            # COCO to YOLO: center coords, normalized
            x_center = (x + w/2) / img_w
            y_center = (y + h/2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Clip to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            class_id = cat_id_to_yolo.get(ann['category_id'], 0)
            f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n')

    # Symlink image
    src = images_dir / file_name
    dst = images_out_dir / file_name
    if src.exists() and not dst.exists():
        os.symlink(src, dst)

    converted += 1

print(f'Converted: {converted} images')

# Create dataset.yaml
yaml_content = {
    'path': str(output_dir),
    'train': 'images/train',
    'val': 'images/train',
    'names': {i: name for i, name in enumerate(class_names)}
}

yaml_path = output_dir / 'dataset.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f)

print(f'Dataset YAML: {yaml_path}')
"

DATASET_YAML="$YOLO_DATASET_DIR/dataset.yaml"

if [[ ! -f "$DATASET_YAML" ]]; then
    echo "ERROR: Failed to create dataset.yaml"
    exit 1
fi
echo "YOLO dataset ready: $DATASET_YAML"

# === CHECKPOINT CONFIGURATION ===
ARCHIVE_DIR=/mnt/evafs/faculty/home/bpiotrowski/YOLO_Checkpoints
YOLO_CHECKPOINT=/mnt/evafs/faculty/home/bpiotrowski/YOLO_Checkpoints/epoch170.pt
PROJECT_DIR=/mnt/evafs/faculty/home/bpiotrowski/YOLO_20k
CHECKPOINT_DIR=$PROJECT_DIR/ckpt_yolo_20k
BEST_MODEL_DIR=$PROJECT_DIR/best_yolo_20k
OUTPUT_DIR=$PROJECT_DIR/train_out_20k

mkdir -p "$PROJECT_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$BEST_MODEL_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_DIR/logs"

echo ""
echo "=== CHECKPOINT CONFIGURATION ==="
echo "  Resume from: $YOLO_CHECKPOINT"
echo "  Project dir: $PROJECT_DIR"
echo "  Checkpoint dir: $CHECKPOINT_DIR"

# Check if checkpoint exists
if [[ ! -f "$YOLO_CHECKPOINT" ]]; then
    echo "WARNING: epoch170.pt not found at $YOLO_CHECKPOINT"
    echo "Checking alternative locations..."
    ALT_CHECKPOINT=/mnt/evafs/faculty/home/bpiotrowski/YOLO_EDEN_TRAIN/exp/weights/epoch170.pt
    if [[ -f "$ALT_CHECKPOINT" ]]; then
        echo "Found at: $ALT_CHECKPOINT"
        YOLO_CHECKPOINT=$ALT_CHECKPOINT
        RESUME_ARGS="--resume --resume_path $YOLO_CHECKPOINT"
    else
        echo "Will start from pretrained YOLOv8m"
        RESUME_ARGS=""
    fi
else
    RESUME_ARGS="--resume --resume_path $YOLO_CHECKPOINT"
fi

# === CHECKPOINT SYNC FUNCTIONS ===
sync_checkpoints() {
    echo "[$(date)] Syncing checkpoints to archive..."
    if [[ -d "$OUTPUT_DIR/exp/weights" ]]; then
        rsync -av --update "$OUTPUT_DIR/exp/weights/" "$ARCHIVE_DIR/20k_finetune/" 2>&1 | grep -E "(epoch|sent|total)" || true
    fi
    echo "[$(date)] Sync completed"
}

# Trap for final sync
trap 'echo "EXIT trap triggered"; sync_checkpoints' EXIT INT TERM

# Background sync every 10 minutes
(
    while true; do
        sleep 600
        sync_checkpoints
    done
) &
SYNC_PID=$!
echo "Background checkpoint sync started (PID: $SYNC_PID)"

# === TRAINING CONFIGURATION ===
echo ""
echo "=== TRAINING CONFIGURATION (DGX-4, 2x A100) ==="
echo "  GPUs: 2x A100 (80GB each)"
echo "  Epochs: 300 (fine-tuning on 20k dataset)"
echo "  Batch size per GPU: 32"
echo "  Effective batch: 64 (2 GPU x 32)"
echo "  Learning rate: 0.0008 (adjusted for smaller batch)"
echo "  LR Scheduler: Cosine Annealing"
echo "  Dataset: 20k selected images"
echo ""

# === START TRAINING ===
cd "$PROJECT_DIR"

python3 yolo_train_20k_finetune.py \
    --dataset_yaml_path "$DATASET_YAML" \
    --project_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --best_model_dir "$BEST_MODEL_DIR" \
    --model_size m \
    --epochs 300 \
    --batch_size 32 \
    --imgsz 640 \
    --learning_rate 0.0008 \
    --workers 8 \
    --device "0,1" \
    --save_period 5 \
    --patience 30 \
    --freeze_backbone 0 \
    $RESUME_ARGS \
    --verbose

TRAIN_EXIT_CODE=$?

echo ""
echo "=== TRAINING FINISHED ==="
echo "Exit code: $TRAIN_EXIT_CODE"

# Stop background sync
kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true

# Final sync
echo "Performing final checkpoint sync..."
sync_checkpoints

# Copy best model to permanent location
if [[ -d "$BEST_MODEL_DIR" ]] && [[ -n "$(ls -A $BEST_MODEL_DIR 2>/dev/null)" ]]; then
    echo "Copying best model to archive..."
    rsync -av "$BEST_MODEL_DIR/" "$ARCHIVE_DIR/best_yolo_20k/"
fi

echo ""
echo "============================================================"
echo "  JOB COMPLETED (DGX-4)"
echo "  Finished at: $(date)"
echo "  Checkpoints archived to: $ARCHIVE_DIR/20k_finetune/"
echo "  Best model at: $ARCHIVE_DIR/best_yolo_20k/"
echo "============================================================"

exit $TRAIN_EXIT_CODE
