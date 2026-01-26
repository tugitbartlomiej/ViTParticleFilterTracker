#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --gres=gpu:7
#SBATCH --mem=150G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=yolo_20k_lr0001
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune
#SBATCH --output=logs/yolo_20k_ep170_lr0001_%j.log
#SBATCH --error=logs/yolo_20k_ep170_lr0001_%j.log
#SBATCH --open-mode=append

set -eu
echo "============================================================"
echo "  YOLO 20K: FINE-TUNE FROM EPOCH 170 -> 500"
echo "  CORRECTED LR: 0.0001 (not 0.001!)"
echo "  DATE: $(date)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""
echo "LESSON LEARNED FROM PREVIOUS RUN:"
echo "  - Previous lr0=0.001 was TOO HIGH"
echo "  - Model achieved best results at epoch 1, then degraded"
echo "  - EarlyStopping triggered after 51 epochs"
echo ""
echo "NEW STRATEGY:"
echo "  - lr0=0.0001 (10x smaller than before)"
echo "  - lrf=0.1 -> final LR = 0.00001"
echo "  - patience=100 (more patience for subtle improvements)"
echo ""

# === PROJECT DIRECTORIES ===
PROJECT_ROOT=/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune
mkdir -p "$PROJECT_ROOT/logs"

# === SETUP TMPDIR ===
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR: $TMPDIR"

# === CONDA ACTIVATION ===
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# === FIND EPOCH 170 CHECKPOINT ===
echo ""
echo "=== LOOKING FOR EPOCH 170 CHECKPOINT ==="

POSSIBLE_WEIGHTS=(
    "$PROJECT_ROOT/epoch170.pt"
    "$PROJECT_ROOT/train_out_5gpu_hopper/exp/weights/last.pt"
)

PRETRAINED_WEIGHTS=""
for WEIGHTS in "${POSSIBLE_WEIGHTS[@]}"; do
    if [[ -f "$WEIGHTS" ]]; then
        echo "Found checkpoint: $WEIGHTS"
        ls -la "$WEIGHTS"
        PRETRAINED_WEIGHTS="$WEIGHTS"
        break
    fi
done

if [[ -z "$PRETRAINED_WEIGHTS" ]]; then
    echo "ERROR: No checkpoint found!"
    exit 1
fi

echo ""
echo "Using pretrained weights: $PRETRAINED_WEIGHTS"

# === INPUT FILES ===
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/20kSelectedImages.tar
COCO_JSON=/mnt/evafs/faculty/home/bpiotrowski/merged_20k_annotations_fixed.json

# === EXTRACT 20K DATASET ===
echo ""
echo "=== EXTRACTING 20K DATASET ==="
EXTRACT_DIR="$TMPDIR/20k_dataset"
mkdir -p "$EXTRACT_DIR"
tar -xf "$ARCHIVE" -C "$EXTRACT_DIR"

# Find images
if [[ -d "$EXTRACT_DIR/images" ]] && [[ $(find "$EXTRACT_DIR/images" -maxdepth 1 -name "*.jpg" 2>/dev/null | wc -l) -gt 0 ]]; then
    IMAGES_DIR="$EXTRACT_DIR/images"
elif [[ $(find "$EXTRACT_DIR" -maxdepth 1 -name "*.jpg" 2>/dev/null | wc -l) -gt 0 ]]; then
    IMAGES_DIR="$EXTRACT_DIR"
else
    echo "ERROR: No images found!"
    exit 1
fi

IMAGE_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -name "*.jpg" | wc -l)
echo "Images: $IMAGE_COUNT"

# === CONVERT COCO TO YOLO FORMAT ===
echo ""
echo "=== CONVERTING COCO TO YOLO FORMAT ==="
YOLO_DATASET_DIR="$TMPDIR/yolo_dataset"

export COCO_JSON IMAGES_DIR YOLO_DATASET_DIR
python3 << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path
import yaml
import random

coco_json = os.environ.get('COCO_JSON', '')
images_dir = Path(os.environ.get('IMAGES_DIR', ''))
output_dir = Path(os.environ.get('YOLO_DATASET_DIR', ''))

print(f'Loading COCO annotations: {coco_json}')
with open(coco_json, 'r') as f:
    coco_data = json.load(f)

print(f'  Images in JSON: {len(coco_data["images"])}')
print(f'  Annotations: {len(coco_data["annotations"])}')

# Create directories
labels_train_dir = output_dir / 'labels' / 'train'
labels_val_dir = output_dir / 'labels' / 'val'
images_train_dir = output_dir / 'images' / 'train'
images_val_dir = output_dir / 'images' / 'val'

for d in [labels_train_dir, labels_val_dir, images_train_dir, images_val_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Build mappings
image_id_to_info = {img['id']: img for img in coco_data['images']}
image_id_to_anns = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id not in image_id_to_anns:
        image_id_to_anns[img_id] = []
    image_id_to_anns[img_id].append(ann)

categories = coco_data.get('categories', [{'id': 1, 'name': 'tooltip'}])
cat_id_to_yolo = {cat['id']: idx for idx, cat in enumerate(categories)}
class_names = [cat['name'] for cat in categories]

# Split 90% train, 10% val
random.seed(42)
image_ids = list(image_id_to_info.keys())
random.shuffle(image_ids)
split_idx = int(len(image_ids) * 0.9)
train_ids = set(image_ids[:split_idx])

converted = 0
for img_id, img_info in image_id_to_info.items():
    img_w = img_info['width']
    img_h = img_info['height']
    file_name = img_info['file_name']

    is_train = img_id in train_ids
    labels_dir = labels_train_dir if is_train else labels_val_dir
    images_out_dir = images_train_dir if is_train else images_val_dir

    label_file = labels_dir / (Path(file_name).stem + '.txt')

    with open(label_file, 'w') as f:
        for ann in image_id_to_anns.get(img_id, []):
            x, y, w, h = ann['bbox']
            x_center = max(0, min(1, (x + w/2) / img_w))
            y_center = max(0, min(1, (y + h/2) / img_h))
            w_norm = max(0, min(1, w / img_w))
            h_norm = max(0, min(1, h / img_h))
            class_id = cat_id_to_yolo.get(ann['category_id'], 0)
            f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n')

    src = images_dir / file_name
    dst = images_out_dir / file_name
    if src.exists() and not dst.exists():
        try:
            os.symlink(src, dst)
        except:
            import shutil
            shutil.copy2(src, dst)
    converted += 1

print(f'Converted {converted} images')

yaml_content = {
    'path': str(output_dir),
    'train': 'images/train',
    'val': 'images/val',
    'names': {i: name for i, name in enumerate(class_names)},
    'nc': len(class_names)
}

yaml_path = output_dir / 'dataset.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f, default_flow_style=False)

print(f'Dataset YAML: {yaml_path}')
PYTHON_SCRIPT

DATASET_YAML="$YOLO_DATASET_DIR/dataset.yaml"

if [[ ! -f "$DATASET_YAML" ]]; then
    echo "ERROR: Failed to create dataset.yaml"
    exit 1
fi

echo "YOLO dataset ready"
cat "$DATASET_YAML"

# === TRAINING ===
echo ""
echo "============================================================"
echo "  TRAINING CONFIGURATION - CORRECTED LR"
echo "============================================================"
echo "  GPUs: 7x H100"
echo "  Checkpoint: epoch 170"
echo "  NEW epochs: 330 (to reach total 500)"
echo "  Batch size: 35 (5 per GPU x 7)"
echo ""
echo "  LEARNING RATE (CORRECTED):"
echo "    lr0 = 0.0001 (was 0.001 - TOO HIGH!)"
echo "    lrf = 0.1"
echo "    Final LR = 0.00001"
echo ""
echo "  Optimizer: SGD (explicit)"
echo "  Patience: 100 epochs"
echo "============================================================"
echo ""

cd "$PROJECT_ROOT"

# Use yolo_train_20k_finetune_v2.py with CORRECTED learning rate
python3 yolo_train_20k_finetune_v2.py \
    --dataset_yaml_path $DATASET_YAML \
    --project_dir $PROJECT_ROOT/train_out_ep170_lr0001 \
    --checkpoint_dir $PROJECT_ROOT/ckpt_ep170_lr0001 \
    --best_model_dir $PROJECT_ROOT/best_ep170_lr0001 \
    --model_size l \
    --epochs 330 \
    --batch_size 35 \
    --imgsz 640 \
    --learning_rate 0.0001 \
    --lrf 0.1 \
    --workers 12 \
    --device 0,1,2,3,4,5,6 \
    --save_period 10 \
    --patience 100 \
    --freeze_backbone 0 \
    --verbose \
    --pretrained_path $PRETRAINED_WEIGHTS

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  FINISHED (exit: $EXIT_CODE)"
echo "  Time: $(date)"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - $PROJECT_ROOT/train_out_ep170_lr0001/"
echo "  - $PROJECT_ROOT/ckpt_ep170_lr0001/"
echo "  - $PROJECT_ROOT/best_ep170_lr0001/"
echo ""

# Copy final checkpoints
if [[ -d "$PROJECT_ROOT/train_out_ep170_lr0001/exp/weights" ]]; then
    FINAL_DIR="$PROJECT_ROOT/final_ep170_lr0001"
    mkdir -p "$FINAL_DIR"
    cp "$PROJECT_ROOT/train_out_ep170_lr0001/exp/weights/"*.pt "$FINAL_DIR/" 2>/dev/null || true
    echo "Final checkpoints copied to: $FINAL_DIR"
fi

exit $EXIT_CODE
