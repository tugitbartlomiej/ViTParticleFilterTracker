#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=yolo_20k_dgx1
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune
#SBATCH --output=logs/yolo_20k_finetune_%j.log
#SBATCH --error=logs/yolo_20k_finetune_%j.log
#SBATCH --open-mode=append

set -eu
echo '============================================================'
echo '  YOLO FINE-TUNING ON 20K SELECTED DATASET (DGX-1, 8 GPUs)'
echo '  Dataset: 20k intelligently selected images'
echo '  Resume: from epoch170.pt checkpoint (if available)'
echo '============================================================'
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURM_NODELIST"

# === PROJECT DIRECTORIES ===
PROJECT_ROOT=/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/ckpt"
mkdir -p "$PROJECT_ROOT/best"
mkdir -p "$PROJECT_ROOT/train_out"

# === SETUP TMPDIR ===
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR: $TMPDIR"

# === CONDA ACTIVATION ===
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# === INPUT FILES ===
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/20kSelectedImages.tar
COCO_JSON=/mnt/evafs/faculty/home/bpiotrowski/merged_20k_annotations_fixed.json

echo ""
echo "=== INPUT FILES ==="
echo "  Archive: $ARCHIVE"
echo "  COCO JSON: $COCO_JSON"

# Check files exist
if [[ ! -f "$ARCHIVE" ]]; then
    echo "ERROR: Archive not found: $ARCHIVE"
    exit 1
fi

if [[ ! -f "$COCO_JSON" ]]; then
    echo "ERROR: COCO JSON not found: $COCO_JSON"
    exit 1
fi

# === EXTRACT 20K DATASET ===
echo ""
echo "=== EXTRACTING 20K DATASET ==="
EXTRACT_DIR="$TMPDIR/20k_dataset"
mkdir -p "$EXTRACT_DIR"

echo "Extracting to: $EXTRACT_DIR"
tar -xf "$ARCHIVE" -C "$EXTRACT_DIR"

# Check what was extracted
echo "Contents after extraction:"
ls -la "$EXTRACT_DIR" | head -10

# Determine where images are
# They might be flat in EXTRACT_DIR or in images/ subfolder
if [[ -d "$EXTRACT_DIR/images" ]] && [[ $(find "$EXTRACT_DIR/images" -maxdepth 1 -name "*.jpg" 2>/dev/null | wc -l) -gt 0 ]]; then
    IMAGES_DIR="$EXTRACT_DIR/images"
    echo "Images found in: $IMAGES_DIR"
elif [[ $(find "$EXTRACT_DIR" -maxdepth 1 -name "*.jpg" 2>/dev/null | wc -l) -gt 0 ]]; then
    # Images extracted flat - use directly without moving (faster)
    echo "Images extracted flat, using directly from: $EXTRACT_DIR"
    IMAGES_DIR="$EXTRACT_DIR"
else
    echo "ERROR: No images found after extraction!"
    echo "Directory contents:"
    find "$EXTRACT_DIR" -type f -name "*.jpg" | head -20
    exit 1
fi

# Count images
IMAGE_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -name "*.jpg" | wc -l)
echo ""
echo "=== DATASET VERIFICATION ==="
echo "  Images directory: $IMAGES_DIR"
echo "  Image count: $IMAGE_COUNT"

if [[ "$IMAGE_COUNT" -lt 1000 ]]; then
    echo "ERROR: Too few images found ($IMAGE_COUNT). Expected ~20000."
    exit 1
fi

echo "Dataset extraction: OK ($IMAGE_COUNT images)"

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

coco_json = os.environ.get('COCO_JSON', '')
images_dir = Path(os.environ.get('IMAGES_DIR', ''))
output_dir = Path(os.environ.get('YOLO_DATASET_DIR', ''))

print(f'Loading COCO annotations: {coco_json}')
with open(coco_json, 'r') as f:
    coco_data = json.load(f)

print(f'  Images in JSON: {len(coco_data["images"])}')
print(f'  Annotations: {len(coco_data["annotations"])}')
print(f'  Categories: {[c["name"] for c in coco_data.get("categories", [])]}')

# Create directories
labels_train_dir = output_dir / 'labels' / 'train'
labels_val_dir = output_dir / 'labels' / 'val'
images_train_dir = output_dir / 'images' / 'train'
images_val_dir = output_dir / 'images' / 'val'

labels_train_dir.mkdir(parents=True, exist_ok=True)
labels_val_dir.mkdir(parents=True, exist_ok=True)
images_train_dir.mkdir(parents=True, exist_ok=True)
images_val_dir.mkdir(parents=True, exist_ok=True)

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

print(f'  Category mapping: {cat_id_to_yolo}')

# Split 90% train, 10% val
import random
random.seed(42)
image_ids = list(image_id_to_info.keys())
random.shuffle(image_ids)
split_idx = int(len(image_ids) * 0.9)
train_ids = set(image_ids[:split_idx])
val_ids = set(image_ids[split_idx:])

print(f'  Train images: {len(train_ids)}')
print(f'  Val images: {len(val_ids)}')

# Convert
converted_train = 0
converted_val = 0
linked_train = 0
linked_val = 0

for img_id, img_info in image_id_to_info.items():
    img_w = img_info['width']
    img_h = img_info['height']
    file_name = img_info['file_name']

    # Determine split
    is_train = img_id in train_ids
    labels_dir = labels_train_dir if is_train else labels_val_dir
    images_out_dir = images_train_dir if is_train else images_val_dir

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

    if is_train:
        converted_train += 1
    else:
        converted_val += 1

    # Symlink image
    src = images_dir / file_name
    dst = images_out_dir / file_name
    if src.exists() and not dst.exists():
        try:
            os.symlink(src, dst)
            if is_train:
                linked_train += 1
            else:
                linked_val += 1
        except OSError:
            import shutil
            shutil.copy2(src, dst)
            if is_train:
                linked_train += 1
            else:
                linked_val += 1

print(f'  Labels created - train: {converted_train}, val: {converted_val}')
print(f'  Images linked - train: {linked_train}, val: {linked_val}')

# Create dataset.yaml
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

print(f'  Dataset YAML: {yaml_path}')
print('COCO to YOLO conversion: DONE')
PYTHON_SCRIPT

DATASET_YAML="$YOLO_DATASET_DIR/dataset.yaml"

if [[ ! -f "$DATASET_YAML" ]]; then
    echo "ERROR: Failed to create dataset.yaml"
    exit 1
fi

echo ""
echo "YOLO dataset ready: $DATASET_YAML"
cat "$DATASET_YAML"

# === CHECKPOINT CONFIGURATION ===
# Check for existing YOLO checkpoint
YOLO_CHECKPOINT=""
POSSIBLE_CHECKPOINTS=(
    "/mnt/evafs/faculty/home/bpiotrowski/Yolo/ckpt_5gpu_hopper/epoch170.pt"
    "/mnt/evafs/faculty/home/bpiotrowski/Yolo/best_5gpu_hopper/best.pt"
    "/mnt/evafs/faculty/home/bpiotrowski/YOLO_Checkpoints/epoch170.pt"
)

for ckpt in "${POSSIBLE_CHECKPOINTS[@]}"; do
    if [[ -f "$ckpt" ]]; then
        YOLO_CHECKPOINT="$ckpt"
        echo "Found checkpoint: $YOLO_CHECKPOINT"
        break
    fi
done

# === TRAINING CONFIGURATION ===
echo ""
echo "=== TRAINING CONFIGURATION ==="
echo "  Node: DGX-1 (8x V100 32GB)"
echo "  GPUs: 8"
echo "  Batch size per GPU: 32"
echo "  Effective batch: 256"
echo "  Epochs: 300"
echo "  Learning rate: 0.001 (reduced for fine-tuning)"
echo "  Checkpoint: ${YOLO_CHECKPOINT:-'None (fresh start)'}"

# === START TRAINING ===
cd "$PROJECT_ROOT"

# Copy training script if not present
if [[ ! -f "$PROJECT_ROOT/yolo_train_20k_finetune.py" ]]; then
    echo "ERROR: yolo_train_20k_finetune.py not found in $PROJECT_ROOT"
    echo "Please copy the script first"
    exit 1
fi

# Build training command
TRAIN_CMD="python3 yolo_train_20k_finetune.py \
    --dataset_yaml_path $DATASET_YAML \
    --project_dir $PROJECT_ROOT/train_out \
    --checkpoint_dir $PROJECT_ROOT/ckpt \
    --best_model_dir $PROJECT_ROOT/best \
    --model_size m \
    --epochs 300 \
    --batch_size 32 \
    --imgsz 640 \
    --learning_rate 0.001 \
    --workers 16 \
    --device 0,1,2,3 \
    --save_period 5 \
    --patience 30 \
    --freeze_backbone 0 \
    --verbose"

# Add resume if checkpoint exists
if [[ -n "$YOLO_CHECKPOINT" ]]; then
    TRAIN_CMD="$TRAIN_CMD --resume --resume_path $YOLO_CHECKPOINT"
fi

echo ""
echo "=== STARTING TRAINING ==="
echo "Command: $TRAIN_CMD"
echo ""

eval $TRAIN_CMD
TRAIN_EXIT_CODE=$?

echo ""
echo "=== TRAINING FINISHED ==="
echo "Exit code: $TRAIN_EXIT_CODE"

# Copy final results to permanent storage
if [[ -d "$PROJECT_ROOT/train_out" ]]; then
    echo "Training output saved to: $PROJECT_ROOT/train_out"
fi

if [[ -d "$PROJECT_ROOT/best" ]]; then
    echo "Best model saved to: $PROJECT_ROOT/best"
fi

echo ""
echo "============================================================"
echo "  JOB COMPLETED"
echo "  Finished at: $(date)"
echo "  Results: $PROJECT_ROOT/train_out/"
echo "============================================================"

exit $TRAIN_EXIT_CODE
