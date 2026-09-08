#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=yolo_20k_scratch
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune
#SBATCH --output=logs/yolo_20k_scratch_%j.log
#SBATCH --error=logs/yolo_20k_scratch_%j.log
#SBATCH --open-mode=append

# =============================================================================
# YOLOv8 — trening OD ZERA (epoch 0 -> 400) na kurowanym zbiorze 20k
#
# ROZNICA vs run_yolo_20k_ep170_lr0001.slurm:
#   - BRAK --pretrained_path  => wrapper laduje yolov8l.pt (COCO), NOWY trening od 0
#   - epochs 400 (zamiast resume 170->), lr0=0.01 (full-training, NIE 0.0001 fine-tune)
#
# Init wag: yolov8l.pt (COCO-pretrained) — taki sam typ inicjalizacji jak base YOLO.
# Dane: 20kSelectedImages.tar + merged_20k_annotations_fixed.json (kurowany 20k).
#
# Usage:
#   sbatch run_yolo_20k_from_scratch.slurm
#   EPOCHS=300 sbatch run_yolo_20k_from_scratch.slurm
# =============================================================================

set -euo pipefail

EPOCHS="${EPOCHS:-400}"
MODEL_SIZE="${MODEL_SIZE:-l}"

echo "============================================================"
echo "  YOLO FROM SCRATCH on 20k | yolov8${MODEL_SIZE}.pt (COCO) | 0 -> $EPOCHS"
echo "  DATE: $(date)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   ${SLURM_NODELIST:-local}"
echo ""
echo "UWAGA: BRAK --pretrained_path => wrapper startuje od yolov8${MODEL_SIZE}.pt (COCO)."
echo "       To NOWY trening od epoki 0 na 20k (NIE fine-tune z ep170 modelu 91k)."
echo ""

# === PROJECT DIRECTORIES ===
PROJECT_ROOT=/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune
mkdir -p "$PROJECT_ROOT/logs"

# === TMPDIR ===
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-$$}}
mkdir -p "$TMPDIR"
echo "TMPDIR: $TMPDIR"

# === CONDA ===
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# === INPUT FILES ===
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/20kSelectedImages.tar
COCO_JSON=/mnt/evafs/faculty/home/bpiotrowski/merged_20k_annotations_fixed.json
for f in "$ARCHIVE" "$COCO_JSON"; do
    if [[ ! -f "$f" ]]; then echo "ERROR: brak pliku: $f"; exit 1; fi
done

# === WRAPPER PRESENT? ===
WRAPPER="$PROJECT_ROOT/yolo_train_20k_finetune_v2.py"
if [[ ! -f "$WRAPPER" ]]; then
    echo "ERROR: brak $WRAPPER (skopiuj z Eden/Scripts/Train20kDataset_YOLO_31.12.25/)"
    exit 1
fi

# === EXTRACT 20K DATASET ===
echo ""
echo "=== EXTRACTING 20K DATASET ==="
EXTRACT_DIR="$TMPDIR/20k_dataset"
mkdir -p "$EXTRACT_DIR"
tar -xf "$ARCHIVE" -C "$EXTRACT_DIR"

if [[ -d "$EXTRACT_DIR/images" ]] && [[ $(find "$EXTRACT_DIR/images" -maxdepth 1 -name "*.jpg" 2>/dev/null | wc -l) -gt 0 ]]; then
    IMAGES_DIR="$EXTRACT_DIR/images"
elif [[ $(find "$EXTRACT_DIR" -maxdepth 1 -name "*.jpg" 2>/dev/null | wc -l) -gt 0 ]]; then
    IMAGES_DIR="$EXTRACT_DIR"
else
    echo "ERROR: brak obrazow po ekstrakcji!"; exit 1
fi
IMAGE_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -name "*.jpg" | wc -l)
echo "Images: $IMAGE_COUNT"
if [[ "$IMAGE_COUNT" -lt 1000 ]]; then echo "ERROR: za malo obrazow ($IMAGE_COUNT)"; exit 1; fi

# === CONVERT COCO -> YOLO (split 90/10, seed 42 — identycznie jak produkcja) ===
echo ""
echo "=== CONVERTING COCO TO YOLO FORMAT ==="
YOLO_DATASET_DIR="$TMPDIR/yolo_dataset"
export COCO_JSON IMAGES_DIR YOLO_DATASET_DIR
python3 << 'PYTHON_SCRIPT'
import json, os, random
from pathlib import Path
import yaml

coco_json = os.environ.get('COCO_JSON', '')
images_dir = Path(os.environ.get('IMAGES_DIR', ''))
output_dir = Path(os.environ.get('YOLO_DATASET_DIR', ''))

with open(coco_json) as f:
    coco_data = json.load(f)
print(f'  Images in JSON: {len(coco_data["images"])}  Annotations: {len(coco_data["annotations"])}')

labels_train_dir = output_dir / 'labels' / 'train'
labels_val_dir   = output_dir / 'labels' / 'val'
images_train_dir = output_dir / 'images' / 'train'
images_val_dir   = output_dir / 'images' / 'val'
for d in [labels_train_dir, labels_val_dir, images_train_dir, images_val_dir]:
    d.mkdir(parents=True, exist_ok=True)

image_id_to_info = {img['id']: img for img in coco_data['images']}
image_id_to_anns = {}
for ann in coco_data['annotations']:
    image_id_to_anns.setdefault(ann['image_id'], []).append(ann)

categories = coco_data.get('categories', [{'id': 1, 'name': 'tooltip'}])
cat_id_to_yolo = {cat['id']: idx for idx, cat in enumerate(categories)}
class_names = [cat['name'] for cat in categories]

random.seed(42)
image_ids = list(image_id_to_info.keys())
random.shuffle(image_ids)
split_idx = int(len(image_ids) * 0.9)
train_ids = set(image_ids[:split_idx])

converted = 0
for img_id, img_info in image_id_to_info.items():
    img_w, img_h = img_info['width'], img_info['height']
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
        except OSError:
            import shutil; shutil.copy2(src, dst)
    converted += 1

print(f'Converted {converted} images')
yaml_content = {
    'path': str(output_dir),
    'train': 'images/train',
    'val': 'images/val',
    'names': {i: name for i, name in enumerate(class_names)},
    'nc': len(class_names),
}
with open(output_dir / 'dataset.yaml', 'w') as f:
    yaml.dump(yaml_content, f, default_flow_style=False)
print(f'Dataset YAML: {output_dir / "dataset.yaml"}')
PYTHON_SCRIPT

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
echo "  TRAINING CONFIGURATION (FROM SCRATCH)"
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
    --project_dir "$PROJECT_ROOT/train_out_scratch_20k" \
    --checkpoint_dir "$PROJECT_ROOT/ckpt_scratch_20k" \
    --best_model_dir "$PROJECT_ROOT/best_scratch_20k" \
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
if [[ -d "$PROJECT_ROOT/train_out_scratch_20k/exp/weights" ]]; then
    FINAL_DIR="$PROJECT_ROOT/final_scratch_20k"
    mkdir -p "$FINAL_DIR"
    cp "$PROJECT_ROOT/train_out_scratch_20k/exp/weights/"*.pt "$FINAL_DIR/" 2>/dev/null || true
    echo "Final checkpoints -> $FINAL_DIR"
fi

exit $EXIT_CODE
