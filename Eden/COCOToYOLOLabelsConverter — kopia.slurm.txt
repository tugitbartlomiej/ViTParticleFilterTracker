#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p short          
#SBATCH --gres=gpu:0      
#SBATCH --mem=8G          
#SBATCH --time=1:00:00    
#SBATCH --job-name=coco_to_yolo_split
#SBATCH --output=./out/coco_to_yolo_split_%j.out

# Path to Python
PYTHON_PATH="/home2/faculty/bpiotrowski/.conda/envs/detr_env_py310/bin/python"

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python not found at path $PYTHON_PATH"
    exit 1
fi

echo "Using Python: $PYTHON_PATH"
$PYTHON_PATH --version

# Go to the directory where this script is located
cd /mnt/evafs/faculty/home/bpiotrowski/datasets/YOLO_03062025

# Create necessary directories
mkdir -p ./out
mkdir -p ./yolo_dataset

# Paths configuration
COCO_JSON_PATH="/mnt/evafs/faculty/home/bpiotrowski/datasets/DETR_augmented_dataset_20250218/augmented_coco_20250417_030014.json"
IMAGES_DIR="/mnt/evafs/faculty/home/bpiotrowski/datasets/DETR_augmented_dataset_20250218/images"
OUTPUT_DIR="./yolo_dataset"

# Check if files exist
if [ ! -f "$COCO_JSON_PATH" ]; then
    echo "COCO JSON not found at: $COCO_JSON_PATH"
    echo "Looking for JSON files in parent directories..."
    
    # Try to find JSON files automatically
    FOUND_JSON=$(find ../DETR_augmented_dataset_20250218/ -name "*.json" -type f | head -1)
    if [ -n "$FOUND_JSON" ]; then
        COCO_JSON_PATH="$FOUND_JSON"
        echo "Found JSON: $COCO_JSON_PATH"
    else
        echo "ERROR: No JSON file found!"
        exit 1
    fi
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found: $IMAGES_DIR"
    exit 1
fi

echo "Configuration:"
echo "COCO JSON: $COCO_JSON_PATH"
echo "Images DIR: $IMAGES_DIR"
echo "Output DIR: $OUTPUT_DIR"
echo ""
echo "Dataset split ratios:"
echo "Train: 70%"
echo "Val: 20%"
echo "Test: 10%"

# Run conversion with dataset split
$PYTHON_PATH ./COCOToYOLOLabelsConverter.py \
    --coco_json "$COCO_JSON_PATH" \
    --images_dir "$IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --class_mapping "0:0" \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --test_ratio 0.1 \
    --random_seed 42 \
    --dataset_name "surgical_tools_yolo" \
    --verbose

echo ""
echo "Conversion completed!"
echo "Output directory structure:"
echo ""

# Show the created structure
echo "Directory tree:"
tree -L 3 "$OUTPUT_DIR" 2>/dev/null || {
    echo "$OUTPUT_DIR/"
    echo "├── dataset.yaml"
    echo "├── data.yaml"
    echo "├── train.txt"
    echo "├── val.txt"
    echo "├── test.txt"
    echo "└── labels/"
    echo "    ├── train/"
    echo "    ├── val/"
    echo "    └── test/"
}

echo ""
echo "File counts:"
echo "Train labels: $(find $OUTPUT_DIR/labels/train -name "*.txt" | wc -l)"
echo "Val labels: $(find $OUTPUT_DIR/labels/val -name "*.txt" | wc -l)"
echo "Test labels: $(find $OUTPUT_DIR/labels/test -name "*.txt" | wc -l)"

echo ""
echo "Sample content of dataset.yaml:"
head -20 "$OUTPUT_DIR/dataset.yaml" 2>/dev/null || echo "Could not read dataset.yaml"