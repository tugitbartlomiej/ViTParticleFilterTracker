#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p short          
#SBATCH --gres=gpu:0      
#SBATCH --mem=2G          
#SBATCH --time=0:10:00    
#SBATCH --job-name=create_yaml
#SBATCH --output=./out/create_yaml_%j.out

# Path to Python
PYTHON_PATH="/home2/faculty/bpiotrowski/.conda/envs/detr_env_py310/bin/python"

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python not found at path $PYTHON_PATH"
    exit 1
fi

echo "Using Python: $PYTHON_PATH"
$PYTHON_PATH --version

# Go to the directory
cd /mnt/evafs/faculty/home/bpiotrowski/datasets/YOLO_03062025

# Configuration
OUTPUT_DIR="./yolo_dataset"
IMAGES_DIR="/mnt/evafs/faculty/home/bpiotrowski/datasets/DETR_augmented_dataset_20250218/images"
COCO_JSON="/mnt/evafs/faculty/home/bpiotrowski/datasets/DETR_augmented_dataset_20250218/augmented_coco_20250417_030014.json"

echo "Configuration:"
echo "Output DIR: $OUTPUT_DIR"
echo "Images DIR: $IMAGES_DIR"
echo "COCO JSON: $COCO_JSON"

# Run YAML creator
$PYTHON_PATH ./YOLOYamlCreator.py \
    --output_dir "$OUTPUT_DIR" \
    --images_dir "$IMAGES_DIR" \
    --coco_json "$COCO_JSON" \
    --dataset_name "surgical_tools_yolo" \
    --create_training_config \
    --verify

echo ""
echo "YAML files created!"
echo ""
echo "Check the files:"
ls -la "$OUTPUT_DIR"/*.yaml

echo ""
echo "Sample content of data.yaml:"
head -20 "$OUTPUT_DIR/data.yaml" 2>/dev/null || echo "Could not read data.yaml"