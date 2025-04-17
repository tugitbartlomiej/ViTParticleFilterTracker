#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p short          # Short partition for lighter workloads
#SBATCH --gres=gpu:0      # No GPU required for augmentation
#SBATCH --mem=16G         # Memory allocation
#SBATCH --time=2:00:00    # 2 hour time limit
#SBATCH --job-name=coco-range-aug
#SBATCH --output=.out/coco_range_augmentation_%j.out

# Path to Python in your environment
PYTHON_PATH="/home2/faculty/bpiotrowski/.conda/envs/detr_env_py310/bin/python"

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python not found at path $PYTHON_PATH"
    exit 1
fi

# Display environment information
echo "Using Python: $PYTHON_PATH"
$PYTHON_PATH --version

# Python libraries directory
PYTHON_LIB_DIR=$(dirname $(dirname $PYTHON_PATH))/lib
export LD_LIBRARY_PATH="$PYTHON_LIB_DIR:$LD_LIBRARY_PATH"

# Go to project directory
cd /mnt/evafs/faculty/home/bpiotrowski/DETR/DETR_fo_a100

# Run the range augmentation script
$PYTHON_PATH coco-augmentation_range.py \
--json_path ./coco_annotations_from_yolo_dataset_20250218.json \
--images_dir /mnt/evafs/faculty/home/bpiotrowski/datasets/yolo_dataset_20250218/images/train \
--output_dir /mnt/evafs/faculty/home/bpiotrowski/datasets/DETR_augmented_dataset_20250218_ranged \
--start_idx 0 \
--end_idx 50 \
--augmentations_per_image 3 \
--augmentation_strength medium \
--preserve_original_images