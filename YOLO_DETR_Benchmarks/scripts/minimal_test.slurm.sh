#!/bin/bash
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=00:15:00
#SBATCH --job-name=yolo_test
#SBATCH --output=test_yolo_%j.out

echo "=== YOLO MINIMAL TEST ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

cd /mnt/evafs/faculty/home/bpiotrowski/Yolo
PYTHON="/home2/faculty/bpiotrowski/.conda/envs/detr_env_py310/bin/python"

# Test 1: Sprawdź środowisko
echo -e "\n1. Environment check:"
$PYTHON --version
nvidia-smi --query-gpu=name,memory.total --format=csv

# Test 2: Sprawdź importy
echo -e "\n2. Import test:"
$PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}')"
$PYTHON -c "from ultralytics import YOLO; print('YOLO: OK')"

# Test 3: Mini trening (1 epoka)
echo -e "\n3. Running 1-epoch training test:"
$PYTHON yolo-train.py \
    --dataset_yaml_path "/mnt/evafs/faculty/home/bpiotrowski/datasets/YOLO_03062025/dataset.yaml" \
    --epochs 1 \
    --batch_size 4 \
    --device "0" \
    --model_size "n" \
    --project_dir "./test_output" \
    --verbose

EXIT_CODE=$?
echo -e "\nTest completed with exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ TEST PASSED - Ready for full training!"
else
    echo "✗ TEST FAILED - Check errors above"
fi