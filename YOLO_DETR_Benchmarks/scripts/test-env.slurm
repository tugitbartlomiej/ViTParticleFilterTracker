#!/bin/bash
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=00:05:00
#SBATCH --job-name=test_env
#SBATCH --output=test_env_%j.out

echo "=== Testing Python Environment ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo

# Test 1: Bezpośrednia ścieżka
echo "Test 1: Direct path to Python"
PYTHON_EXE="/home2/faculty/bpiotrowski/.conda/envs/detr_env_py310/bin/python"
if [ -f "$PYTHON_EXE" ]; then
    echo "Python found at: $PYTHON_EXE"
    $PYTHON_EXE --version
    echo "Testing imports..."
    $PYTHON_EXE -c "import torch; print(f'PyTorch: {torch.__version__}')"
    $PYTHON_EXE -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
else
    echo "ERROR: Python NOT found at $PYTHON_EXE"
    echo "Listing directory:"
    ls -la /home2/faculty/bpiotrowski/.conda/envs/detr_env_py310/bin/ | grep python
fi

echo
echo "Test 2: Conda activation method 1"
module load anaconda/4.0
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate /home2/faculty/bpiotrowski/.conda/envs/detr_env_py310
which python
python --version 2>&1 || echo "FAILED"

echo
echo "Test 3: Conda activation method 2"
eval "$(/mnt/evafs/software/anaconda/v.4.0/bin/conda shell.bash hook)"
conda activate /home2/faculty/bpiotrowski/.conda/envs/detr_env_py310
which python
python --version 2>&1 || echo "FAILED"

echo
echo "Test 4: Available conda environments"
conda env list

echo
echo "=== Tests Complete ==="