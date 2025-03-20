#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p long             # Change to common partition where dgx-2 is accessible
#SBATCH -w dgx-1             # Explicitly request the dgx-2 node
#SBATCH --gres=gpu:2         # Request 2 GPUs (since dgx-2 shows 2/8 GPUs available)
#SBATCH --mem=45G            # Adjust memory based on availability (dgx-2 has 192GB free)
#SBATCH --time=24:00:00      # Longer time limit for common partition
#SBATCH --job-name=detr_annotator_train_dgx2
#SBATCH --output=detr_annotator_train_dgx2_%j.out

# Bezpośrednia ścieżka do Pythona w Twoim środowisku
PYTHON_PATH="/home2/faculty/bpiotrowski/.conda/envs/detr_env_py310/bin/python"

# First install scipy if not already installed
$PYTHON_PATH -m pip install scipy

# Sprawdź, czy Python istnieje
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Błąd: Nie znaleziono Pythona w ścieżce $PYTHON_PATH"
    exit 1
fi

# Wyświetl informacje o środowisku
echo "Używany Python: $PYTHON_PATH"
$PYTHON_PATH --version

# Katalog bibliotek Pythona
PYTHON_LIB_DIR=$(dirname $(dirname $PYTHON_PATH))/lib
export LD_LIBRARY_PATH="$PYTHON_LIB_DIR:$LD_LIBRARY_PATH"

# Przejdź do katalogu projektu
cd /mnt/evafs/faculty/home/bpiotrowski/DETR

# Uruchom skrypt treningowy
$PYTHON_PATH detr_annotator_train.py