#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p short          # Partycja common - mniejsze wymagania dla wizualizacji
#SBATCH --gres=gpu:0       # Bez GPU dla wizualizacji
#SBATCH --mem=16G          # Mniejsze zapotrzebowanie na pamięć
#SBATCH --time=2:00:00     # Krótszy czas wykonania
#SBATCH --job-name=coco_visualization
#SBATCH --output=out/coco_visualization_%j.out

# Bezpośrednia ścieżka do Pythona w Twoim środowisku
PYTHON_PATH="/home2/faculty/bpiotrowski/.conda/envs/detr_env_py310/bin/python"

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
cd /mnt/evafs/faculty/home/bpiotrowski/DETR/DETR_fo_a100

# Parametry dla skryptu
IMAGES_DIR="/mnt/evafs/faculty/home/bpiotrowski/datasets/DETR_augmented_dataset_20250218_ranged/images"
ANNOTATIONS_FILE="/mnt/evafs/faculty/home/bpiotrowski/datasets/DETR_augmented_dataset_20250218_ranged/augmented_coco_0-50_20250417_012413.json"
OUTPUT_DIR="./visualized_annotations"
NUM_SAMPLES=100  # Liczba losowo wybranych obrazów do wizualizacji

# Uruchom skrypt
$PYTHON_PATH visualize_coco_annotations.py \
    --images_dir "$IMAGES_DIR" \
    --annotations_file "$ANNOTATIONS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --draw_labels \
    --color green