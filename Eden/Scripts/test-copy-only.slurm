#!/bin/bash

###############################################################################
#  TEST - Tylko kopiowanie datasetu (debug)
###############################################################################

#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-1
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --job-name=test_copy_only
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo
#SBATCH --output=out/test_copy_only_%j.out
#SBATCH --error=out/test_copy_only_%j.err
#SBATCH --open-mode=append

set -euo pipefail

echo "=== TEST KOPIOWANIA DATASETU | $(date) ==="

TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR → $TMPDIR"

# Sprawdź miejsce
echo "💾 Dostępne miejsce na start:"
df -h "$TMPDIR"

# Rozpakuj
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606.tar.gz
echo "📦 Rozpakowuję..."
tar -xzf "$ARCHIVE" -C "$TMPDIR"

echo "💾 Miejsce po rozpakowaniu:"
df -h "$TMPDIR"

# Znajdź katalogi
IMAGES_DIR=$(find "$TMPDIR" -name "images" -type d | head -n 1)
YOLO_DATASET_DIR=$(find "$TMPDIR" -name "yolo_dataset" -type d | head -n 1)

echo "🖼️ Images dir: $IMAGES_DIR"
echo "📁 Dataset dir: $YOLO_DATASET_DIR"

# Policz pliki
images_count=$(find "$IMAGES_DIR" -name "*.jpg" -o -name "*.png" | wc -l)
labels_count=$(find "$YOLO_DATASET_DIR" -name "*.txt" | wc -l)

echo "📊 Znaleziono:"
echo "  Images: $images_count"
echo "  Labels: $labels_count"

# Test kopiowania małej próbki (1000 plików)
UNIFIED_DIR="$TMPDIR/test_unified"
mkdir -p "$UNIFIED_DIR/images/test"
mkdir -p "$UNIFIED_DIR/labels/test"

echo ""
echo "🧪 Test kopiowania pierwszych 1000 obrazów..."

ALL_IMAGES=($(find "$IMAGES_DIR" -name "*.jpg" -o -name "*.png" | head -n 1000))
ALL_LABELS=($(find "$YOLO_DATASET_DIR" -name "*.txt"))

counter=0
for image_path in "${ALL_IMAGES[@]}"; do
    image_name=$(basename "$image_path")
    image_base="${image_name%.*}"
    
    # Kopiuj obraz
    cp "$image_path" "$UNIFIED_DIR/images/test/"
    
    # Znajdź etykietę
    for label_path in "${ALL_LABELS[@]}"; do
        label_name=$(basename "$label_path")
        label_base="${label_name%.*}"
        
        if [[ "$image_base" == "$label_base" ]]; then
            cp "$label_path" "$UNIFIED_DIR/labels/test/"
            break
        fi
    done
    
    ((counter++))
    
    if [[ $((counter % 100)) -eq 0 ]]; then
        echo "  Skopiowano $counter/1000 plików..."
        
        # Sprawdź miejsce
        available=$(df "$TMPDIR" | tail -1 | awk '{print $4}')
        echo "    Dostępne miejsce: ${available}KB"
        
        if [[ $available -lt 500000 ]]; then
            echo "❌ Za mało miejsca na dysku!"
            exit 1
        fi
    fi
done

echo ""
echo "✅ Test zakończony pomyślnie!"
echo "📊 Skopiowano:"
echo "  Images: $(find "$UNIFIED_DIR/images/test" -name "*.jpg" -o -name "*.png" | wc -l)"
echo "  Labels: $(find "$UNIFIED_DIR/labels/test" -name "*.txt" | wc -l)"

echo "💾 Finalne zużycie miejsca:"
df -h "$TMPDIR"
