#!/bin/bash

###############################################################################
#  YOLO – 8×GPU na DGX-1 | Klaster Eden
#  Wersja z wyłączonym DDP i automatyczną poprawą ścieżek
###############################################################################

#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-1
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --job-name=yolo_8gpu_dgx1
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo
#SBATCH --output=out/yolo_8gpu_%j.out
#SBATCH --error=out/yolo_8gpu_%j.err
#SBATCH --open-mode=append

set -euo pipefail

echo "=== YOLO 8×GPU on $(hostname) | $(date) ==="

# --- Definiowanie katalogu tymczasowego z opcją zapasową ---
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR → $TMPDIR"

# ─── Aktywacja środowiska Conda ─────────────────────────────────────────────
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# ─── Rozpakowanie datasetu ──────────────────────────────────────────────────
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606.tar.gz
echo "📦 Rozpakowuję dataset do $TMPDIR..."
tar -xzf "$ARCHIVE" -C "$TMPDIR"
DATASET_PATH=$(find "$TMPDIR" -name dataset.yaml | head -n 1)
if [[ -z "$DATASET_PATH" ]]; then
    echo "❌ ERROR: dataset.yaml not found in archive!"
    exit 1
fi
echo "✓ dataset.yaml → $DATASET_PATH"

# ─── Poprawka ścieżek w dataset.yaml ──────────────────────────────────────────
echo "🔧 Poprawiam ścieżki w dataset.yaml..."
DATASET_DIR=$(dirname "$DATASET_PATH")
YOLO_DATASET_DIR=$(find "$TMPDIR" -type d -name "yolo_dataset" | head -n 1)
IMAGES_DIR=$(find "$TMPDIR" -type d -name "images" | head -n 1)

if [[ -z "$YOLO_DATASET_DIR" ]]; then
    echo "❌ ERROR: yolo_dataset directory not found!"
    exit 1
fi

if [[ -z "$IMAGES_DIR" ]]; then
    echo "❌ ERROR: images directory not found!"
    exit 1
fi

echo "📁 Dataset dir: $YOLO_DATASET_DIR"
echo "🖼️ Images dir: $IMAGES_DIR"

# Backup oryginalnego pliku
cp "$DATASET_PATH" "$DATASET_PATH.backup"

# Aktualizuj ścieżki w dataset.yaml
sed -i "s|path: .*|path: $YOLO_DATASET_DIR|g" "$DATASET_PATH"
sed -i "s|train: .*\.txt|train: $YOLO_DATASET_DIR/train.txt|g" "$DATASET_PATH"
sed -i "s|val: .*\.txt|val: $YOLO_DATASET_DIR/val.txt|g" "$DATASET_PATH"
sed -i "s|test: .*\.txt|test: $YOLO_DATASET_DIR/test.txt|g" "$DATASET_PATH"

# Aktualizuj ścieżki w plikach list (train.txt, val.txt, test.txt)
echo "🔧 Poprawiam ścieżki w plikach list obrazów..."
for LIST_FILE in "$YOLO_DATASET_DIR"/{train,val,test}.txt; do
    if [[ -f "$LIST_FILE" ]]; then
        echo "  → Aktualizuję $LIST_FILE"
        # Backup oryginalnego pliku
        cp "$LIST_FILE" "$LIST_FILE.backup"
        # Zamień stare ścieżki na nowe
        sed -i "s|/mnt/evafs/faculty/home/bpiotrowski/datasets/DETR_augmented_dataset_20250218/images|$IMAGES_DIR|g" "$LIST_FILE"
        # Sprawdź ile linii zostało zaktualizowanych
        UPDATED_LINES=$(wc -l < "$LIST_FILE")
        echo "    ✓ Zaktualizowano $UPDATED_LINES ścieżek obrazów"
    fi
done

echo "✅ Ścieżki zostały poprawione!"
echo "📄 Sprawdź nową konfigurację:"
head -15 "$DATASET_PATH"

# ─── Weryfikacja struktury datasetu ──────────────────────────────────────────
echo ""
echo "🔍 Weryfikuję strukturę datasetu..."
echo "Katalogi:"
echo "  YOLO dataset: $YOLO_DATASET_DIR"
echo "  Images: $IMAGES_DIR"
echo "  Labels: $YOLO_DATASET_DIR/labels"

# Sprawdź czy istnieją podstawowe katalogi
for DIR in "$YOLO_DATASET_DIR/labels/train" "$YOLO_DATASET_DIR/labels/val" "$YOLO_DATASET_DIR/labels/test"; do
    if [[ -d "$DIR" ]]; then
        LABEL_COUNT=$(find "$DIR" -name "*.txt" | wc -l)
        echo "  ✓ $DIR: $LABEL_COUNT plików etykiet"
    else
        echo "  ⚠️ $DIR: nie istnieje"
    fi
done

# Sprawdź pierwsze 3 ścieżki z każdego pliku listy
echo ""
echo "📋 Przykładowe ścieżki obrazów:"
for LIST_FILE in "$YOLO_DATASET_DIR"/{train,val,test}.txt; do
    if [[ -f "$LIST_FILE" ]]; then
        LIST_NAME=$(basename "$LIST_FILE" .txt)
        echo "  $LIST_NAME.txt (pierwsze 3 linie):"
        head -3 "$LIST_FILE" | sed 's/^/    /'
        
        # Sprawdź czy pierwszych 3 obrazów rzeczywiście istnieje
        head -3 "$LIST_FILE" | while read IMAGE_PATH; do
            if [[ -f "$IMAGE_PATH" ]]; then
                echo "    ✓ $(basename "$IMAGE_PATH")"
            else
                echo "    ✗ $(basename "$IMAGE_PATH") - nie istnieje!"
            fi
        done
    fi
done

# ─── Uruchom trening YOLO ───────────────────────────────────────────────────
echo ""
echo "🚀 Uruchamiam trening YOLO na 8 GPU z wyłączonym DDP..."

python yolo-train.py \
  --dataset_yaml_path "$DATASET_PATH" \
  --project_dir "./train_out_8gpu" \
  --checkpoint_dir "./ckpt_8gpu" \
  --best_model_dir "./best_8gpu" \
  --model_size l \
  --epochs 200 \
  --batch_size 96 \
  --imgsz 640 \
  --learning_rate 0.01 \
  --workers 24 \
  --device "0,1,2,3,4,5,6,7" \
  --save_period 10 \
  --patience 50 \
  --verbose \
  --disable_ddp

echo -e "\n✓ Zadanie zakończone z kodem: $?"
