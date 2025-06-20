#!/bin/bash

###############################################################################
#  YOLO – 8×GPU na DGX-1 | Klaster Eden | Wersja 3 - Fix etykiet
#  Kopiuje etykiety obok obrazów żeby YOLO mogło je znaleźć
###############################################################################

#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-1
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --job-name=yolo_8gpu_dgx1_v3
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

# ─── Znajdowanie katalogów ──────────────────────────────────────────────────
echo "🔍 Szukam katalogów w rozpakownym archiwum..."

# Znajdź katalogi
YOLO_DATASET_DIR=$(find "$TMPDIR" -name "yolo_dataset" -type d | head -n 1)
IMAGES_DIR=$(find "$TMPDIR" -name "images" -type d | head -n 1)
DATASET_YAML=$(find "$TMPDIR" -name "dataset.yaml" | head -n 1)

if [[ -z "$YOLO_DATASET_DIR" || -z "$IMAGES_DIR" || -z "$DATASET_YAML" ]]; then
    echo "❌ ERROR: Nie znaleziono wymaganych katalogów w archiwum!"
    echo "YOLO_DATASET_DIR: $YOLO_DATASET_DIR"
    echo "IMAGES_DIR: $IMAGES_DIR"
    echo "DATASET_YAML: $DATASET_YAML"
    exit 1
fi

echo "📁 Dataset dir: $YOLO_DATASET_DIR"
echo "🖼️ Images dir: $IMAGES_DIR"
echo "📄 Dataset YAML: $DATASET_YAML"

# ─── Weryfikacja struktury ──────────────────────────────────────────────────
echo "🔍 Weryfikuję strukturę datasetu..."
echo "Katalogi:"
echo "  YOLO dataset: $YOLO_DATASET_DIR"
echo "  Images: $IMAGES_DIR"
echo "  Labels: $YOLO_DATASET_DIR/labels"

# Sprawdź liczby plików etykiet
for split in train val test; do
    labels_count=$(find "$YOLO_DATASET_DIR/labels/$split" -name "*.txt" 2>/dev/null | wc -l)
    if [[ $labels_count -gt 0 ]]; then
        echo "  ✓ $YOLO_DATASET_DIR/labels/$split: $labels_count plików etykiet"
    else
        echo "  ✓ $YOLO_DATASET_DIR/labels/$split: 0 plików etykiet"
    fi
done

# ─── Tworzenie list obrazów ─────────────────────────────────────────────────
echo ""
echo "🔧 Tworzę pliki list obrazów na podstawie etykiet..."

# Sprawdź czy skrypt istnieje
CREATE_LISTS_SCRIPT="./create_image_lists.py"
if [[ ! -f "$CREATE_LISTS_SCRIPT" ]]; then
    echo "❌ ERROR: Skrypt $CREATE_LISTS_SCRIPT nie istnieje!"
    exit 1
fi

echo "📝 Używam skryptu: $CREATE_LISTS_SCRIPT"
python "$CREATE_LISTS_SCRIPT" \
    --yolo_dataset_dir "$YOLO_DATASET_DIR" \
    --images_dir "$IMAGES_DIR" \
    --verify --verbose

# ─── Kopiowanie etykiet obok obrazów ────────────────────────────────────────
echo ""
echo "🔧 Kopiuję etykiety obok obrazów (fix dla YOLO)..."

# Sprawdź czy skrypt istnieje
COPY_LABELS_SCRIPT="./copy_labels_to_images.py"
if [[ ! -f "$COPY_LABELS_SCRIPT" ]]; then
    echo "❌ ERROR: Skrypt $COPY_LABELS_SCRIPT nie istnieje!"
    exit 1
fi

echo "📝 Używam skryptu: $COPY_LABELS_SCRIPT"
python "$COPY_LABELS_SCRIPT" \
    --yolo_dataset_dir "$YOLO_DATASET_DIR" \
    --images_dir "$IMAGES_DIR" \
    --verify --verbose

# ─── Aktualizacja dataset.yaml ──────────────────────────────────────────────
echo ""
echo "🔧 Poprawiam ścieżki w dataset.yaml..."

# Backup
cp "$DATASET_YAML" "$DATASET_YAML.backup"
echo "✓ dataset.yaml → $DATASET_YAML.backup"

# Aktualizuj ścieżki w dataset.yaml
cat > "$DATASET_YAML" << EOF
# YOLOv8 Dataset Configuration
# Tooltip Detection Dataset

# Dataset root - gdzie są foldery labels/
path: $YOLO_DATASET_DIR

# Ścieżki do obrazów - używamy list, bo obrazy są w innym miejscu
train: $YOLO_DATASET_DIR/train.txt
val: $YOLO_DATASET_DIR/val.txt
test: $YOLO_DATASET_DIR/test.txt

# Klasy
nc: 1
names: ['tooltip']

# Informacja o strukturze (komentarz dla użytkownika)
# Etykiety są w: path/labels/[train|val|test]/
# Obrazy są listowane w plikach train.txt, val.txt, test.txt
EOF

echo "✅ Ścieżki zostały poprawione!"
echo "📄 Sprawdź nową konfigurację:"
head -20 "$DATASET_YAML"

# ─── Ostateczna weryfikacja ─────────────────────────────────────────────────
echo "🔍 Ostateczna weryfikacja..."
for list_file in train.txt val.txt test.txt; do
    list_path="$YOLO_DATASET_DIR/$list_file"
    if [[ -f "$list_path" ]]; then
        line_count=$(wc -l < "$list_path" 2>/dev/null || echo "0")
        echo "  ✓ $list_file: $line_count obrazów"
        echo "📋 Przykładowe ścieżki obrazów:"
        head -3 "$list_path" | while read img_path; do
            if [[ -f "$img_path" ]]; then
                echo "    ✓ $(basename "$img_path")"
            else
                echo "    ✗ $(basename "$img_path") - nie istnieje!"
            fi
        done
    else
        echo "  ✗ $list_file: nie istnieje!"
    fi
done

# Sprawdź czy etykiety zostały skopiowane
COPIED_LABELS_DIR="$(dirname "$IMAGES_DIR")/labels"
if [[ -d "$COPIED_LABELS_DIR" ]]; then
    echo "  ✓ Etykiety skopiowane do: $COPIED_LABELS_DIR"
    for split in train val test; do
        if [[ -d "$COPIED_LABELS_DIR/$split" ]]; then
            labels_count=$(find "$COPIED_LABELS_DIR/$split" -name "*.txt" 2>/dev/null | wc -l)
            echo "    ✓ $split: $labels_count plików etykiet"
        fi
    done
else
    echo "  ⚠️ Brak skopiowanych etykiet w: $COPIED_LABELS_DIR"
fi

# ─── Uruchom trening YOLO ───────────────────────────────────────────────────
echo ""
echo "🚀 Uruchamiam trening YOLO na 8 GPU z wyłączonym DDP..."

python yolo-train.py \
  --dataset_yaml_path "$DATASET_YAML" \
  --project_dir "./train_out_8gpu" \
  --checkpoint_dir "./ckpt_8gpu" \
  --best_model_dir "./best_8gpu" \
  --model_size l \
  --epochs 200 \
  --batch_size 96 \
  --imgsz 640 \
  --learning_rate 0.01 \
  --workers 8 \
  --device "0,1,2,3,4,5,6,7" \
  --save_period 10 \
  --patience 50 \
  --verbose \
  --disable_ddp

echo -e "\n✓ Zadanie zakończone z kodem: $?"
