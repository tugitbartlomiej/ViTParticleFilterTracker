#!/bin/bash

###############################################################################
#  YOLO – 8×GPU na DGX-1 | Klaster Eden
#  Wersja z naprawą problemu brakujących plików list obrazów
###############################################################################

#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-1
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --job-name=yolo_8gpu_fixed
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

# ─── Znajdź katalogi po rozpakowaniu ───────────────────────────────────────
echo "🔍 Szukam katalogów w rozpakownym archiwum..."
YOLO_DATASET_DIR=$(find "$TMPDIR" -name "yolo_dataset" -type d | head -n 1)
IMAGES_DIR=$(find "$TMPDIR" -name "images" -type d | head -n 1)
DATASET_YAML=$(find "$TMPDIR" -name "dataset.yaml" | head -n 1)

if [[ -z "$YOLO_DATASET_DIR" ]]; then
    echo "❌ ERROR: Nie znaleziono katalogu yolo_dataset!"
    exit 1
fi

if [[ -z "$IMAGES_DIR" ]]; then
    echo "❌ ERROR: Nie znaleziono katalogu images!"
    exit 1
fi

if [[ -z "$DATASET_YAML" ]]; then
    echo "❌ ERROR: Nie znaleziono pliku dataset.yaml!"
    exit 1
fi

echo "📁 Dataset dir: $YOLO_DATASET_DIR"
echo "🖼️ Images dir: $IMAGES_DIR"
echo "📄 Dataset YAML: $DATASET_YAML"

# ─── Sprawdź strukturę etykiet ─────────────────────────────────────────────
echo "🔍 Weryfikuję strukturę datasetu..."
echo "Katalogi:"
echo "  YOLO dataset: $YOLO_DATASET_DIR"
echo "  Images: $IMAGES_DIR"
echo "  Labels: $YOLO_DATASET_DIR/labels"

for split in train val test; do
    labels_dir="$YOLO_DATASET_DIR/labels/$split"
    if [[ -d "$labels_dir" ]]; then
        label_count=$(find "$labels_dir" -name "*.txt" | wc -l)
        echo "  ✓ $YOLO_DATASET_DIR/labels/$split: $label_count plików etykiet"
    else
        echo "  ⚠️ $YOLO_DATASET_DIR/labels/$split: nie istnieje"
    fi
done

# ─── Stwórz pliki list obrazów ─────────────────────────────────────────────
echo ""
echo "🔧 Tworzę pliki list obrazów na podstawie etykiet..."

# Sprawdź czy istnieje skrypt do tworzenia list
CREATE_LISTS_SCRIPT="./create_image_lists.py"
if [[ -f "$CREATE_LISTS_SCRIPT" ]]; then
    echo "📝 Używam skryptu: $CREATE_LISTS_SCRIPT"
    python "$CREATE_LISTS_SCRIPT" \
        --yolo_dataset_dir "$YOLO_DATASET_DIR" \
        --images_dir "$IMAGES_DIR" \
        --verify \
        --verbose
else
    echo "⚠️ Skrypt $CREATE_LISTS_SCRIPT nie istnieje - tworzę listy ręcznie..."
    
    # Ręczne tworzenie list obrazów
    for split in train val test; do
        labels_dir="$YOLO_DATASET_DIR/labels/$split"
        list_file="$YOLO_DATASET_DIR/$split.txt"
        
        if [[ -d "$labels_dir" ]]; then
            echo "  → Tworzę $split.txt..."
            
            # Wyczyść plik
            > "$list_file"
            
            # Dla każdego pliku etykiety znajdź odpowiadający obraz
            find "$labels_dir" -name "*.txt" | while read label_file; do
                base_name=$(basename "$label_file" .txt)
                
                # Sprawdź różne rozszerzenia obrazów
                for ext in jpg jpeg png bmp tiff tif; do
                    image_file="$IMAGES_DIR/$base_name.$ext"
                    if [[ -f "$image_file" ]]; then
                        echo "$image_file" >> "$list_file"
                        break
                    fi
                done
            done
            
            image_count=$(wc -l < "$list_file")
            echo "    ✓ Utworzono $split.txt z $image_count obrazami"
        else
            echo "    ⚠️ Katalog $labels_dir nie istnieje - pomijam $split"
        fi
    done
fi

# ─── Popraw ścieżki w dataset.yaml ─────────────────────────────────────────
echo ""
echo "🔧 Poprawiam ścieżki w dataset.yaml..."

# Stwórz backup
cp "$DATASET_YAML" "$DATASET_YAML.backup"
echo "✓ dataset.yaml → $DATASET_YAML.backup"

# Popraw ścieżki w dataset.yaml
sed -i "s|^path:.*|path: $YOLO_DATASET_DIR|" "$DATASET_YAML"
sed -i "s|^train:.*|train: $YOLO_DATASET_DIR/train.txt|" "$DATASET_YAML"
sed -i "s|^val:.*|val: $YOLO_DATASET_DIR/val.txt|" "$DATASET_YAML"
sed -i "s|^test:.*|test: $YOLO_DATASET_DIR/test.txt|" "$DATASET_YAML"

echo "✅ Ścieżki zostały poprawione!"

# ─── Pokaż nową konfigurację ───────────────────────────────────────────────
echo "📄 Sprawdź nową konfigurację:"
head -20 "$DATASET_YAML"

# ─── Ostateczna weryfikacja ────────────────────────────────────────────────
echo ""
echo "🔍 Ostateczna weryfikacja..."
for split in train val test; do
    list_file="$YOLO_DATASET_DIR/$split.txt"
    if [[ -f "$list_file" ]]; then
        line_count=$(wc -l < "$list_file")
        echo "  ✓ $split.txt: $line_count obrazów"
        
        # Sprawdź pierwsze 3 obrazy
        echo "📋 Przykładowe ścieżki obrazów:"
        head -3 "$list_file" | while read img_path; do
            if [[ -f "$img_path" ]]; then
                echo "    ✓ $(basename "$img_path")"
            else
                echo "    ✗ $(basename "$img_path") - nie istnieje!"
            fi
        done
    else
        echo "  ✗ $split.txt: nie istnieje!"
    fi
done

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
  --workers 24 \
  --device "0,1,2,3,4,5,6,7" \
  --save_period 10 \
  --patience 50 \
  --verbose \
  --disable_ddp

echo -e "\n✓ Zadanie zakończone z kodem: $?"
