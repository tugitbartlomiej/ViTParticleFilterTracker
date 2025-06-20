#!/bin/bash

###############################################################################
#  YOLO – 8×GPU na DGX-1 | Klaster Eden | Wersja 4 - Finalna
#  Poprawiona struktura katalogów i ścieżek
###############################################################################

#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-1
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --job-name=yolo_8gpu_dgx1_v4
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo
#SBATCH --output=out/yolo_8gpu_%j.out
#SBATCH --error=out/yolo_8gpu_%j.err
#SBATCH --open-mode=append

set -euo pipefail

echo "=== YOLO 8×GPU FINALNA WERSJA on $(hostname) | $(date) ==="

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
echo "🔍 Analizuję strukturę rozpakowanego archiwum..."

# Lista wszystkich katalogów w archiwum
echo "📂 Wszystkie katalogi w archiwum:"
find "$TMPDIR" -type d | head -20

# Znajdź główne katalogi
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

# Określ katalog docelowy dla unified struktury
UNIFIED_DATASET_DIR="$TMPDIR/unified_dataset"
mkdir -p "$UNIFIED_DATASET_DIR"
echo "🎯 Unified dataset dir: $UNIFIED_DATASET_DIR"

# ─── Tworzenie unified struktury datasetu ───────────────────────────────────
echo ""
echo "🔧 Tworzę unified strukturę datasetu dla YOLO..."

# Kopiuj obrazy do unified struktury
echo "📋 Kopiuję obrazy..."
for split in train val test; do
    mkdir -p "$UNIFIED_DATASET_DIR/images/$split"
    images_src_dir="$IMAGES_DIR/$split"
    
    if [[ -d "$images_src_dir" ]]; then
        echo "  Kopiuję $split obrazy z $images_src_dir"
        cp -r "$images_src_dir"/* "$UNIFIED_DATASET_DIR/images/$split/" 2>/dev/null || true
        image_count=$(find "$UNIFIED_DATASET_DIR/images/$split" -name "*.jpg" -o -name "*.png" | wc -l)
        echo "    ✓ Skopiowano $image_count obrazów do images/$split"
    else
        echo "    ⚠️ Brak katalogu $images_src_dir"
    fi
done

# Kopiuj etykiety do unified struktury
echo "📋 Kopiuję etykiety..."
for split in train val test; do
    mkdir -p "$UNIFIED_DATASET_DIR/labels/$split"
    labels_src_dir="$YOLO_DATASET_DIR/labels/$split"
    
    if [[ -d "$labels_src_dir" ]]; then
        echo "  Kopiuję $split etykiety z $labels_src_dir"
        cp -r "$labels_src_dir"/* "$UNIFIED_DATASET_DIR/labels/$split/" 2>/dev/null || true
        label_count=$(find "$UNIFIED_DATASET_DIR/labels/$split" -name "*.txt" | wc -l)
        echo "    ✓ Skopiowano $label_count etykiet do labels/$split"
    else
        echo "    ⚠️ Brak katalogu $labels_src_dir"
    fi
done

# ─── Weryfikacja unified struktury ───────────────────────────────────────────
echo ""
echo "🔍 Weryfikuję unified strukturę..."
echo "📊 Statystyki datasetu:"

for split in train val test; do
    images_count=$(find "$UNIFIED_DATASET_DIR/images/$split" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
    labels_count=$(find "$UNIFIED_DATASET_DIR/labels/$split" -name "*.txt" 2>/dev/null | wc -l)
    echo "  $split: $images_count obrazów, $labels_count etykiet"
    
    # Sprawdź czy liczby się zgadzają
    if [[ $images_count -gt 0 && $labels_count -gt 0 ]]; then
        if [[ $images_count -eq $labels_count ]]; then
            echo "    ✅ Liczby obrazów i etykiet się zgadzają"
        else
            echo "    ⚠️ Różnica w liczbie obrazów ($images_count) i etykiet ($labels_count)"
        fi
    fi
done

# ─── Tworzenie nowego dataset.yaml ──────────────────────────────────────────
echo ""
echo "🔧 Tworzę nowy dataset.yaml z unified strukturą..."

NEW_DATASET_YAML="$UNIFIED_DATASET_DIR/dataset.yaml"
cat > "$NEW_DATASET_YAML" << EOF
# YOLOv8 Dataset Configuration - Unified Structure
# Tooltip Detection Dataset

# Dataset root - katalog bazowy z images/ i labels/
path: $UNIFIED_DATASET_DIR

# Ścieżki do setów (relatywne do path)
train: images/train
val: images/val
test: images/test

# Klasy
nc: 1
names: ['tooltip']

# Struktura:
# $UNIFIED_DATASET_DIR/
# ├── images/
# │   ├── train/
# │   ├── val/
# │   └── test/
# └── labels/
#     ├── train/
#     ├── val/
#     └── test/
EOF

echo "✅ Nowy dataset.yaml utworzony: $NEW_DATASET_YAML"
echo "📄 Konfiguracja:"
cat "$NEW_DATASET_YAML"

# ─── Ostateczna weryfikacja ─────────────────────────────────────────────────
echo ""
echo "🔍 Ostateczna weryfikacja przed treningiem..."

# Sprawdź czy YOLO może znaleźć etykiety
echo "🧪 Test dostępu do plików:"
for split in train val test; do
    echo "  Testing $split split:"
    
    # Przykładowy obraz
    sample_image=$(find "$UNIFIED_DATASET_DIR/images/$split" -name "*.jpg" -o -name "*.png" | head -n 1)
    if [[ -n "$sample_image" ]]; then
        echo "    ✓ Przykładowy obraz: $(basename "$sample_image")"
        
        # Szukaj odpowiadającej etykiety
        image_name=$(basename "$sample_image" | sed 's/\.[^.]*$//')
        label_file="$UNIFIED_DATASET_DIR/labels/$split/${image_name}.txt"
        
        if [[ -f "$label_file" ]]; then
            echo "    ✓ Znaleziono etykietę: $(basename "$label_file")"
        else
            echo "    ⚠️ Brak etykiety dla obrazu: $image_name"
        fi
    else
        echo "    ⚠️ Brak obrazów w $split"
    fi
done

# ─── Dodatkowe zabezpieczenia środowiska ────────────────────────────────────
echo ""
echo "🛡️ Ustawiam dodatkowe zmienne środowiskowe dla stabilności CUDA..."

# Zmienne dla stabilności CUDA/NVML
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ─── Uruchom trening YOLO ───────────────────────────────────────────────────
echo ""
echo "🚀 Uruchamiam trening YOLO na 8 GPU z unified dataset (stabilna wersja)..."

python yolo-train-stable.py \
  --dataset_yaml_path "$NEW_DATASET_YAML" \
  --project_dir "./train_out_8gpu_v4" \
  --checkpoint_dir "./ckpt_8gpu_v4" \
  --best_model_dir "./best_8gpu_v4" \
  --model_size l \
  --epochs 200 \
  --batch_size 96 \
  --imgsz 640 \
  --learning_rate 0.01 \
  --workers 1 \
  --device "0,1,2,3,4,5,6,7" \
  --save_period 10 \
  --patience 50 \
  --verbose \
  --disable_ddp

echo -e "\n✅ Zadanie zakończone z kodem: $?"
echo "📊 Lokalizacje wyników:"
echo "  - Logi treningu: ./train_out_8gpu_v4/"
echo "  - Checkpointy: ./ckpt_8gpu_v4/"
echo "  - Najlepszy model: ./best_8gpu_v4/"
