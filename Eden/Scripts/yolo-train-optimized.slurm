#!/bin/bash

###############################################################################
#  YOLO – 8×GPU | Wersja 5 - Zoptymalizowane kopiowanie
###############################################################################

#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-1
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --job-name=yolo_8gpu_optimized
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo
#SBATCH --output=out/yolo_8gpu_optimized_%j.out
#SBATCH --error=out/yolo_8gpu_optimized_%j.err
#SBATCH --open-mode=append

set -euo pipefail

echo "=== YOLO 8×GPU OPTIMIZED VERSION | $(date) ==="

# --- Definiowanie katalogu tymczasowego ---
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR → $TMPDIR"

# Sprawdź dostępne miejsce
echo "💾 Dostępne miejsce:"
df -h "$TMPDIR"

# ─── Aktywacja środowiska ───────────────────────────────────────────────────
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# ─── Rozpakowanie datasetu ──────────────────────────────────────────────────
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606.tar.gz
echo "📦 Rozpakowuję dataset do $TMPDIR..."
tar -xzf "$ARCHIVE" -C "$TMPDIR"

echo "💾 Miejsce po rozpakowaniu:"
df -h "$TMPDIR"

# ─── Znajdowanie katalogów ──────────────────────────────────────────────────
echo "🔍 Analizuję strukturę..."

YOLO_DATASET_DIR=$(find "$TMPDIR" -name "yolo_dataset" -type d | head -n 1)
IMAGES_DIR=$(find "$TMPDIR" -name "images" -type d | head -n 1)
DATASET_YAML=$(find "$TMPDIR" -name "dataset.yaml" | head -n 1)

echo "📁 Dataset dir: $YOLO_DATASET_DIR"
echo "🖼️ Images dir: $IMAGES_DIR"
echo "📄 Dataset YAML: $DATASET_YAML"

# ─── Unified struktura ──────────────────────────────────────────────────────
UNIFIED_DATASET_DIR="$TMPDIR/unified_dataset"
mkdir -p "$UNIFIED_DATASET_DIR"
echo "🎯 Unified dataset dir: $UNIFIED_DATASET_DIR"

for split in train val test; do
    mkdir -p "$UNIFIED_DATASET_DIR/images/$split"
    mkdir -p "$UNIFIED_DATASET_DIR/labels/$split"
done

# ─── SUPER ZOPTYMALIZOWANE KOPIOWANIE Z HASH MAPĄ ──────────────────────────
echo ""
echo "🚀 SUPER ZOPTYMALIZOWANE KOPIOWANIE Z HASH MAPĄ..."

ALL_IMAGES=($(find "$IMAGES_DIR" -name "*.jpg" -o -name "*.png" | sort))
TOTAL_IMAGES=${#ALL_IMAGES[@]}

echo "📊 Znaleziono $TOTAL_IMAGES obrazów"

# Oblicz podziały
TRAIN_COUNT=$((TOTAL_IMAGES * 80 / 100))
VAL_COUNT=$((TOTAL_IMAGES * 15 / 100))
TEST_COUNT=$((TOTAL_IMAGES - TRAIN_COUNT - VAL_COUNT))

echo "📈 Podział: Train=$TRAIN_COUNT, Val=$VAL_COUNT, Test=$TEST_COUNT"

# ═══ KROK 1: Tworzenie hash mapy etykiet ═══
echo ""
echo "🗂️ Tworzę hash mapę etykiet..."

# Tworzymy asocjacyjną tablicę (hash mapę) dla szybkiego wyszukiwania
declare -A LABEL_MAP

while IFS= read -r -d '' label_path; do
    label_name=$(basename "$label_path")
    label_base="${label_name%.*}"
    LABEL_MAP["$label_base"]="$label_path"
done < <(find "$YOLO_DATASET_DIR" -name "*.txt" -print0)

echo "✅ Utworzono hash mapę dla ${#LABEL_MAP[@]} etykiet"

# ═══ KROK 2: Szybkie kopiowanie z hash mapą ═══
echo ""
echo "📋 Kopiuję pliki z hash mapą (bardzo szybko)..."

counter=0
missing_labels=0

for image_path in "${ALL_IMAGES[@]}"; do
    image_name=$(basename "$image_path")
    image_base="${image_name%.*}"
    
    # Określ split
    if [[ $counter -lt $TRAIN_COUNT ]]; then
        split="train"
    elif [[ $counter -lt $((TRAIN_COUNT + VAL_COUNT)) ]]; then
        split="val"
    else
        split="test"
    fi
    
    # Kopiuj obraz
    cp "$image_path" "$UNIFIED_DATASET_DIR/images/$split/"
    
    # Szybkie wyszukiwanie etykiety w hash mapie O(1)
    if [[ -n "${LABEL_MAP[$image_base]:-}" ]]; then
        cp "${LABEL_MAP[$image_base]}" "$UNIFIED_DATASET_DIR/labels/$split/"
    else
        ((missing_labels++))
    fi
    
    ((counter++))
    
    # Pokaż postęp co 500 plików (częściej)
    if [[ $((counter % 500)) -eq 0 ]]; then
        percent=$((counter * 100 / TOTAL_IMAGES))
        echo "  ✅ $counter/$TOTAL_IMAGES ($percent%) | Brak etykiet: $missing_labels"
        
        # Sprawdź miejsce co 2000 plików
        if [[ $((counter % 2000)) -eq 0 ]]; then
            available_space=$(df "$TMPDIR" | tail -1 | awk '{print $4}')
            echo "    💾 Dostępne: ${available_space}KB"
            sync
        fi
    fi
done

echo ""
echo "✅ Kopiowanie zakończone!"
echo "📊 Podsumowanie:"
echo "  Skopiowano obrazów: $counter"
echo "  Brak etykiet dla: $missing_labels obrazów"

# Zwolnij pamięć hash mapy
unset LABEL_MAP

# ─── Weryfikacja ────────────────────────────────────────────────────────────
echo "🔍 Weryfikacja:"
for split in train val test; do
    images_count=$(find "$UNIFIED_DATASET_DIR/images/$split" -name "*.jpg" -o -name "*.png" | wc -l)
    labels_count=$(find "$UNIFIED_DATASET_DIR/labels/$split" -name "*.txt" | wc -l)
    echo "  $split: $images_count obrazów, $labels_count etykiet"
done

echo "💾 Finalne zużycie miejsca:"
df -h "$TMPDIR"

# ─── Utworzenie dataset.yaml ────────────────────────────────────────────────
NEW_DATASET_YAML="$UNIFIED_DATASET_DIR/dataset.yaml"
cat > "$NEW_DATASET_YAML" << EOF
path: $UNIFIED_DATASET_DIR
train: images/train
val: images/val
test: images/test
nc: 1
names: ['tooltip']
EOF

echo "✅ Dataset.yaml utworzony: $NEW_DATASET_YAML"

# ─── Zmienne środowiskowe ───────────────────────────────────────────────────
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export YOLO_DISABLE_DDP=1

# ─── Uruchom trening ────────────────────────────────────────────────────────
echo ""
echo "🚀 Uruchamiam trening YOLO..."

python yolo-train.py \
  --dataset_yaml_path "$NEW_DATASET_YAML" \
  --project_dir "./train_out_8gpu_optimized" \
  --checkpoint_dir "./ckpt_8gpu_optimized" \
  --best_model_dir "./best_8gpu_optimized" \
  --model_size l \
  --epochs 200 \
  --batch_size 96 \
  --imgsz 640 \
  --learning_rate 0.01 \
  --workers 1 \
  --device "0,1,2,3,4,5,6,7" \
  --save_period 10 \
  --patience 50 \
  --disable_ddp \
  --verbose

echo "✅ Zadanie zakończone!"
