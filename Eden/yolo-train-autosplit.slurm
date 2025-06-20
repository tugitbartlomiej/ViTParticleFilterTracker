#!/bin/bash

###############################################################################
#  YOLO – 8×GPU na DGX-1 | Klaster Eden | Wersja 4 - Auto Split
#  Automatyczny podział datasetu jeśli wszystkie obrazy są w jednym katalogu
###############################################################################

#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-1
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --job-name=yolo_8gpu_autosplit
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo
#SBATCH --output=out/yolo_8gpu_autosplit_%j.out
#SBATCH --error=out/yolo_8gpu_autosplit_%j.err
#SBATCH --open-mode=append

set -euo pipefail

echo "=== YOLO 8×GPU AUTO SPLIT WERSJA on $(hostname) | $(date) ==="

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

YOLO_DATASET_DIR=$(find "$TMPDIR" -name "yolo_dataset" -type d | head -n 1)
IMAGES_DIR=$(find "$TMPDIR" -name "images" -type d | head -n 1)
DATASET_YAML=$(find "$TMPDIR" -name "dataset.yaml" | head -n 1)

if [[ -z "$YOLO_DATASET_DIR" || -z "$IMAGES_DIR" || -z "$DATASET_YAML" ]]; then
    echo "❌ ERROR: Nie znaleziono wymaganych katalogów w archiwum!"
    exit 1
fi

echo "📁 Dataset dir: $YOLO_DATASET_DIR"
echo "🖼️ Images dir: $IMAGES_DIR"
echo "📄 Dataset YAML: $DATASET_YAML"

# Określ katalog docelowy dla unified struktury
UNIFIED_DATASET_DIR="$TMPDIR/unified_dataset"
mkdir -p "$UNIFIED_DATASET_DIR"
echo "🎯 Unified dataset dir: $UNIFIED_DATASET_DIR"

# ─── Auto split strategy ────────────────────────────────────────────────────
echo ""
echo "🤖 Uruchamiam automatyczny podział datasetu..."

# Tworzenie katalogów
for split in train val test; do
    mkdir -p "$UNIFIED_DATASET_DIR/images/$split"
    mkdir -p "$UNIFIED_DATASET_DIR/labels/$split"
done

# Znajdź wszystkie obrazy
ALL_IMAGES=($(find "$IMAGES_DIR" -name "*.jpg" -o -name "*.png" | sort))
TOTAL_IMAGES=${#ALL_IMAGES[@]}

echo "📊 Znaleziono łącznie $TOTAL_IMAGES obrazów"

if [[ $TOTAL_IMAGES -eq 0 ]]; then
    echo "❌ ERROR: Nie znaleziono żadnych obrazów!"
    exit 1
fi

# Znajdź wszystkie etykiety
ALL_LABELS=($(find "$YOLO_DATASET_DIR" -name "*.txt" | sort))
TOTAL_LABELS=${#ALL_LABELS[@]}

echo "🏷️ Znaleziono łącznie $TOTAL_LABELS etykiet"

# Oblicz podziały (80% train, 15% val, 5% test)
TRAIN_COUNT=$((TOTAL_IMAGES * 80 / 100))
VAL_COUNT=$((TOTAL_IMAGES * 15 / 100))
TEST_COUNT=$((TOTAL_IMAGES - TRAIN_COUNT - VAL_COUNT))

echo "📈 Podział datasetu:"
echo "  Train: $TRAIN_COUNT obrazów"
echo "  Val: $VAL_COUNT obrazów"
echo "  Test: $TEST_COUNT obrazów"

# Kopiuj obrazy i etykiety z podziałem
echo ""
echo "📋 Kopiuję pliki z automatycznym podziałem..."

counter=0
for image_path in "${ALL_IMAGES[@]}"; do
    image_name=$(basename "$image_path")
    image_base="${image_name%.*}"
    
    # Określ split na podstawie licznika
    if [[ $counter -lt $TRAIN_COUNT ]]; then
        split="train"
    elif [[ $counter -lt $((TRAIN_COUNT + VAL_COUNT)) ]]; then
        split="val"
    else
        split="test"
    fi
    
    # Kopiuj obraz
    cp "$image_path" "$UNIFIED_DATASET_DIR/images/$split/"
    
    # Znajdź i kopiuj odpowiadającą etykietę
    for label_path in "${ALL_LABELS[@]}"; do
        label_name=$(basename "$label_path")
        label_base="${label_name%.*}"
        
        if [[ "$image_base" == "$label_base" ]]; then
            cp "$label_path" "$UNIFIED_DATASET_DIR/labels/$split/"
            break
        fi
    done
    
    ((counter++))
    
    # Pokaż postęp co 1000 plików
    if [[ $((counter % 1000)) -eq 0 ]]; then
        echo "  Przetworzono $counter/$TOTAL_IMAGES obrazów..."
    fi
done

# ─── Weryfikacja podziału ───────────────────────────────────────────────────
echo ""
echo "🔍 Weryfikuję podział datasetu..."

for split in train val test; do
    images_count=$(find "$UNIFIED_DATASET_DIR/images/$split" -name "*.jpg" -o -name "*.png" | wc -l)
    labels_count=$(find "$UNIFIED_DATASET_DIR/labels/$split" -name "*.txt" | wc -l)
    echo "📊 $split: $images_count obrazów, $labels_count etykiet"
    
    if [[ $images_count -eq $labels_count ]]; then
        echo "    ✅ Liczby się zgadzają"
    else
        echo "    ⚠️ Różnica w liczbie plików!"
    fi
done

# ─── Tworzenie nowego dataset.yaml ──────────────────────────────────────────
NEW_DATASET_YAML="$UNIFIED_DATASET_DIR/dataset.yaml"
cat > "$NEW_DATASET_YAML" << EOF
# YOLOv8 Dataset Configuration - Auto Split
path: $UNIFIED_DATASET_DIR
train: images/train
val: images/val
test: images/test
nc: 1
names: ['tooltip']

# Auto split info:
# Total images: $TOTAL_IMAGES
# Train: $TRAIN_COUNT (80%)
# Val: $VAL_COUNT (15%)
# Test: $TEST_COUNT (5%)
EOF

echo "✅ Nowy dataset.yaml utworzony: $NEW_DATASET_YAML"
echo "📄 Konfiguracja:"
cat "$NEW_DATASET_YAML"

# ─── Dodatkowe zabezpieczenia środowiska ────────────────────────────────────
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export YOLO_DISABLE_DDP=1

# ─── Uruchom trening YOLO ───────────────────────────────────────────────────
echo ""
echo "🚀 Uruchamiam trening YOLO z automatycznym podziałem..."

python yolo-train.py \
  --dataset_yaml_path "$NEW_DATASET_YAML" \
  --project_dir "./train_out_8gpu_autosplit" \
  --checkpoint_dir "./ckpt_8gpu_autosplit" \
  --best_model_dir "./best_8gpu_autosplit" \
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

echo -e "\n✅ Zadanie zakończone z kodem: $?"
echo "📊 Lokalizacje wyników:"
echo "  - Logi treningu: ./train_out_8gpu_autosplit/"
echo "  - Checkpointy: ./ckpt_8gpu_autosplit/"
echo "  - Najlepszy model: ./best_8gpu_autosplit/"
