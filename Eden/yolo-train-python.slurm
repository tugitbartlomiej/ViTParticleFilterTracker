#!/bin/bash

###############################################################################
#  YOLO 8×GPU + Python Dataset Preparation | Eden DGX-1
#  Wersja 6 - Python + SLURM hybrid approach
###############################################################################

#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-1
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --job-name=yolo_8gpu_python
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo
#SBATCH --output=out/yolo_8gpu_python_%j.out
#SBATCH --error=out/yolo_8gpu_python_%j.err
#SBATCH --open-mode=append

set -euo pipefail

echo "=" * 80
echo "🚀 YOLO 8×GPU + PYTHON DATASET PREP | $(date)"
echo "=" * 80

# --- Konfiguracja środowiska ---
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "📁 TMPDIR → $TMPDIR"

UNIFIED_DATASET_DIR="$TMPDIR/unified_dataset"
echo "🎯 Unified dataset dir → $UNIFIED_DATASET_DIR"

# Sprawdź dostępne miejsce
echo ""
echo "💾 Dostępne miejsce na start:"
df -h "$TMPDIR"

# ─── Aktywacja conda environment ────────────────────────────────────────────
echo ""
echo "🐍 Aktywuję conda environment..."
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# Sprawdź czy Python ma wymagane biblioteki
echo "📦 Sprawdzam wymagane biblioteki Python..."
python -c "import yaml, tqdm, pathlib; print('✅ Wszystkie biblioteki dostępne')" || {
    echo "📦 Instaluję brakujące biblioteki..."
    pip install pyyaml tqdm
}

# ─── Rozpakowanie archiwum ──────────────────────────────────────────────────
echo ""
echo "📦 Rozpakowuję dataset..."
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606.tar.gz

if [[ ! -f "$ARCHIVE" ]]; then
    echo "❌ Nie znaleziono archiwum: $ARCHIVE"
    exit 1
fi

tar -xzf "$ARCHIVE" -C "$TMPDIR"

echo "💾 Miejsce po rozpakowaniu:"
df -h "$TMPDIR"

# ─── Uruchomienie Python skryptu przygotowania datasetu ─────────────────────
echo ""
echo "🐍 Uruchamiam Python skrypt przygotowania datasetu..."

python prepare_dataset.py \
    --tmpdir "$TMPDIR" \
    --unified_dir "$UNIFIED_DATASET_DIR" \
    --seed 42

# Sprawdź kod wyjścia Python skryptu
PYTHON_EXIT_CODE=$?
if [[ $PYTHON_EXIT_CODE -ne 0 ]]; then
    echo "❌ Python skrypt przygotowania datasetu się nie powiódł (kod: $PYTHON_EXIT_CODE)"
    exit 1
fi

echo "✅ Python skrypt przygotowania datasetu zakończony pomyślnie!"

# ─── Weryfikacja wyników ────────────────────────────────────────────────────
echo ""
echo "🔍 Weryfikuję wyniki przygotowania datasetu..."

DATASET_YAML="$UNIFIED_DATASET_DIR/dataset.yaml"
if [[ ! -f "$DATASET_YAML" ]]; then
    echo "❌ Nie znaleziono dataset.yaml: $DATASET_YAML"
    exit 1
fi

echo "📄 Dataset YAML znaleziony: $DATASET_YAML"
echo "📄 Zawartość dataset.yaml:"
cat "$DATASET_YAML"

# Sprawdź czy istnieją katalogi
for split in train val test; do
    images_dir="$UNIFIED_DATASET_DIR/images/$split"
    labels_dir="$UNIFIED_DATASET_DIR/labels/$split"
    
    if [[ -d "$images_dir" && -d "$labels_dir" ]]; then
        images_count=$(find "$images_dir" -name "*.jpg" -o -name "*.png" | wc -l)
        labels_count=$(find "$labels_dir" -name "*.txt" | wc -l)
        echo "📊 $split: $images_count obrazów, $labels_count etykiet"
    else
        echo "❌ Brak katalogów dla $split"
        exit 1
    fi
done

echo "💾 Finalne zużycie miejsca po przygotowaniu:"
df -h "$TMPDIR"

# ─── Konfiguracja środowiska CUDA ───────────────────────────────────────────
echo ""
echo "⚙️ Konfiguruję środowisko CUDA..."

export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export YOLO_DISABLE_DDP=1

echo "✅ Zmienne środowiskowe CUDA ustawione"

# ─── Uruchomienie treningu YOLO ─────────────────────────────────────────────
echo ""
echo "🚀 Uruchamiam trening YOLO na 8 GPU..."
echo "📄 Dataset: $DATASET_YAML"

# Stwórz katalogi wyjściowe
OUTPUT_PREFIX="python_8gpu_$(date +%Y%m%d_%H%M%S)"
PROJECT_DIR="./train_out_${OUTPUT_PREFIX}"
CHECKPOINT_DIR="./ckpt_${OUTPUT_PREFIX}"
BEST_MODEL_DIR="./best_${OUTPUT_PREFIX}"

mkdir -p "$PROJECT_DIR" "$CHECKPOINT_DIR" "$BEST_MODEL_DIR"

echo "📁 Katalogi wyjściowe:"
echo "  Project: $PROJECT_DIR"
echo "  Checkpoints: $CHECKPOINT_DIR"  
echo "  Best model: $BEST_MODEL_DIR"

# Uruchom trening
echo ""
echo "🎯 Rozpoczynam trening YOLOv8..."

python yolo-train.py \
  --dataset_yaml_path "$DATASET_YAML" \
  --project_dir "$PROJECT_DIR" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --best_model_dir "$BEST_MODEL_DIR" \
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

TRAINING_EXIT_CODE=$?

# ─── Podsumowanie ───────────────────────────────────────────────────────────
echo ""
echo "=" * 80
echo "📊 PODSUMOWANIE ZADANIA"
echo "=" * 80

if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then
    echo "🎉 Trening YOLO zakończony pomyślnie!"
else
    echo "❌ Trening YOLO zakończony z błędem (kod: $TRAINING_EXIT_CODE)"
fi

echo ""
echo "📁 Lokalizacje wyników:"
echo "  📊 Logi treningu: $PROJECT_DIR"
echo "  💾 Checkpointy: $CHECKPOINT_DIR"
echo "  🏆 Najlepszy model: $BEST_MODEL_DIR"
echo "  📄 Dataset YAML: $DATASET_YAML"

echo ""
echo "⏱️ Zadanie zakończone: $(date)"
echo "🖥️ Host: $(hostname)"
echo "👤 User: $USER"
echo "🆔 Job ID: $SLURM_JOB_ID"

exit $TRAINING_EXIT_CODE
