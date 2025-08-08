#!/bin/bash
###############################################################################
#  DETR FINE-TUNING – Background class addition with DDP on 4×GPU | Any DGX
###############################################################################
#SBATCH -A transformers_vsc
#SBATCH -p short
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=detr_finetune_bg_dgx
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_finetune_bg_dgx_%j.log
#SBATCH --error=logs/detr_finetune_bg_dgx_%j.log
#SBATCH --open-mode=append

set -euo pipefail
echo "=== DETR FINE-TUNING Background Class on $(hostname) | $(date) ==="
echo "=== Fine-tuning pretrained model with background class - 4 GPU ==="

# ───────────────────── TMPDIR (NVMe węzła) ───────────────────────────────────
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR → $TMPDIR"

# ───────────────────── Conda env ──────────────────────────────────────────────
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310
echo "[Env] Aktywowane środowisko: $(conda env list | grep '\*')"

# ───────────────────── FINE-TUNING - tylko background dataset ─────────────────────────────────
# UWAGA: Fine-tuning wymaga TYLKO nowych danych background!
# Stary dataset NIE jest potrzebny - model już został wytrenowany i zapisany w checkpoincie

# ───────────────────── Rozpakowanie datasetu background do TMPDIR ────────────────────
BACKGROUND_ZIP="/mnt/evafs/faculty/home/bpiotrowski/datasets/DETR_Background_dataset.zip"
BACKGROUND_TMPDIR="$TMPDIR/background_dataset"

echo "📦 Rozpakowuję dataset background $BACKGROUND_ZIP do $BACKGROUND_TMPDIR..."
mkdir -p "$BACKGROUND_TMPDIR"

if [[ ! -f "$BACKGROUND_ZIP" ]]; then
    echo "❌ Brak background ZIP: $BACKGROUND_ZIP!"
    exit 1
fi

# Rozpakuj background dataset
unzip -q "$BACKGROUND_ZIP" -d "$BACKGROUND_TMPDIR"

# ZIP zawiera bezpośrednio foldery: annotations/, train/, val/
BACKGROUND_BASE="$BACKGROUND_TMPDIR"

# Ścieżki do background datasetu
BACKGROUND_ANNOT_JSON="$BACKGROUND_BASE/annotations/train_annotations.json"
BACKGROUND_IMAGES_DIR="$BACKGROUND_BASE/train"

if [[ ! -f "$BACKGROUND_ANNOT_JSON" ]]; then
    echo "❌ Brak background annotations: $BACKGROUND_ANNOT_JSON!"
    echo "📋 Struktura rozpakowanego background datasetu:"
    find "$BACKGROUND_TMPDIR" -type f -name "*.json" | head -10
    exit 1
fi
if [[ ! -d "$BACKGROUND_IMAGES_DIR" ]]; then
    echo "❌ Brak background images directory: $BACKGROUND_IMAGES_DIR!"
    echo "📋 Dostępne foldery:"
    find "$BACKGROUND_TMPDIR" -type d | head -10
    exit 1
fi

# Sprawdź liczbę obrazów background
BG_IMAGE_COUNT=$(find "$BACKGROUND_IMAGES_DIR" -name "*.jpg" -o -name "*.png" | wc -l)
echo "✅ Background dataset OK  |  JSON  → $BACKGROUND_ANNOT_JSON"
echo "                           images → $BACKGROUND_IMAGES_DIR"
echo "                           Liczba obrazów background: $BG_IMAGE_COUNT"

# ───────────────────── Ścieżka do checkpointa pretrenowanego modelu ────────────────────────
CHECKPOINT_PATH="/mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints/final_checkpoint.pth"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "❌ Brak checkpointa w: $CHECKPOINT_PATH"
    echo "📋 Dostępne checkpointy:"
    ls -la /mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints/checkpoint_epoch_*.pth || echo "Brak checkpointów"
    ls -la /mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints/final_checkpoint.pth || echo "Brak final_checkpoint.pth"
    exit 1
fi
echo "✅ Checkpoint pretrenowanego modelu  → $CHECKPOINT_PATH"

# ───────────────────── Ustawienia DDP ────────────────────────────────────────
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=29505  # Port dla DGX węzłów

# Optymalizacje dla DDP
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN

# Dodatkowe optymalizacje dla DGX-2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# ───────────────────── Uruchomienie fine-tuningu DDP ───────────────────────
echo "🚀 Startuję FINE-TUNING DETR z dodatkową klasą 'background'..."
echo "📊 Strategia: Fine-tuning z zamrożonym backbone, różne LR dla warstw"
echo "🎯 Checkpoint: $CHECKPOINT_PATH"
echo "📁 Model zostanie zapisany jako: DETR_fine_tuned_background_model"
echo "🖥️  DGX 4 GPU - dostosowane parametry"

# Uruchomienie fine-tuningu (based on original working script pattern)
torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    detr_background_finetune_fixed.py \
    --checkpoint_path           "$CHECKPOINT_PATH" \
    --background_images_dir     "$BACKGROUND_IMAGES_DIR" \
    --background_annotations_path "$BACKGROUND_ANNOT_JSON" \
    --output_dir                "./output_DETR_fine_tuned_background_model" \
    --checkpoint_dir            "/mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints/DETR_fine_tuned_background_model" \
    --best_model_dir            "./best_models/DETR_fine_tuned_background_model" \
    --epochs                    15 \
    --batch_size                4 \
    --num_workers               1 \
    --num_queries               100 \
    --save_interval             2 \
    --patience                  8 \
    --augment \
    --model_checkpoint          facebook/detr-resnet-50 \
    --lr                        1e-4 \
    --lr_backbone               1e-5 \
    --weight_decay              1e-4 \
    --gradient_accumulation_steps 1 \
    --use_amp \
    --compile_model \
    --resume_training

echo -e "\n✓ Fine-tuning DETR zakończony: kod $?"

# ───────────────────── Czyszczenie TMPDIR ─────────────────────────────────────
echo "🧹 Czyszczenie TMPDIR..."
rm -rf "$TMPDIR"
echo "✅ Zadanie zakończone pomyślnie!"