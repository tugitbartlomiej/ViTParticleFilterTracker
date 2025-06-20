#!/bin/bash
###############################################################################
#  DETR OPTIMIZED – DDP training on 4×GPU | Pascal node
###############################################################################
#SBATCH -A transformers_vsc
#SBATCH -p experimental
#SBATCH --nodelist=pascal
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=detr_ddp_4gpu_pascal
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_ddp_4gpu_pascal_%j.log
#SBATCH --error=logs/detr_ddp_4gpu_pascal_%j.log
#SBATCH --open-mode=append

set -euo pipefail
echo "=== DETR OPTIMIZED DDP 4×GPU on $(hostname) | $(date) ==="
echo "=== Pascal node - 4x GPU ==="

# ───────────────────── TMPDIR (NVMe węzła) ───────────────────────────────────
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR → $TMPDIR"

# ───────────────────── Conda env ──────────────────────────────────────────────
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310
echo "[Env] Aktywowane środowisko: $(conda env list | grep '\*')"

# ───────────────────── Sprawdzenie GPU ────────────────────────────────────────
echo "🔍 Sprawdzam dostępne GPU na Pascal..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo ""

# ───────────────────── Rozpakowanie datasetu ─────────────────────────────────
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606.tar.gz
echo "📦 Rozpakowuję $ARCHIVE do $TMPDIR..."
tar -xzf "$ARCHIVE" -C "$TMPDIR"

# Szukanie JSON & images/
ANNOT_JSON=$(find "$TMPDIR" -type f -name 'augmented_coco_20250417_030014.json' | head -n 1 || true)
IMAGES_DIR="$(dirname "$ANNOT_JSON")/images"

if [[ ! -f "$ANNOT_JSON" ]]; then
    echo "❌ Brak pliku augmented_coco_20250417_030014.json!"
    exit 1
fi
if [[ ! -d "$IMAGES_DIR" ]]; then
    echo "❌ Brak katalogu images/ obok JSON-a!"
    exit 1
fi
echo "✅ Dataset OK  |  JSON  → $ANNOT_JSON"
echo "                 images → $IMAGES_DIR"

# ───────────────────── Ustawienia DDP dla 4 GPU ──────────────────────────────
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=29505

# Optymalizacje dla DDP
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN

# ───────────────────── Kopiuj najnowszy checkpoint ───────────────────────────
CHECKPOINT_DIR="./ckpt_ddp_4gpu_pascal"
mkdir -p "$CHECKPOINT_DIR"

# Szukaj najnowszego checkpointa z innych folderów
echo "📋 Szukam najnowszego checkpointa..."
latest_checkpoint=""
latest_epoch=0

for dir in ./ckpt_ddp_*/; do
    if [ -d "$dir" ]; then
        for ckpt in "$dir"checkpoint_epoch_*.pth; do
            if [ -f "$ckpt" ]; then
                epoch=$(echo "$ckpt" | grep -o '[0-9]\+' | tail -1)
                if [ "$epoch" -gt "$latest_epoch" ]; then
                    latest_epoch=$epoch
                    latest_checkpoint=$ckpt
                fi
            fi
        done
    fi
done

if [ -n "$latest_checkpoint" ]; then
    echo "✅ Znaleziono checkpoint z epoki $latest_epoch: $latest_checkpoint"
    cp "$latest_checkpoint" "$CHECKPOINT_DIR/"
    echo "📋 Skopiowano do $CHECKPOINT_DIR"
else
    echo "⚠️  Brak checkpointów - trening rozpocznie się od początku"
fi

# ───────────────────── Uruchomienie treningu ─────────────────────────────────
echo ""
echo "🚀 Startuję trening DETR na Pascal (4 GPU)..."
echo "📊 Konfiguracja:"
echo "   - GPU: 4"
echo "   - Batch size: 8 (4 GPU × 2)"
echo "   - Workers: 4"
echo "   - Mixed Precision: TAK"
echo ""

torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    detr_train_optimized.py \
    --images_dir        "$IMAGES_DIR" \
    --annotations_path  "$ANNOT_JSON" \
    --output_dir        "./train_out_ddp_4gpu_pascal" \
    --checkpoint_dir    "$CHECKPOINT_DIR" \
    --best_model_dir    "./best_ddp_4gpu_pascal" \
    --epochs            200 \
    --batch_size        8 \
    --num_workers       4 \
    --num_queries       100 \
    --save_interval     5 \
    --patience          20 \
    --augment \
    --model_checkpoint  facebook/detr-resnet-50 \
    --lr                1e-4 \
    --lr_backbone       1e-5 \
    --weight_decay      1e-4 \
    --gradient_accumulation_steps 2 \
    --use_amp \
    --resume_training

echo -e "\n✓ Zadanie zakończone: kod $?"
echo "📈 Monitorowanie:"
echo "   tail -f logs/detr_ddp_4gpu_pascal_*.log | grep 'loss='"