#!/bin/bash
###############################################################################
#  DETR OPTIMIZED – DDP training on 3×GPU | Klaster Eden (dgx-3)
###############################################################################
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-3
#SBATCH --gres=gpu:3
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=detr_ddp_3gpu_dgx3
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_ddp_3gpu_dgx3_%j.log
#SBATCH --error=logs/detr_ddp_3gpu_dgx3_%j.log
#SBATCH --open-mode=append

set -euo pipefail
echo "=== DETR OPTIMIZED DDP 3×GPU on $(hostname) | $(date) ==="
echo "=== Mixed Precision + Performance Optimizations (DGX-3) ==="

# ───────────────────── TMPDIR (NVMe węzła) ───────────────────────────────────
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR → $TMPDIR"

# ───────────────────── Conda env ──────────────────────────────────────────────
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310
echo "[Env] Aktywowane środowisko: $(conda env list | grep '\*')"

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

# ───────────────────── Ustawienia DDP ────────────────────────────────────────
# Liczba GPU - DGX-3 ma 3 wolne GPU
export WORLD_SIZE=3
export MASTER_ADDR=localhost
export MASTER_PORT=29503  # Unikalny port dla DGX-3!

# Optymalizacje dla DDP
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN  # Mniej verbose dla wydajności

# Dodatkowe optymalizacje
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# ───────────────────── Uruchomienie treningu DDP OPTIMIZED ───────────────────
echo "🚀 Startuję ZOPTYMALIZOWANY trening DETR z DistributedDataParallel na DGX-3..."
echo "📊 Włączone: Mixed Precision (AMP), 3×GPU, zwiększony batch size"
echo "💡 Effective batch size: 3 GPU × 16 batch = 48 samples/step"
echo "🎯 Node: DGX-3 (V100 GPUs) - 300 epochs training"

# WAŻNE: Dedykowane foldery dla DGX-3!
torchrun \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    detr_train_optimized.py \
    --images_dir        "$IMAGES_DIR" \
    --annotations_path  "$ANNOT_JSON" \
    --output_dir        "./train_out_ddp_3gpu_dgx3" \
    --checkpoint_dir    "./ckpt_ddp_3gpu_dgx3" \
    --best_model_dir    "./best_ddp_3gpu_dgx3" \
    --epochs            300 \
    --batch_size        16 \
    --num_workers       6 \
    --num_queries       100 \
    --save_interval     10 \
    --patience          30 \
    --augment \
    --model_checkpoint  facebook/detr-resnet-50 \
    --lr                1e-4 \
    --lr_backbone       1e-5 \
    --weight_decay      1e-4 \
    --gradient_accumulation_steps 1 \
    --use_amp \
    --compile_model \
    --resume_training

echo -e "\n✓ Zadanie DGX-3 OPTIMIZED DDP zakończone: kod $?"
