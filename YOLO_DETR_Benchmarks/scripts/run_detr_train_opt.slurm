#!/bin/bash
###############################################################################
#  DETR OPTIMIZED – DDP training on 8×GPU | Klaster Eden (dgx-2)
###############################################################################
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-2
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=detr_ddp_8gpu_opt
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_ddp_8gpu_opt_%j.log
#SBATCH --error=logs/detr_ddp_8gpu_opt_%j.log
#SBATCH --open-mode=append

set -euo pipefail
echo "=== DETR OPTIMIZED DDP 8×GPU on $(hostname) | $(date) ==="
echo "=== Mixed Precision + Performance Optimizations ==="

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
# Liczba GPU
export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29501  # Inny port niż w pierwszym treningu!

# Optymalizacje dla DDP
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN  # Mniej verbose dla wydajności

# Dodatkowe optymalizacje dla DGX-2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# ───────────────────── Uruchomienie treningu DDP OPTIMIZED ───────────────────
echo "🚀 Startuję ZOPTYMALIZOWANY trening DETR z DistributedDataParallel..."
echo "📊 Włączone: Mixed Precision (AMP), zwiększony batch size, więcej workerów"

# WAŻNE: Używamy INNYCH folderów wyjściowych niż w oryginalnym treningu!
torchrun \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    detr_train_optimized.py \
    --images_dir        "$IMAGES_DIR" \
    --annotations_path  "$ANNOT_JSON" \
    --output_dir        "./train_out_ddp_8gpu_opt" \
    --checkpoint_dir    "./ckpt_ddp_8gpu_opt" \
    --best_model_dir    "./best_ddp_8gpu_opt" \
    --epochs            200 \
    --batch_size        16 \
    --num_workers       8 \
    --num_queries       100 \
    --save_interval     5 \
    --patience          20 \
    --augment \
    --model_checkpoint  facebook/detr-resnet-50 \
    --lr                1e-4 \
    --lr_backbone       1e-5 \
    --weight_decay      1e-4 \
    --gradient_accumulation_steps 1 \
    --use_amp \
    --compile_model \
    --resume_training

echo -e "\n✓ Zadanie OPTIMIZED DDP zakończone: kod $?"
