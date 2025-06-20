#!/bin/bash
###############################################################################
#  DETR – DDP training on 8×GPU | Klaster Eden (dgx-3)
###############################################################################
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-3
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --job-name=detr_ddp_8gpu
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_ddp_8gpu_%j.log
#SBATCH --error=logs/detr_ddp_8gpu_%j.log
#SBATCH --open-mode=append

set -euo pipefail
echo "=== DETR DDP 8×GPU on $(hostname) | $(date) ==="

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
export MASTER_PORT=29500

# Optymalizacje dla DDP
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=INFO  # Możesz zmienić na WARN po debugowaniu

# ───────────────────── Uruchomienie treningu DDP ─────────────────────────────
echo "🚀 Startuję trening DETR z DistributedDataParallel..."

# Zapisz nowy skrypt Python jako detr_train_ddp.py przed uruchomieniem!
# Użyj torchrun (preferowane) lub python -m torch.distributed.launch
torchrun \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    detr_annotator_train_corrected.py \
    --images_dir        "$IMAGES_DIR" \
    --annotations_path  "$ANNOT_JSON" \
    --output_dir        "./train_out_ddp_8gpu" \
    --checkpoint_dir    "./ckpt_ddp_8gpu" \
    --best_model_dir    "./best_ddp_8gpu" \
    --epochs            200 \
    --batch_size        8 \
    --num_workers       2 \
    --num_queries       100 \
    --save_interval     5 \
    --patience          20 \
    --augment \
    --model_checkpoint  facebook/detr-resnet-50 \
    --lr                1e-4 \
    --lr_backbone       1e-5 \
    --weight_decay      1e-4 \
    --resume_training

# Alternatywnie, jeśli torchrun nie działa:
# python -m torch.distributed.launch \
#     --nproc_per_node=$WORLD_SIZE \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     detr_train_ddp.py \
#     [... reszta argumentów ...]

echo -e "\n✓ Zadanie DDP zakończone: kod $?"