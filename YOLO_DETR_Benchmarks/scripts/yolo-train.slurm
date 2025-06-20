#!/bin/bash
###############################################################################
#  YOLO – 6×GPU na DGX-3 | Klaster Eden
###############################################################################
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-3
#SBATCH --gres=gpu:6
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --job-name=yolo_6gpu_dgx3
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo
#SBATCH --output=out/yolo_6gpu_%j.out
#SBATCH --error=out/yolo_6gpu_%j.err
#SBATCH --open-mode=append
set -euo pipefail

echo "=== YOLO 6×GPU on $(hostname) | $(date) ==="

# Bezpieczne ustawienie TMPDIR
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    TMPDIR="$SLURM_TMPDIR"
else
    TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"
    mkdir -p "$TMPDIR"
fi
echo "TMPDIR → $TMPDIR"

# ─── WAŻNE: Ustawienie zmiennych środowiskowych dla CUDA/NCCL ─────────────
# Rozwiązuje problemy z komunikacją między GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# Wyłącz NVML który powoduje problemy
export CUDA_DISABLE_NVML=1

# Ustaw backend komunikacji na GLOO zamiast NCCL jeśli będą dalsze problemy
# export TORCH_DISTRIBUTED_BACKEND=gloo

# ─── Aktywacja środowiska Conda ─────────────────────────────────────────────
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# ─── Weryfikacja środowiska ─────────────────────────────────────────────────
echo "Weryfikacja środowiska Python..."
python -c "import torch, ultralytics, numpy; print(f'PyTorch: {torch.__version__}, Ultralytics: {ultralytics.__version__}, NumPy: {numpy.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')"

# ─── Rozpakowanie datasetu ──────────────────────────────────────────────────
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606.tar.gz
echo "Rozpakowuję dataset do $TMPDIR..."
tar -xzf "$ARCHIVE" -C "$TMPDIR"

# ─── Znajdź dataset.yaml ────────────────────────────────────────────────────
DATASET_PATH=$(find "$TMPDIR" -name dataset.yaml | head -n 1)
if [[ -z "$DATASET_PATH" ]]; then
    echo "ERROR: dataset.yaml not found in archive!"
    exit 1
fi
echo "dataset.yaml → $DATASET_PATH"

# ─── OPCJA 1: Trening na pojedynczym GPU (stabilniejsze) ───────────────────
# Jeśli multi-GPU dalej nie działa, odkomentuj tę sekcję i zakomentuj OPCJA 2
# echo "Uruchamiam trening YOLO na GPU 0..."
# python yolo-train.py \
#   --dataset_yaml_path "$DATASET_PATH" \
#   --project_dir "./train_out_1gpu" \
#   --checkpoint_dir "./ckpt_1gpu" \
#   --best_model_dir "./best_1gpu" \
#   --model_size l \
#   --epochs 200 \
#   --batch_size 16 \
#   --imgsz 640 \
#   --learning_rate 0.01 \
#   --workers 8 \
#   --device "0" \
#   --save_period 10 \
#   --patience 50 \
#   --verbose

# ─── OPCJA 2: Trening na 6 GPU z poprawkami ────────────────────────────────
echo "Uruchamiam trening YOLO na 6 GPU..."
python yolo-train.py \
  --dataset_yaml_path "$DATASET_PATH" \
  --project_dir "./train_out_6gpu" \
  --checkpoint_dir "./ckpt_6gpu" \
  --best_model_dir "./best_6gpu" \
  --model_size l \
  --epochs 200 \
  --batch_size 96 \
  --imgsz 640 \
  --learning_rate 0.01 \
  --workers 24 \
  --device "0,1,2,3,4,5" \
  --save_period 10 \
  --patience 50 \
  --verbose

echo -e "\nZadanie zakończone z kodem: $?"