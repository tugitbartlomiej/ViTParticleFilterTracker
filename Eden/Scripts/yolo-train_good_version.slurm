#!/bin/bash
###############################################################################
#  YOLO – 8×GPU na DGX-1 | Wersja Finalna
###############################################################################

#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-1
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --job-name=yolo_8gpu_dgx1
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo
#SBATCH --output=out/yolo_8gpu_dgx1_%j.out
#SBATCH --error=out/yolo_8gpu_dgx1_%j.err
#SBATCH --open-mode=append

set -euo pipefail

echo "============================================================"
echo "🚀 YOLO 8×GPU (Target: dgx-1) | $(date)"
echo "============================================================"

# --- Przygotowanie Środowiska ---
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "📁 TMPDIR → $TMPDIR"

# ─── Aktywacja środowiska Conda ─────────────────────────────────────────────
echo "🐍 Aktywuję środowisko Conda (z PyTorch +cu121)..."
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# ─── Automatyczne Przygotowanie Danych ──────────────────────────────────────
echo "📦 Uruchamiam skrypt przygotowujący i naprawiający dataset..."
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606.tar.gz
tar -xzf "$ARCHIVE" -C "$TMPDIR"
python prepare_dataset.py --tmpdir "$TMPDIR" --unified_dir "$TMPDIR/unified_dataset"

# Weryfikacja, czy skrypt prepare_dataset.py zadziałał
DATASET_YAML="$TMPDIR/unified_dataset/dataset.yaml"
if [[ ! -f "$DATASET_YAML" ]]; then
    echo "❌ KRYTYCZNY BŁĄD: Skrypt prepare_dataset.py nie utworzył pliku dataset.yaml!"
    exit 1
fi
echo "✅ Dataset przygotowany i zweryfikowany: $DATASET_YAML"

# ─── Uruchomienie Treningu YOLO w trybie DDP ────────────────────────────────
echo "🚀 Uruchamiam trening YOLO na 8 GPU w szybkim trybie DDP..."

python yolo-train.py \
  --dataset_yaml_path "$DATASET_YAML" \
  --project_dir "./train_out_8gpu_dgx1" \
  --checkpoint_dir "./ckpt_8gpu_dgx1" \
  --best_model_dir "./best_8gpu_dgx1" \
  --model_size l \
  --epochs 200 \
  --batch_size 96 \
  --imgsz 640 \
  --learning_rate 0.01 \
  --workers 24 \
  --device "0,1,2,3,4,5,6,7" \
  --save_period 10 \
  --patience 50 \
  --verbose

EXIT_CODE=$?
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "🎉 Trening Zakończony Pomyślnie!"
else
    echo "❌ Trening Zakończony z Błędem (kod: $EXIT_CODE)"
fi
echo "============================================================"
exit $EXIT_CODE