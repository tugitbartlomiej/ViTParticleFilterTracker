#!/bin/bash
###############################################################################
#  YOLO – Elastyczny skrypt na 8×GPU | Klaster Eden
###############################################################################
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --job-name=yolo_8gpu_flexible
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo
#SBATCH --output=out/yolo_8gpu_%j.out
#SBATCH --error=out/yolo_8gpu_%j.err
#SBATCH --open-mode=append

set -euo pipefail

echo "=== YOLO 8×GPU on $(hostname) | $(date) ==="

# --- Katalog tymczasowy ---
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR → $TMPDIR"

# ─── Aktywacja środowiska Conda ─────────────────────────────────────────────
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# ─── Automatyczne Przygotowanie Danych ──────────────────────────────────────
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606.tar.gz
echo "📦 Rozpakowuję dataset z $ARCHIVE do $TMPDIR..."
tar -xzf "$ARCHIVE" -C "$TMPDIR"
python prepare_dataset.py --tmpdir "$TMPDIR" --unified_dir "$TMPDIR/unified_dataset"

# Weryfikacja, czy skrypt prepare_dataset.py zadziałał
DATASET_YAML="$TMPDIR/unified_dataset/dataset.yaml"
if [[ ! -f "$DATASET_YAML" ]]; then
    echo "❌ KRYTYCZNY BŁĄD: Skrypt prepare_dataset.py nie utworzył pliku dataset.yaml!"
    exit 1
fi
echo "✅ Dataset przygotowany i zweryfikowany: $DATASET_YAML"

# ─── Uruchomienie Treningu YOLO ──────────────────────────────────────────────
echo "🚀 Uruchamiam trening YOLO na 8 GPU w trybie DDP..."

python yolo-train.py \
  --dataset_yaml_path "$DATASET_YAML" \
  --project_dir "./train_out_8gpu" \
  --checkpoint_dir "./ckpt_8gpu" \
  --best_model_dir "./best_8gpu" \
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

echo -e "\n✓ Zadanie zakończone z kodem: $?"