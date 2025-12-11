#!/bin/bash

###############################################################################
#  YOLO – 8×GPU na DGX-1 | Klaster Eden
#  Wersja z Pythonem do poprawy ścieżek
###############################################################################

#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-1
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --job-name=yolo_8gpu_dgx1
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo
#SBATCH --output=out/yolo_8gpu_%j.out
#SBATCH --error=out/yolo_8gpu_%j.err
#SBATCH --open-mode=append

set -euo pipefail

echo "=== YOLO 8×GPU on $(hostname) | $(date) ==="

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

# ─── Poprawka ścieżek za pomocą skryptu Python ─────────────────────────────────
echo "🔧 Poprawiam ścieżki w konfiguracji datasetu..."

# Sprawdź czy skrypt istnieje w katalogu roboczym
FIX_SCRIPT="./fix_dataset_paths.py"
if [[ ! -f "$FIX_SCRIPT" ]]; then
    echo "❌ ERROR: Skrypt $FIX_SCRIPT nie istnieje!"
    echo "Skopiuj plik fix_dataset_paths.py do katalogu roboczego."
    exit 1
fi

# Uruchom skrypt do poprawy ścieżek
python "$FIX_SCRIPT" --tmpdir "$TMPDIR" --verify

# Znajdź zaktualizowany plik dataset.yaml
DATASET_PATH=$(find "$TMPDIR" -name dataset.yaml | head -n 1)
if [[ -z "$DATASET_PATH" ]]; then
    echo "❌ ERROR: dataset.yaml not found after path fixing!"
    exit 1
fi

echo "✅ Poprawione dataset.yaml → $DATASET_PATH"

# ─── Uruchom trening YOLO ───────────────────────────────────────────────────
echo ""
echo "🚀 Uruchamiam trening YOLO na 8 GPU z wyłączonym DDP..."

python yolo-train.py \
  --dataset_yaml_path "$DATASET_PATH" \
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
  --verbose \
  --disable_ddp

echo -e "\n✓ Zadanie zakończone z kodem: $?"
