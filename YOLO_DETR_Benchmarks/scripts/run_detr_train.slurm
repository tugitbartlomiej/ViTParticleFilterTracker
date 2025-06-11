#!/bin/bash
###############################################################################
#  DETR – prosty trening na 8×GPU | Klaster Eden (dgx-2)
###############################################################################
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-3          # <-- wymusza węzeł dgx-2
#SBATCH --gres=gpu:8              # rezerwuje wszystkie 8 GPU węzła
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --job-name=detr_7gpu_simple
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_8gpu_%j.log   # stdout + stderr → jeden plik
#SBATCH --error=logs/detr_8gpu_%j.log    # (zbędne, ale jawne)
#SBATCH --open-mode=append

set -euo pipefail
echo "=== DETR 8×GPU on $(hostname) | $(date) ==="

# ───────────────────── TMPDIR (NVMe węzła) ───────────────────────────────────
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR → $TMPDIR"

# ───────────────────── Conda env (yolo_py310) ────────────────────────────────
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310
echo "[Env] Aktywowane środowisko: $(conda env list | grep '\*')"

# ───────────────────── Rozpakowanie datasetu ─────────────────────────────────
#ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606.tar.gz
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606_test.tar.gz
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

# ───────────────────── Uruchomienie treningu DETR ────────────────────────────
echo "🚀 Startuję trening DETR…"
python detr_annotator_train_corrected.py \
  --images_dir        "$IMAGES_DIR" \
  --annotations_path  "$ANNOT_JSON" \
  --output_dir        "./train_out_8gpu" \
  --checkpoint_dir    "./ckpt_8gpu" \
  --best_model_dir    "./best_8gpu" \
  --epochs            200 \
  --batch_size        32 \
  --num_workers       24 \
  --num_queries       100 \
  --save_interval     5 \
  --patience          20 \
  --augment \
  --model_checkpoint  facebook/detr-resnet-50 \
  --lr                1e-4 \
  --lr_backbone       1e-5 \
  --weight_decay      1e-4 


echo -e "\n✓ Zadanie zakończone: kod $?"
