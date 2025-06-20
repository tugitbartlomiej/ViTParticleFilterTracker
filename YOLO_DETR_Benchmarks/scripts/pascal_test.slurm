#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p experimental
#SBATCH --nodelist=pascal
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --job-name=yolo_tar_test
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/Yolo
#SBATCH --output=out/yolo_tar_%j.out
#SBATCH --error=out/yolo_tar_%j.err
#SBATCH --open-mode=append
set -euo pipefail

# ─── lokalizacja archiwum ──────────────────────────────────────────────────
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606.tar.gz
ARCHIVE_SUBDIR=YOLO_03062025/yolo_dataset

# ─── TMPDIR (pascal nie ma --tmp) ──────────────────────────────────────────
TMPDATA=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDATA"
echo "TMPDIR → $TMPDATA"

# ─── Conda ────────────────────────────────────────────────────────────────
module load anaconda
source "$CONDA_SOURCE"          # wymagane przez konfigurację Edena
conda activate yolo_py310       # <- poprawna env-ka

# ─── Rozpakowanie datasetu ────────────────────────────────────────────────
if [[ ! -f $ARCHIVE ]]; then
  echo "ERROR: archive $ARCHIVE not found"; exit 1; fi

echo "Extracting $ARCHIVE …"
tar -xzf "$ARCHIVE" -C "$TMPDATA"

DATASET_PATH="$TMPDATA/$ARCHIVE_SUBDIR/dataset.yaml"
if [[ ! -f $DATASET_PATH ]]; then
  echo "ERROR: dataset.yaml not found inside archive"; exit 1; fi
echo "✓ dataset.yaml → $DATASET_PATH"

# ─── Smoke-test Ultralytics (1 epoka) ──────────────────────────────────────
python yolo-train.py \
    --dataset_yaml_path "$DATASET_PATH" \
    --project_dir "./test_output" \
    --checkpoint_dir "./test_ckpt" \
    --best_model_dir "./test_best" \
    --model_size n --epochs 1 --batch_size 4 --imgsz 640 \
    --workers 2 --device 0 --save_period 1 --patience 1 --verbose

rc=$?
echo -e "\nExit code: $rc"
[[ $rc -eq 0 ]] && rm -rf ./test_output ./test_ckpt ./test_best
exit $rc
