#!/bin/bash
#SBATCH --job-name=vit1z256
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_1z256/logs/%x_%A_%a.out
#SBATCH --error=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_1z256/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_1z256
PY=/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python
DEPS=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-28_maly_vit_od_zera/deps
STEPS=${1:-500}
RESULT_SUBDIR=${2:-results}
export PYTHONPATH="$DEPS"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
mkdir -p "$ROOT/logs" "$ROOT/$RESULT_SUBDIR"
cd "$ROOT/code"
"$PY" -u -B train_variant.py --variant "${SLURM_ARRAY_TASK_ID}" --steps "$STEPS" --data "$ROOT/input_pack" --checkpoint "$ROOT/assets/MALY_VIT_K8000_s47_losowe.pth" --output "$ROOT/$RESULT_SUBDIR"
