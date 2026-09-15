#!/bin/bash
# One H100 prepares frozen DINOv2 features and the immutable campaign archive.
# Required: PROJECT_DIR, INDEX_DIR, PREPARED_DIR, CACHE_ROOT, PY, DINO_HUB_REPO.
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --job-name=static_prep
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${PROJECT_DIR:?}" "${INDEX_DIR:?}" "${PREPARED_DIR:?}" "${CACHE_ROOT:?}"
: "${PY:?}" "${DINO_HUB_REPO:?}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
unset PYTHONPATH
cd "$PROJECT_DIR"
"$PY" -B -c 'import torch; assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0), flush=True)'
"$PY" -B tools/prepare_tracking_static_pilot.py \
  --index-dir "$INDEX_DIR" --out "$PREPARED_DIR" --cache-root "$CACHE_ROOT" \
  --dino-hub-repo "$DINO_HUB_REPO" --selection-seed 42 --batch-size 64
test -s "${PREPARED_DIR}.tar" -a -s "${PREPARED_DIR}.tar.sha256.json"
