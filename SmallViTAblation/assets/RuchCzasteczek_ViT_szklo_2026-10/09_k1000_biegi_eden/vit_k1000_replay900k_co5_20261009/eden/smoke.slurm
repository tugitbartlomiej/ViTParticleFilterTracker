#!/usr/bin/env bash
#SBATCH --job-name=k1k_r9_smoke
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009/logs/%x_%j.out

set -euo pipefail
source /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009/eden/common.sh
cd "$CODE"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.free,driver_version --format=csv,noheader
"$PY" fit_analysis_pca32.py \
  --manifest "$MANIFEST" --old-index "$OLD_INDEX" --old-production "$OLD_PRODUCTION" \
  --panel "$PANEL" --output "$PCA32" --manifest-output "$PCA32_MANIFEST"
"$PY" uruchom_replay.py \
  --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --archive "$ARCHIVE" \
  --pca "$PCA" --analysis-pca32 "$PCA32" --analysis-pca32-manifest "$PCA32_MANIFEST" \
  --panel "$PANEL" --old-index "$OLD_INDEX" \
  --old-production "$OLD_PRODUCTION" --output "$RESULTS/smoke" --mode smoke \
  --indices-json "$SMOKE_INDICES"
"$PY" waliduj_smoke.py --output "$RESULTS/smoke" --indices "$SMOKE_INDICES" \
  --panel "$PANEL" --analysis-pca32-manifest "$PCA32_MANIFEST"
