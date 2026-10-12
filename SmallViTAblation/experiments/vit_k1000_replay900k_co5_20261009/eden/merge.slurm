#!/usr/bin/env bash
#SBATCH --job-name=k1k_r9_merge
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009/logs/%x_%j.out

set -euo pipefail
source /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009/eden/common.sh
cd "$CODE"
"$PY" scal_replay.py \
  --production "$RESULTS/production" --panel-contract "$PANEL" \
  --analysis-pca32-manifest "$PCA32_MANIFEST" --output "$RESULTS/merged"
"$PY" export_dense_timeline.py \
  --source-manifest "$RESULTS/merged/VISUALIZATION_MANIFEST.json" \
  --output "$RESULTS/visualization_dense" --steps-key steps
