#!/usr/bin/env bash
#SBATCH --job-name=k1k_r9_prod
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:30:00
#SBATCH --array=0-7%8
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@300
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009/logs/%x_%A_%a.out

set -euo pipefail
source /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009/eden/common.sh
cd "$CODE"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.free,driver_version --format=csv,noheader
"$PY" uruchom_replay.py \
  --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --archive "$ARCHIVE" \
  --pca "$PCA" --analysis-pca32 "$PCA32" --analysis-pca32-manifest "$PCA32_MANIFEST" \
  --panel "$PANEL" --old-index "$OLD_INDEX" \
  --old-production "$OLD_PRODUCTION" --output "$RESULTS/production" --mode production \
  --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards 8
