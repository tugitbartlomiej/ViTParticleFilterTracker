#!/usr/bin/env bash
#SBATCH --job-name=k1k1500_oracle
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:30:00
#SBATCH --array=0-7%8
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@300
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008/logs/%x_%A_%a.out

set -euo pipefail
HERE=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008/eden
source "$HERE/common.sh"
cd "$CODE"
"$PY" uruchom_oracle.py \
  --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --archive "$ARCHIVE" \
  --output "$RESULTS/production" --mode production \
  --shard-id "$SLURM_ARRAY_TASK_ID" --num-shards 8 \
  --control-panel "$CONTROL_PANEL" --horizon 500
