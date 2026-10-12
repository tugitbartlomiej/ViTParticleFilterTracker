#!/usr/bin/env bash
#SBATCH --job-name=k1k1500_smoke
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008/logs/%x_%j.out

set -euo pipefail
HERE=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008/eden
source "$HERE/common.sh"
cd "$CODE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PY" uruchom_oracle.py \
  --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --archive "$ARCHIVE" \
  --output "$RESULTS/smoke" --mode smoke --indices-json "$SMOKE_INDICES" \
  --control-panel "$CONTROL_PANEL" --horizon 500
"$PY" waliduj_smoke.py \
  --smoke "$RESULTS/smoke" --indices "$SMOKE_INDICES" \
  --output "$RESULTS/smoke/SMOKE_VALIDATION.json"
