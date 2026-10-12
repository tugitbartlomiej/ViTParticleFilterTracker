#!/usr/bin/env bash
#SBATCH --job-name=k1k1500_dmdc
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008/logs/%x_%j.out

set -euo pipefail
HERE=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008/eden
source "$HERE/common.sh"
cd "$CODE"
"$PY" analiza_dmdc.py \
  --production "$RESULTS/production" --control-panel "$CONTROL_PANEL" \
  --output "$RESULTS/dmdc" --rank 32
