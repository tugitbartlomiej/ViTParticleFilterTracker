#!/usr/bin/env bash
#SBATCH --job-name=k1k_patch_joint_smoke
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wspolna_korekta_patch_20261009/logs/%x_%j.out

set -euo pipefail
source /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wspolna_korekta_patch_20261009/eden/common.sh
cd "$CODE"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
"$PY" uruchom_joint.py \
  --mode smoke --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" \
  --archive "$ARCHIVE" --output "$RESULTS" --smoke-indices "$SMOKE_INDICES"
