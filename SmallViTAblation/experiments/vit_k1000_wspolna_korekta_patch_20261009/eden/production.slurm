#!/usr/bin/env bash
#SBATCH --job-name=k1k_patch_joint_train
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@300
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wspolna_korekta_patch_20261009/logs/%x_%j.out

set -euo pipefail
source /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wspolna_korekta_patch_20261009/eden/common.sh
cd "$CODE"
"$PY" uruchom_joint.py \
  --mode train --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" \
  --archive "$ARCHIVE" --output "$RESULTS" \
  --old-evaluation-summary "$OLD_EVALUATION" \
  --capture-batch-size 16 --microbatch-size 32 --eval-batch-size 32
