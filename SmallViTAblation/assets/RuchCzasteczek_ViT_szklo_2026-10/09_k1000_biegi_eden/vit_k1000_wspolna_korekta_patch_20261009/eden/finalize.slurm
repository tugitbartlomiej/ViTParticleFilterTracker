#!/usr/bin/env bash
#SBATCH --job-name=k1k_patch_joint_final
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wspolna_korekta_patch_20261009/logs/%x_%j.out

set -euo pipefail
source /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wspolna_korekta_patch_20261009/eden/common.sh
cd "$CODE"
"$PY" finalizuj.py \
  --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" --archive "$ARCHIVE" \
  --training "$RESULTS/training" --code-root "$CODE" --output "$RESULTS" \
  --batch-size 16
