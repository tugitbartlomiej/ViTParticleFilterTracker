#!/usr/bin/env bash
#SBATCH --job-name=k1k1500_eval
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008/logs/%x_%j.out

set -euo pipefail
HERE=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008/eden
source "$HERE/common.sh"
cd "$CODE"
"$PY" pakuj_modele.py \
  --checkpoint "$CHECKPOINT" \
  --regime-model "$RESULTS/regimes/REGIME_MODEL_ALL.npz" \
  --output "$RESULTS/packages" --source-code "$CODE"
"$PY" ewaluuj_korekty.py \
  --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --archive "$ARCHIVE" \
  --production "$RESULTS/production" --regimes "$RESULTS/regimes" \
  --packages "$RESULTS/packages" --output "$RESULTS/evaluation"
