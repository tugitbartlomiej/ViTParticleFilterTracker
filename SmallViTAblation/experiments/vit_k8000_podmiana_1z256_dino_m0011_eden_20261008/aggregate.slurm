#!/bin/bash
#SBATCH --job-name=vit1z256agg
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_1z256/logs/%x_%j.out
#SBATCH --error=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_1z256/logs/%x_%j.err
set -euo pipefail
ROOT=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-08_vit_k8000_podmiana_1z256
PY=/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python
"$PY" -u -B "$ROOT/code/aggregate.py" "$ROOT/results" "$ROOT/input_pack"
