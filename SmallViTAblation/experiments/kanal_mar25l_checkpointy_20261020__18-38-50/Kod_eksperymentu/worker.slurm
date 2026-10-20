#!/bin/bash
# Worker GPU: smoke (seed 3047, VAL, 3 kroki) albo jeden seed z tablicy (indeks SLURM_ARRAY_TASK_ID).
set -euo pipefail
exp="$1"
python="$2"
mode="$3"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$exp/Kod_eksperymentu"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if [[ "$mode" == smoke ]]; then
  "$python" -B run_kanal.py --smoke --seed 3047
else
  "$python" -B run_kanal.py
fi
