#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
code_dir="$1"
shift
python_bin=/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/przeciwstawne_eden_20261017/environment/venv/bin/python
exec "$python_bin" "$code_dir/train_continue.py" --config "$code_dir/config.json" "$@"
