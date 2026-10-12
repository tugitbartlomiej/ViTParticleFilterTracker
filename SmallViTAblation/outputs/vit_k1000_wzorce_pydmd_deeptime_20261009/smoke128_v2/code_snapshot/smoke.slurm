#!/bin/bash
#SBATCH --job-name=k1000_temporal_smoke
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:45:00
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/logs/smoke_%j.out
set -euo pipefail
ROOT=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=$ROOT/.mpl-cache
cd "$ROOT/code"
"$ROOT/.venv/bin/python" -m pytest -q -p no:cacheprovider test_pipeline.py | tee "$ROOT/results/TESTS_SMOKE_V2.txt"
"$ROOT/.venv/bin/python" -m pip freeze > "$ROOT/results/ENVIRONMENT_FREEZE.txt"
"$ROOT/.venv/bin/python" -u analiza_temporalna.py --old-root /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008 --output "$ROOT/results/smoke128_v2" --rank 16 --max-windows 80
