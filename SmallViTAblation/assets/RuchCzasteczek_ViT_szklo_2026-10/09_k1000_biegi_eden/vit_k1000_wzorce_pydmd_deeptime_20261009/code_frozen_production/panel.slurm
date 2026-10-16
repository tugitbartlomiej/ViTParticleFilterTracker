#!/bin/bash
#SBATCH --job-name=k1000_temporal_panel
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/logs/panel_%j.out
set -euo pipefail
ROOT=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=$ROOT/.mpl-cache
cd "$ROOT/code_frozen_production"
PANEL_MANIFEST="${1:?Pass verified producer manifest path as script argument 1}"
"$ROOT/.venv/bin/python" -m pytest -q -p no:cacheprovider test_pipeline.py | tee "$ROOT/results/TESTS_PANEL.txt"
"$ROOT/.venv/bin/python" -m pip freeze > "$ROOT/results/ENVIRONMENT_FREEZE_PANEL.txt"
"$ROOT/.venv/bin/python" -u analiza_temporalna.py --panel-manifest "$PANEL_MANIFEST" --output "$ROOT/results/panel64" --rank 16 --max-windows 80
