#!/bin/bash
#SBATCH --job-name=k1000_dmdc900k_supp
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/logs/population_dmdc_supp_%j.out
set -euo pipefail
ROOT=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009
PRODUCER=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=$ROOT/.mpl-cache
cd "$ROOT/code_frozen_population_dmdc_supplement"
"$ROOT/.venv/bin/python" -m pytest -q -p no:cacheprovider test_supplement.py | tee "$ROOT/results/TESTS_POPULATION_DMDC_SUPPLEMENT_JOB.txt"
"$ROOT/.venv/bin/python" -u population_dmdc_supplement.py --producer-root "$PRODUCER" --output "$ROOT/results/population900k_dmdc_supplement"
