#!/bin/bash
#SBATCH --job-name=k1000_temporal_900k
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/logs/population_%j.out
set -euo pipefail
ROOT=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009
PRODUCER=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=$ROOT/.mpl-cache
cd "$ROOT/code_frozen_production"
"$ROOT/.venv/bin/python" -m pytest -q -p no:cacheprovider test_pipeline.py | tee "$ROOT/results/TESTS_POPULATION.txt"
"$ROOT/.venv/bin/python" -u analiza_populacji.py --index "$PRODUCER/results/merged/REPLAY_INDEX.jsonl" --panel-contract "$PRODUCER/provenance/PANEL_64.json" --analytical-key analysis_positions_pca32 --analytical-provenance "$PRODUCER/results/merged/ANALYTICAL_PCA32_MANIFEST.json" --output "$ROOT/results/population900k"
