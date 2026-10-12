#!/bin/bash
#SBATCH --job-name=vit128gelu
#SBATCH --account=transformers_vsc
#SBATCH --partition=short,hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
RUN=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-04_vit_optymalizacja_latentu128
export PYTHONPATH="$RUN:/home2/faculty/bpiotrowski/smallvit_runs/2026-09-28_maly_vit_od_zera/deps"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$RUN"
sha256sum -c SOURCE.sha256
sha256sum -c GELU_SOURCE.sha256
/home2/faculty/bpiotrowski/venvs/nndmd_expanded_20261002/bin/python run_gelu.py --input input --output results/gelu_r1
