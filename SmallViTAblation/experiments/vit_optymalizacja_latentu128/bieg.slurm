#!/bin/bash
#SBATCH --job-name=vitlatent128
#SBATCH --account=transformers_vsc
#SBATCH --partition=short,hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
RUN=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-04_vit_optymalizacja_latentu128
PY=/home2/faculty/bpiotrowski/venvs/nndmd_expanded_20261002/bin/python
export PYTHONPATH="$RUN:/home2/faculty/bpiotrowski/smallvit_runs/2026-09-28_maly_vit_od_zera/deps"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$RUN"
sha256sum -c SOURCE.sha256
if [[ "${1:-}" == smoke ]]; then
  "$PY" test_core.py
  "$PY" run.py --input input --output "results/smoke_${SLURM_JOB_ID}" --smoke
else
  radii=(0.25 0.5 1.0)
  "$PY" run.py --input input --output "results/r${radii[$SLURM_ARRAY_TASK_ID]}" --radius "${radii[$SLURM_ARRAY_TASK_ID]}"
fi
