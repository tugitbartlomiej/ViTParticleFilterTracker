#!/bin/bash
#SBATCH --job-name=nndmd2infra
#SBATCH --account=transformers_vsc
#SBATCH --partition=long,hopper
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_eden/infra-%j.out
#SBATCH --error=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_eden/infra-%j.err
set -euo pipefail
OLD=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_eden
PY=/home2/faculty/bpiotrowski/venvs/nndmd_expanded_20261002/bin/python
export FROZEN_OLD_RUN="$OLD" FROZEN_EXECUTED_SOURCE="$OLD/EXECUTED_SOURCE"
export INFRA_RUN="$OLD/infra_job_${SLURM_JOB_ID}"
export PYTHONPATH="$FROZEN_EXECUTED_SOURCE:/home2/faculty/bpiotrowski/smallvit_runs/2026-09-28_maly_vit_od_zera/deps"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir "$INFRA_RUN"
nvidia-smi
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 "$OLD/gpu_infra_preflight.py"
