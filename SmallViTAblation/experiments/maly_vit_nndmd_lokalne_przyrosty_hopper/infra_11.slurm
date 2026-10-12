#!/bin/bash
#SBATCH --job-name=nndmd11infra
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper-2
#SBATCH --nodelist=hopper-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4,gpu:h200_4g.71gb:4,gpu:h200_3g.71gb:3
#SBATCH --cpus-per-task=22
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_lokalne_przyrosty_hopper/infra11-%j.out
#SBATCH --error=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_lokalne_przyrosty_hopper/infra11-%j.err
set -euo pipefail
RUN=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_lokalne_przyrosty_hopper
PY=/home2/faculty/bpiotrowski/venvs/nndmd_expanded_20261002/bin/python
cd "$RUN"
sha256sum -c INFRA11.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=INFO
export INFRA_OUTPUT="$RUN/infra11_${SLURM_JOB_ID}"
mkdir "$INFRA_OUTPUT"
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,uuid --format=csv
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=11 infra_11.py
