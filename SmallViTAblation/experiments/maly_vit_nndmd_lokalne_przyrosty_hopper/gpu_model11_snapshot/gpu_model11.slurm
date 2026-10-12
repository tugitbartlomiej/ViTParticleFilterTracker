#!/bin/bash
#SBATCH --job-name=nndmd11model
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper-2
#SBATCH --nodelist=hopper-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4,gpu:h200_4g.71gb:4,gpu:h200_3g.71gb:3
#SBATCH --cpus-per-task=22
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_lokalne_przyrosty_hopper/gpumodel11-%j.out
#SBATCH --error=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_lokalne_przyrosty_hopper/gpumodel11-%j.err
set -euo pipefail
RUN=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_lokalne_przyrosty_hopper
PY=/home2/faculty/bpiotrowski/venvs/nndmd_expanded_20261002/bin/python
cd "$RUN/gpu_model11_snapshot"
sha256sum -c GPU_MODEL11.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=INFO NCCL_MNNVL_ENABLE=0
export LD_PRELOAD="$RUN/runtime_nccl232/nvidia/nccl/lib/libnccl.so.2"
export INFRA_OUTPUT="$RUN/gpumodel11_${SLURM_JOB_ID}"
mkdir "$INFRA_OUTPUT"
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,uuid --format=csv
"$PY" prepare_uuid_map.py
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=11 uuid_model_entry.py --tokens 600 --output "$INFRA_OUTPUT/model"
