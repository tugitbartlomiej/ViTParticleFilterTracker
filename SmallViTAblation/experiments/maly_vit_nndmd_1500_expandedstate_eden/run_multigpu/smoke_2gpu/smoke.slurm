#!/bin/bash
#SBATCH --job-name=nndmd2equiv
#SBATCH --account=transformers_vsc
#SBATCH --partition=long
#SBATCH --nodelist=dgx-4
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_2gpu_model_smoke/smoke-%j.out
#SBATCH --error=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_2gpu_model_smoke/smoke-%j.err
set -euo pipefail
RUN=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_2gpu_model_smoke
PY=/home2/faculty/bpiotrowski/venvs/nndmd_expanded_20261002/bin/python
cd "$RUN"
sha256sum -c SNAPSHOT.sha256
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 gpu_smoke.py --tokens 4 --output "GPU_SMOKE_SMALL_${SLURM_JOB_ID}"
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 gpu_smoke.py --tokens 600 --output "GPU_SMOKE_ALL600_${SLURM_JOB_ID}"
