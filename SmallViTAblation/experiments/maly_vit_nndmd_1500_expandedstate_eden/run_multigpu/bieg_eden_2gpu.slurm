#!/bin/bash
#SBATCH --job-name=nndmd1500x2
#SBATCH --account=transformers_vsc
#SBATCH --partition=long
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_multigpu/slurm-%j.out
#SBATCH --error=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_multigpu/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-4
set -euo pipefail
RUN=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_multigpu
PY=/home2/faculty/bpiotrowski/venvs/nndmd_expanded_20261002/bin/python
export EDEN_BASE=/home2/faculty/bpiotrowski
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$RUN:$EDEN_BASE/smallvit_runs/2026-09-28_maly_vit_od_zera/deps"
source "$RUN/scratch_selector.sh"
cleanup() {
  local target="${EXPANDED_SCRATCH:-}"
  [[ -n "$target" && -d "$target" && ! -L "$target" ]] || return 0
  if [[ "$(readlink -f -- "$target")" == "$target" &&
        "$(basename -- "$target")" == "${USER}_nndmd_expanded_${SLURM_JOB_ID}" &&
        -f "$target/.owner" && "$(cat "$target/.owner")" == "$USER:$SLURM_JOB_ID" ]]; then
    rm -rf -- "$target"
  else
    echo 'Unsafe cleanup path/owner rejected' >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
select_expanded_scratch "${SLURM_TMPDIR:-}" "${TMPDIR:-}" "/raid/$USER" "/raid/users/$USER" "/raid/user/$USER" "/raid/home/$USER" /raid /dev/shm
cd "$RUN"
export EXPANDED_LOGPATH="$RUN/slurm-${SLURM_JOB_ID}.out"
echo "RUNNING job=$SLURM_JOB_ID host=$(hostname) cache=$EXPANDED_CACHE_MODE scratch=$EXPANDED_SCRATCH"
df -h "$EXPANDED_SCRATCH"
nvidia-smi
sha256sum -c EXECUTION_FREEZE.sha256
"$PY" przygotuj_eden.py
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 gpu_smoke.py --tokens 600 --output GPU_PREFLIGHT_FINAL_CORE
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 reader_preflight.py
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 train.py --config CONFIG.json
