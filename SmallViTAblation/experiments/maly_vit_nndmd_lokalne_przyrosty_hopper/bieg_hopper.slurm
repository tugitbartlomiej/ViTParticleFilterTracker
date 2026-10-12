#!/bin/bash
#SBATCH --job-name=nndmdlate11
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper-2
#SBATCH --nodelist=hopper-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4,gpu:h200_4g.71gb:4,gpu:h200_3g.71gb:3
#SBATCH --cpus-per-task=44
#SBATCH --mem=384G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_lokalne_przyrosty_hopper/slurm-%j.out
#SBATCH --error=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_lokalne_przyrosty_hopper/slurm-%j.err
set -euo pipefail
RUN=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_lokalne_przyrosty_hopper
PY=/home2/faculty/bpiotrowski/venvs/nndmd_expanded_20261002/bin/python
export EDEN_BASE=/home2/faculty/bpiotrowski
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$RUN:$EDEN_BASE/smallvit_runs/2026-09-28_maly_vit_od_zera/deps"
cd "$RUN"
sha256sum -c FINAL_SOURCE.sha256
export LD_PRELOAD="$RUN/runtime_nccl232/nvidia/nccl/lib/libnccl.so.2"
export NCCL_MNNVL_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export INFRA_OUTPUT="$RUN/execution_${SLURM_JOB_ID}"
mkdir "$INFRA_OUTPUT"
mkdir "$INFRA_OUTPUT/EXECUTED_SOURCE"
while read -r digest filename; do cp -- "$filename" "$INFRA_OUTPUT/EXECUTED_SOURCE/"; done < FINAL_SOURCE.sha256
cp FINAL_SOURCE.sha256 "$INFRA_OUTPUT/EXECUTED_SOURCE/"
"$PY" prepare_uuid_map.py
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
"$PY" przygotuj_eden.py
export EXPANDED_ENTRYPOINT=reader_preflight.py
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=11 uuid_entry.py
export EXPANDED_ENTRYPOINT=train.py
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=11 uuid_entry.py --config CONFIG.json
