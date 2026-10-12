#!/bin/bash
#SBATCH --job-name=nndmd1500exp
#SBATCH --account=transformers_vsc
#SBATCH --partition=long,hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_eden/slurm-%j.out
#SBATCH --error=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_eden/slurm-%j.err
set -euo pipefail
RUN=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-02_nndmd_1500_expandedstate_eden
PY=/home2/faculty/bpiotrowski/venvs/nndmd_expanded_20261002/bin/python
export EDEN_BASE=/home2/faculty/bpiotrowski
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="$RUN:$EDEN_BASE/smallvit_runs/2026-09-28_maly_vit_od_zera/deps"
if [[ -d /raid && $(df -Pk /raid | awk 'NR==2 {print $4}') -gt 350000000 ]]; then
  ROOT=/raid; export EXPANDED_CACHE_MODE=full101
elif [[ -d /raid && $(df -Pk /raid | awk 'NR==2 {print $4}') -gt 50000000 ]]; then
  ROOT=/raid; export EXPANDED_CACHE_MODE=rolling13
elif [[ $(df -Pk /dev/shm | awk 'NR==2 {print $4}') -gt 50000000 ]]; then
  ROOT=/dev/shm; export EXPANDED_CACHE_MODE=rolling13
else
  echo 'Insufficient verified scratch; no extraction or training' >&2; exit 2
fi
export EXPANDED_SCRATCH="$ROOT/${USER}_nndmd_expanded_${SLURM_JOB_ID}"
export EXPANDED_CACHE="$EXPANDED_SCRATCH/cache"
mkdir -p "$EXPANDED_CACHE"
cleanup() {
  case "$EXPANDED_SCRATCH" in
    /raid/"${USER}_nndmd_expanded_${SLURM_JOB_ID}"|/dev/shm/"${USER}_nndmd_expanded_${SLURM_JOB_ID}") rm -rf -- "$EXPANDED_SCRATCH" ;;
    *) echo 'Unsafe cleanup path rejected' >&2 ;;
  esac
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$RUN"
export EXPANDED_LOGPATH="$RUN/slurm-${SLURM_JOB_ID}.out"
echo "RUNNING job=$SLURM_JOB_ID host=$(hostname) cache=$EXPANDED_CACHE_MODE scratch=$EXPANDED_SCRATCH"
df -h "$ROOT"
nvidia-smi
"$PY" przygotuj_eden.py
"$PY" train.py --config CONFIG.json
