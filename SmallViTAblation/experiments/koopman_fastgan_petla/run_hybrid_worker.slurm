#!/bin/bash
#SBATCH --job-name=koopman_hybrid
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --export=NIL
#SBATCH --no-requeue
set -euo pipefail

# Argument stanowi wyłącznie bezpieczny identyfikator; ścieżki są stałe.
REQUEST_ID=${1:-}
[[ "$REQUEST_ID" =~ ^[0-9a-f]{32}$ ]] || exit 90
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || exit 91
EDEN_HOME=/home2/faculty/bpiotrowski
ROOT="$EDEN_HOME/smallvit_runs/2026-09-24_koopman_fastgan_petla"
PY="$EDEN_HOME/venvs/rfdetr_nano_clean_1_10_1/bin/python"
export PATH=/usr/local/bin:/usr/bin:/bin
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SCRATCH=$(mktemp -d "/tmp/koopman-hybrid-${SLURM_JOB_ID}-${REQUEST_ID}-XXXXXX")
cleanup() {
  RESULT=$?
  trap - EXIT INT TERM
  if [[ "$RESULT" -ne 0 && -f "$SCRATCH/code/hybryda_worker.py" ]]; then
    "$PY" -B "$SCRATCH/code/hybryda_worker.py" fail --request-id "$REQUEST_ID" \
      --message "Worker SLURM zakończony kodem $RESULT" || true
  fi
  # Nigdy nie usuwamy shared ROOT ani katalogu innego zadania.
  case "$SCRATCH" in
    /tmp/koopman-hybrid-"${SLURM_JOB_ID}"-"${REQUEST_ID}"-*)
      if [[ -d "$SCRATCH" && ! -L "$SCRATCH" && "$(realpath -e -- "$SCRATCH")" == "$SCRATCH" ]]; then
        rm -rf -- "$SCRATCH"
      fi;;
    *) RESULT=92;;
  esac
  exit "$RESULT"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export MPLCONFIGDIR="$SCRATCH/mpl"
mkdir "$SCRATCH/code"
cp -a "$ROOT/code/." "$SCRATCH/code/"
export PYTHONPATH="$SCRATCH/code:$ROOT/deps"
"$PY" -u -B "$SCRATCH/code/hybryda_worker.py" run --request-id "$REQUEST_ID" --scratch "$SCRATCH"
