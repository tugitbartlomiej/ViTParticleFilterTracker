#!/bin/bash
#SBATCH --job-name=pfastgan
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
# Krok FastGAN petli Large (PLAN.md): $1 = sciezka zadanie.json (w katalogu MY/fastgan/...).
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
Z=${1:?zadanie.json}
BASE=/home2/faculty/bpiotrowski
OLD="$BASE/smallvit_runs/2026-09-24_koopman_fastgan_petla"
MY="$BASE/smallvit_runs/2026-09-27_petla_large"
PY="$BASE/venvs/rfdetr_nano_clean_1_10_1/bin/python"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 DETERMINIZM_OSTRZEZENIE=1
[ -f "$Z" ] || { echo "BRAK $Z"; exit 2; }
SCRATCH=$(mktemp -d "/dev/shm/pfastgan-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/pfastgan-*) rm -rf -- "$SCRATCH";; *) exit 91;; esac; }
trap cleanup EXIT INT TERM
export MPLCONFIGDIR="$SCRATCH/mpl" TORCH_HOME="$SCRATCH/torch"
mkdir -p "$SCRATCH/code"
cp "$OLD/code/"*.py "$SCRATCH/code/"
cp "$MY/code/czastki.py" "$MY/code/generowanie.py" "$MY/code/fastgan_large.py" "$MY/code/deterministyka.py" "$SCRATCH/code/"
export PYTHONPATH="$SCRATCH/code:$OLD/deps:$BASE/smallvit_runs/2026-09-23_fastgan/deps"
echo "fastgan $Z node $(hostname) start $(date)"
"$PY" -u -B "$SCRATCH/code/fastgan_large.py" --zadanie "$Z"
echo "KONIEC fastgan $(date)"
