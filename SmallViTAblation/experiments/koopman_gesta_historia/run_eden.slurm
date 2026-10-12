#!/bin/bash
#SBATCH --job-name=koopman_dense8
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
BASE=/home2/faculty/bpiotrowski
ROOT="$BASE/smallvit_runs/2026-09-24_koopman_gesta_historia"
OLD="$BASE/smallvit_runs/2026-09-24_koopman_fastgan_petla"
PY="$BASE/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$BASE/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SCRATCH=$(mktemp -d "/tmp/koopman-dense-${SLURM_JOB_ID}-XXXXXX")
cleanup() {
 case "$SCRATCH" in /tmp/koopman-dense-"${SLURM_JOB_ID}"-*) rm -rf -- "$SCRATCH";; *) exit 91;; esac
}
trap cleanup EXIT
export MPLCONFIGDIR="$SCRATCH/mpl"
cp -a "$ROOT/code" "$SCRATCH/code"
export PYTHONPATH="$SCRATCH/code:$OLD/deps:$BASE/smallvit_runs/2026-09-23_fastgan/deps"
mkdir -p "$SCRATCH/data"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
"$PY" -u -B "$SCRATCH/code/pretrain.py" \
 --config "$SCRATCH/code/config.json" --pool "$SCRATCH/data/pool_manifest.jsonl" \
 --data-root "$SCRATCH/data" --selection "$OLD/initial_selection.json" \
 --weights "$OLD/weights/rf-detr-nano.pth" --probe-source "$ROOT/probe" \
 --archive "$ARCHIVE" --out "$ROOT/results/run_${SLURM_JOB_ID}"
