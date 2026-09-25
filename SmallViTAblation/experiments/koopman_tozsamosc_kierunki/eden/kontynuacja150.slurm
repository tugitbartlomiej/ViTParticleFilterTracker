#!/bin/bash
#SBATCH --job-name=kont245b
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --array=0-2
#SBATCH --signal=B:TERM@120
# Kontynuacja 1 epoki od model_0000245.pth: 3 ramiona (bez_dodania, koopman_fastgan, losowe) x ziarno z tablicy.
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
BASE=/home2/faculty/bpiotrowski
ROOT="$BASE/smallvit_runs/2026-09-24_koopman_gesta_historia"
OLD="$BASE/smallvit_runs/2026-09-24_koopman_fastgan_petla"
MY="$BASE/smallvit_runs/2026-09-25_kontynuacja_dobor150"
PY="$BASE/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$BASE/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEEDS=(42 43 44); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
SCRATCH=$(mktemp -d "/dev/shm/kont245b-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/kont245b-*) rm -rf -- "$SCRATCH";; *) exit 91;; esac; }
trap cleanup EXIT INT TERM
export MPLCONFIGDIR="$SCRATCH/mpl"
cp -a "$ROOT/code" "$SCRATCH/code"
export PYTHONPATH="$SCRATCH/code:$OLD/deps:$BASE/smallvit_runs/2026-09-23_fastgan/deps"
mkdir -p "$SCRATCH/data" "$MY/results"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
echo "seed $SEED, node $(hostname), scratch $SCRATCH, start $(date)"
for ARM in bez_dodania koopman_fastgan losowe; do
  OUT="$MY/results/seed${SEED}_${ARM}"
  if [ -f "$OUT/REPORT.json" ]; then echo "pomijam gotowe $OUT"; continue; fi
  rm -rf "$OUT"
  "$PY" -u -B "$MY/code/kontynuacja.py" --code "$SCRATCH/code" --config "$MY/config_seed${SEED}.json" \
    --pool "$SCRATCH/data/pool_manifest.jsonl" --data-root "$SCRATCH/data" --selection "$MY/selekcja_${ARM}.json" \
    --weights "$ROOT/results/run_1810662/pretrain/model_0000245.pth" --probe "$ROOT/probe/manifest.json" \
    --out "$OUT" --ramie "$ARM"
  echo "gotowe $ARM seed $SEED $(date)"
done
echo "KONIEC seed $SEED $(date)"
