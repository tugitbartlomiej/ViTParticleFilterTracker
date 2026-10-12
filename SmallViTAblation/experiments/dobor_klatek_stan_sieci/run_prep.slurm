#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=stan_prep
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
: "${ROOT:?}" "${PROJECT_DIR:?}" "${DATA_ARCHIVE:?}" "${PY:?}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONPATH
WORK="/dev/shm/${USER}_${SLURM_JOB_ID}_stan_prep"
cleanup() {
  case "$WORK" in /dev/shm/*_stan_prep) rm -rf -- "$WORK" ;; *) exit 2 ;; esac
}
trap cleanup EXIT
trap 'exit 130' INT TERM
test ! -e "$WORK"
mkdir -p "$WORK/data"
tar -xf "$DATA_ARCHIVE" -C "$WORK/data"
cd "$ROOT"
test -s "$ROOT/cache/L0_tokens/meta.json"
"$PY" -B -m SmallViTAblation.experiments.dobor_klatek_stan_sieci.token_wybierz \
  --pool "$WORK/data/pool_manifest.jsonl" --cache "$ROOT/cache/L0_tokens" \
  --initial-selection "$WORK/data/variants/tracking_uniform_time_b500_s42.json" \
  --out "$ROOT/selection" --device cuda
"$PY" -B -m SmallViTAblation.experiments.dobor_klatek_stan_sieci.uruchom \
  --project-dir "$PROJECT_DIR" --variants-dir "$ROOT/selection/variants" \
  --output-root "$ROOT/pilot" --base-protocol "$PROJECT_DIR/configs/protocol_tracking_calibration.yaml" \
  --seed 42 --epochs 10
"$PY" -B "$PROJECT_DIR/eden/queue.py" init "$ROOT/pilot/queue" --queue-file "$ROOT/pilot/queue.jsonl"
