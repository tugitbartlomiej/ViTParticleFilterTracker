#!/bin/bash
#SBATCH --job-name=koopman_fastgan_v1
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
export HOME=/home2/faculty/bpiotrowski
export PATH=/usr/local/bin:/usr/bin:/bin
ROOT="$HOME/smallvit_runs/2026-09-24_koopman_fastgan_petla"
PY="$HOME/venvs/rfdetr_nano_clean_1_10_1/bin/python"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SCRATCH=$(mktemp -d "/tmp/koopman-fastgan-${SLURM_JOB_ID}-XXXXXX")
cleanup() {
  case "$SCRATCH" in /tmp/koopman-fastgan-"${SLURM_JOB_ID}"-*) rm -rf -- "$SCRATCH";; *) exit 91;; esac
}
trap cleanup EXIT INT TERM
export MPLCONFIGDIR="$SCRATCH/mpl"
cp -a "$ROOT/code" "$SCRATCH/code"
tar -cf "$ROOT/source_${SLURM_JOB_ID}.tar" -C "$SCRATCH/code" .
export PYTHONPATH="$SCRATCH/code:$ROOT/deps:$HOME/smallvit_runs/2026-09-23_fastgan/deps"
mkdir -p "$SCRATCH/data" "$SCRATCH/fastgan"
tar -xf "$HOME/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar" -C "$SCRATCH/data"
tar -xf "$HOME/smallvit_runs/2026-09-23_fastgan/eden_inputs.tar" -C "$SCRATCH/fastgan"
MODE=${1:-smoke}
OUT="$ROOT/results/${MODE}_${SLURM_JOB_ID}"
EXTRA=()
case "$MODE" in
 smoke) SCRIPT=smoke.py;;
 check) SCRIPT=smoke.py; [[ "${2:-}" =~ ^[0-9]+$ ]] || exit 92; EXTRA=(--training-from "$ROOT/results/smoke_$2");;
 run) SCRIPT=petla.py;;
 *) exit 93;;
esac
echo "MODE=$MODE OUT=$OUT GPU=$CUDA_VISIBLE_DEVICES"
"$PY" -u -B "$SCRATCH/code/$SCRIPT" \
  --config "$SCRATCH/code/config.json" \
  --pool "$SCRATCH/data/pool_manifest.jsonl" --data-root "$SCRATCH/data" \
  --initial-selection "$ROOT/initial_selection.json" \
  --initial-weights "$ROOT/weights/rf-detr-nano.pth" \
  --crop-manifest "$SCRATCH/fastgan/dane/pilot_1000/MANIFEST.json" \
  --crop-root "$SCRATCH/fastgan/dane/pilot_1000" \
  --generator "$HOME/smallvit_runs/2026-09-23_fastgan/results_50k/checkpoint_050000.pth" \
  --vendor "$SCRATCH/fastgan/vendor/deepfake_gi_fastGAN" \
  --hub "$HOME/.cache/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8" \
  --dino-weights "$HOME/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth" \
  --out "$OUT" --device cuda "${EXTRA[@]}"
