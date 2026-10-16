#!/bin/bash
#SBATCH --job-name=ocena_klas
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_pokretla/logi/%x_%j.out
# Ocena biegów Imagenette ze stanów pelny/stan_eNN.pt -> wyniki/OCENA_KLASYFIKACJI.json (kod/ocena_klasyfikacji.py).
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_pokretla"
E35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
DANE="$B/smallvit_data/imagenette/imagenette2-320"
SCRATCH="/dev/shm/${USER}_ocena_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH/tmp"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$E35/wejscia" MALY_VIT_DANE="$SCRATCH" MALY_VIT_CACHE="$SCRATCH"
export PYTHONPATH="$E35/deps:$E35/kod"
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/tmp" TORCH_HOME="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/tmp"
cd "$EXP/kod"
"$PY" -u -B ocena_klasyfikacji.py "$EXP/wyniki" "$DANE" klas_lr_1.25e-4 klas_lr_5e-4 klas_lr_2e-3 klas_cnn_lr_5e-4 klas_dlugi_lr_5e-4
