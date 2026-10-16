#!/bin/bash
#SBATCH --job-name=deit_cifar
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-13_deit_cifar10/logi/%x_%j.out
# DeiT-Tiny na CIFAR-10 od zera (kod/trening_cifar_deit.py), przepis jak bieg s47 MałegoViT.
#   sbatch deit_cifar.slurm <nazwa> [argumenty trening_cifar_deit.py]
#   sbatch deit_cifar.slurm s47_obciecie --epoki 60 --obserwuj --historia_wag
#   sbatch deit_cifar.slurm s47_bez_obciecia --epoki 60 --obserwuj --historia_wag --obciecie 0
#   sbatch --time=0:20:00 deit_cifar.slurm proba --proba --obserwuj --historia_wag
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
NAZWA=${1:?nazwa}; shift
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-13_deit_cifar10"
POK="$B/smallvit_runs/2026-10-10_szklo_pokretla"
E35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
OUT="$EXP/wyniki/$NAZWA"
[ -e "$OUT/dane/metadata.json" ] && [[ " $* " != *" --wznow "* ]] && { echo "ISTNIEJE $OUT - nie nadpisuję (dodaj --wznow)"; exit 4; }
SCRATCH="/dev/shm/${USER}_deit_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH/tmp" "$EXP/logi" "$OUT"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$E35/deps:$E35/kod:$POK/kod:$EXP/kod"
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/tmp" TORCH_HOME="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/tmp"
echo "DeiT-Tiny CIFAR-10 $NAZWA: argumenty $*, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
cd "$EXP/kod"
"$PY" -u -B trening_cifar_deit.py --wyniki "$OUT" --dane "$EXP/dane/cifar10_uint8.npz" "$@"
echo "KONIEC $NAZWA $(date)"
