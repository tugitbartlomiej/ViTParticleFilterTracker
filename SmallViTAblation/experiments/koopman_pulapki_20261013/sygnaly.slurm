#!/bin/bash
#SBATCH --job-name=sygnaly
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-13_koopman_pulapki/logi/%x_%j.out
# Sygnały sieci jako wejścia (kod/sygnaly_sieci.py): powtórka uczenia od stanu, co krok z6, xin, siła własna sond.
#   sbatch sygnaly.slurm <nazwa> <bieg z pokretla/wyniki> <stan_eNN.pt> <lr> [dodatkowe argumenty sygnaly_sieci.py]
#   sbatch sygnaly.slurm e12_lr5e-4 dlugi_lr_5e-4_rp stan_e12.pt 5e-4
# Sondy (12 zdjęć) i manifest zawsze z biegu dlugi_lr_5e-4_rp (te same sondy we wszystkich warunkach); stan z <bieg>.
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
NAZWA=${1:?nazwa}; BIEG=${2:?bieg}; STAN=${3:?stan}; LR=${4:?lr}; shift 4
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-13_koopman_pulapki"
POK="$B/smallvit_runs/2026-10-10_szklo_pokretla"
E35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
OUT="$EXP/wyniki/$NAZWA"
[ -e "$OUT/meta.json" ] && { echo "ISTNIEJE $OUT"; exit 4; }
SCRATCH="/dev/shm/${USER}_fdt_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$SCRATCH/tmp" "$EXP/logi" "$OUT"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MALY_VIT_WEJSCIA="$E35/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache"
export PYTHONPATH="$E35/deps:$E35/kod:$POK/kod"
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/tmp" TORCH_HOME="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/tmp"
echo "FDT $NAZWA: bieg $BIEG, $STAN, lr $LR, argumenty $*, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
cd "$EXP/kod"
"$PY" -u -B sygnaly_sieci.py --bieg "$POK/wyniki/dlugi_lr_5e-4_rp" --stan "$POK/wyniki/$BIEG/pelny/$STAN" --lr "$LR" --out "$OUT" "$@"
echo "KONIEC $NAZWA $(date)"
