#!/bin/bash
#SBATCH --job-name=szklo_bd
#SBATCH --account=transformers_vsc
#SBATCH --partition=short,hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/logi/%x_%j.out
# Szkło koloidalne (dynamika browna miękkich kul) na jednej karcie, 8 gęstości naraz.
#   sbatch szklo_eden.slurm stala      # stała temperatura T = 3e-4, skan φ 0,56–0,66, 10^7 kroków
#   sbatch szklo_eden.slurm stygnace   # stygnięcie własne T(t) jak energia drgań tokenów ViT
#   sbatch szklo_eden.slurm proba      # próba kodu (3 gęstości, 2·10^5 kroków)
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
TRYB=${1:?stala, stygnace albo proba}
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_koloidalne"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
case "$TRYB" in
  stala)    ARG="--tryb=stala --out=$EXP/wyniki/stala";;
  stygnace) ARG="--tryb=stygnace --out=$EXP/wyniki/stygnace";;
  proba)    ARG="--tryb=stala --out=$EXP/wyniki/proba --phi=0.58,0.62,0.66 --n_komp=20000 --n_rown=30000 --n_rec=200000";;
  szybko_stala)    ARG="--tryb=stala --out=$EXP/wyniki/szybko_stala --N=2048 --phi=0.58,0.60,0.62,0.64 --n_komp=150000 --n_rown=300000 --n_rec=700000 --co=100 --co_os=10";;
  szybko_stygnace) ARG="--tryb=stygnace --out=$EXP/wyniki/szybko_stygnace --N=2048 --phi=0.58,0.60,0.62,0.64 --n_komp=150000 --n_rown=300000 --n_rec=700000 --co=100 --co_os=10 --tzal=2000 --t1=140000";;
  *) echo "zły tryb $TRYB"; exit 2;; esac
[ -f "$EXP/kod/model_szkla_eden.py" ] && [ -f "$EXP/kod/odciski.py" ] || { echo "BRAK kodu w $EXP/kod"; exit 2; }
SCRATCH="/dev/shm/${USER}_szklo_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH" "$EXP/wyniki" "$EXP/logi"
trap 'rm -rf -- "$SCRATCH"; du -sh "$EXP/wyniki"/* 2>/dev/null; lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1' EXIT
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TMPDIR="$SCRATCH" MPLCONFIGDIR="$SCRATCH" TORCH_HOME="$SCRATCH" XDG_CACHE_HOME="$SCRATCH"
echo "tryb $TRYB, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
"$PY" "$EXP/kod/model_szkla_eden.py" $ARG
echo "koniec $(date)"
