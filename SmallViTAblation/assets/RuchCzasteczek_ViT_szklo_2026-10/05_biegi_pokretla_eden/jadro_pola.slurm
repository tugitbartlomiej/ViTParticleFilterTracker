#!/bin/bash
#SBATCH --job-name=jadro
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper-2,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_pokretla/logi/%x_%j.out
# Test jądra (kod/jadro_pola.py): czy pole przesunięć tokenów przewiduje sam Jakobian sieci ze wspólnymi wagami.
#   sbatch jadro_pola.slurm <bieg> <stan_eNN.pt> [n_kier] [n_paczek] [od do]   → wyniki/<bieg>/JADRO_<stan>.json
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
BIEG=${1:?bieg}; STAN=${2:?stan_eNN.pt}; NK=${3:-100}; NP=${4:-40}
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_pokretla"
E35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
OUT="$EXP/wyniki/$BIEG"
[ -f "$OUT/pelny/$STAN" ] || { echo "BRAK $OUT/pelny/$STAN"; exit 2; }
SCRATCH="/dev/shm/${USER}_jadro_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH/tmp" "$EXP/logi"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE CUBLAS_WORKSPACE_CONFIG=:4096:8
export MALY_VIT_WEJSCIA="$E35/wejscia" MALY_VIT_DANE="$SCRATCH" MALY_VIT_CACHE="$SCRATCH"
export PYTHONPATH="$E35/deps:$E35/kod"
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/tmp"
OKNO=""; [ $# -ge 6 ] && OKNO="--okno $5 $6"
cd "$EXP/kod"
echo "jądro $BIEG $STAN: $NK kierunków, $NP paczek, $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), $(date)"
"$PY" -u -B jadro_pola.py "$OUT" "$OUT/pelny/$STAN" "$OUT/JADRO_${STAN%.pt}.json" --n_kier "$NK" --n_paczek "$NP" $OKNO
echo "KONIEC $(date)"
