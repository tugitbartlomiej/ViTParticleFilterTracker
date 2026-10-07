#!/bin/bash
#SBATCH --job-name=dopasuj_sily
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_maly_vit_sily_czastek/logi/%x_%j.out
# Liniowy model ruchu każdej cząstki z zapisanych sił (dopasuj_sily.py, ../PLAN.md, etap 3). Bez kopiowania danych.
#   sbatch dopasuj_sily.slurm pelny
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
TRYB=${1:?proba albo pelny}
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-07_maly_vit_sily_czastek"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
DANE="$EXP/wyniki/$TRYB/dane"
[ -f "$DANE/S_nowa.npy" ] || { echo "BRAK $DANE/S_nowa.npy"; exit 2; }
SCRATCH="/dev/shm/${USER}_dopasuj_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH/tmp"
sprzataj() { set +e; case "$SCRATCH" in /dev/shm/${USER}_dopasuj_*) rm -rf -- "$SCRATCH";; esac
  echo "moje wpisy w /dev/shm po sprzątaniu (puste = czysto):"; find /dev/shm -maxdepth 1 -user "$(id -u)" 2>/dev/null; }
trap sprzataj EXIT
# stare resztki po dawnych zadaniach na węźle hopper (zgoda użytkownika 2026-10-07: „niepotrzebne możesz usunąć”)
for STARY in /dev/shm/plarge-1817335-6Xmkq4 /dev/shm/kont245-1814592-tII0K9; do
  [ -d "$STARY" ] && [ "$(stat -c %U "$STARY")" = "$USER" ] && rm -rf -- "$STARY" && echo "usunięto $STARY"
done
trap 'echo "SYGNAŁ - przerywam"; exit 143' INT TERM
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE TMPDIR="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/xdg"
cd "$EXP/kod"
"$PY" -u -B dopasuj_sily.py --dane "$DANE"
ls -la "$EXP/wyniki/$TRYB"
echo "KONIEC $(date)"
