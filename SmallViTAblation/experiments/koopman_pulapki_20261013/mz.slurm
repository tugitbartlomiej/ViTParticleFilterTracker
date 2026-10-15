#!/bin/bash
#SBATCH --job-name=mz_studnie
#SBATCH --account=transformers_vsc
#SBATCH --partition=short,hopper-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-13_koopman_pulapki/logi/%x_%j.out
# Mori–Zwanzig, studnie i wiry na pułapkach (kod/mz_studnie_pulapki.py); forward jak koopman.slurm.
#   sbatch mz.slurm <nazwa> [dodatkowe argumenty, np. --proba]
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
NAZWA=${1:?nazwa}; shift
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-13_koopman_pulapki"
E35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
RP="$B/smallvit_runs/2026-10-10_szklo_koloidalne/wyniki/ruch60_rp128_50tr_50val"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
ZIPY="$B/smallvit_data/cataract_lmm/3_Object_Tracking/annotations"
OUT="$EXP/wyniki/$NAZWA"
[ -e "$OUT/MZ_STUDNIE.json" ] && { echo "ISTNIEJE $OUT - nie nadpisuję"; exit 4; }
SCRATCH="/dev/shm/${USER}_koopman_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$SCRATCH/tmp" "$EXP/logi" "$OUT"
trap 'rm -rf -- "$SCRATCH"' EXIT
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$E35/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache"
export PYTHONPATH="$E35/deps:$E35/kod"
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/tmp" TORCH_HOME="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/tmp"
echo "koopman $NAZWA $*, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
cd "$EXP/kod"
"$PY" -u -B mz_studnie_pulapki.py --log "$E35/wyniki/pelny/dane/LOG.json" --wagi "$E35/wyniki/pelny/wagi" --manifest "$E35/wejscia/MANIFEST_35000.jsonl" --zipy "$ZIPY" \
      --rp128 "$RP" --out "$OUT" "$@"
echo "KONIEC $NAZWA $(date)"
