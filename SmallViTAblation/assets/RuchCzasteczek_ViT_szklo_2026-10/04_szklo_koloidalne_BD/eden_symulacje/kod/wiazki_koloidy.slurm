#!/bin/bash
#SBATCH --job-name=wiazki
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/logi/%x_%j.out
# Test „768 wiązek trzyma 600 koloidów” (kod/wiazki_koloidy.py): rozkład ruchu tokenów na obrót wiązek W fc1 i ruch kulek u.
#   sbatch wiazki_koloidy.slurm
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
NRP=${1:-50}; N768=${2:-2}; TRYB=${3:-rp128}
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_koloidalne"
EXP35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
ZIPY="$B/smallvit_data/cataract_lmm/3_Object_Tracking/annotations"
OUT="$EXP/wyniki/wiazki_koloidy"
for f in "$EXP35/kod/trening35000.py" "$EXP35/deps/timm/__init__.py" "$EXP35/wejscia/MANIFEST_35000.jsonl" "$EXP35/wyniki/pelny/wagi/k00000.pt" \
         "$ARCHIVE" "$ZIPY/TR_0001_S1_P03.zip" "$EXP/kod/wiazki_koloidy.py"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
SCRATCH="/dev/shm/${USER}_wiazki_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$SCRATCH/tmp" "$EXP/logi" "$EXP/wyniki"
trap 'rm -rf -- "$SCRATCH"; du -sh "$OUT" 2>/dev/null; lfs quota -h -p 1299 /mnt/evafs 2>/dev/null | tail -1' EXIT
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$EXP35/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache"
export PYTHONPATH="$EXP35/deps"
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/tmp" TORCH_HOME="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/tmp"
echo "wiązki $NRP+$NRP zdjęć, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
cp "$EXP/kod/wiazki_koloidy.py" "$EXP35/kod/wiazki_koloidy.py"
cd "$EXP35/kod"
EXTRA=""
"$PY" -u -B wiazki_koloidy.py --wagi "$EXP35/wyniki/pelny/wagi" --manifest "$EXP35/wejscia/MANIFEST_35000.jsonl" --zipy "$ZIPY" \
      --out "$OUT" --n "$NRP" $EXTRA
echo "KONIEC $(date)"
