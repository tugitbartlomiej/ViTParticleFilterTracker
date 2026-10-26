#!/bin/bash
#SBATCH --job-name=mosty_pomiar
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-23
# Część 1: pomiar mostów A, B, C i IoU/AP na 12 checkpointach jednego z 24 seedów kampanii kompensacji (cwd = folder eksperymentu).
#   sbatch Kod_eksperymentu/pomiar.slurm            # tablica 24 seedów
#   sbatch --array=0 Kod_eksperymentu/pomiar.slurm proba   # próba: 1 seed, 2 checkpointy, 6 klatek
set -euo pipefail
SEEDS=(3047 4047 5047 6047 7047 8047 9047 10047 11047 12047 13047 14047 15047 16047 17047 18047 19047 20047 21047 22047 23047 24047 25047 26047)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
KAMPANIA=/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/kompensacja_funkcjonalna_dgx2_20261017
PY=/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/przeciwstawne_eden_20261017/environment/venv/bin/python
OUT=Wyniki/Dane/pomiar/s$S
mkdir -p Wyniki/Dane/logi "$OUT"
exec > "Wyniki/Dane/logi/pomiar_s${S}_${SLURM_JOB_ID}.log" 2>&1
# BLAS jednowątkowy: wartości własne liczy pula 16 procesów (16 × 16 wątków dławiło węzeł: 0,8 s na macierz zamiast 5 ms)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
echo "seed $S, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
EXTRA=""
[ "${1:-}" = proba ] && EXTRA="--kroki 0 65640 --proba 6"
for k in 00000 00300 01000 03000 06000 10000 16000 24000 33000 44000 55000 65640; do
  [ -f "$KAMPANIA/replications/s$S/weights/k$k.pt" ] || { echo "BRAK checkpointu k$k seedu $S"; exit 2; }
done
sha256sum "$KAMPANIA/replications/s$S/weights/"k*.pt > "$OUT/WAGI_SHA256.txt"
"$PY" -B Kod_eksperymentu/pomiar_mosty_eden.py --wagi "$KAMPANIA/replications/s$S/weights" --dane "$KAMPANIA/input" \
      --out "$OUT" --wariant base --rola evaluation --cpus 16 $EXTRA
echo "{\"seed\": $S, \"status\": \"complete\", \"job\": \"$SLURM_JOB_ID\", \"koniec\": \"$(date -Is)\"}" > "$OUT/COMPLETE.json"
echo "KONIEC seed $S $(date)"
