#!/bin/bash
#SBATCH --job-name=mosty_ablacje
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-17
#SBATCH --signal=B:TERM@300
# Część 2: trening jednego wariantu ablacji na jednym seedzie (6 wariantów × 3 seedy = 18 zadań) i pomiar mostów na jego 12 checkpointach.
#   sbatch Kod_eksperymentu/ablacje.slurm                   # tablica 18 biegów
#   sbatch --array=1 Kod_eksperymentu/ablacje.slurm smoke   # próba: 64 klatki, 4 kroki, pomiar na 2 checkpointach i 6 klatkach
#   sbatch --array=K Kod_eksperymentu/ablacje.slurm wznow   # wznowienie zadania K z restart.pt (wagi, AdamW, kolejka, RNG)
set -euo pipefail
WAR=(base parallel frozen_mlp frozen_attn no_wd shuffled_labels)
SEEDS=(3047 4047 5047)
W=${WAR[$((SLURM_ARRAY_TASK_ID / 3))]}; S=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
KAMPANIA=/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/kompensacja_funkcjonalna_dgx2_20261017
ZIPY=/mnt/evafs/faculty/home/bpiotrowski/smallvit_data/cataract_lmm/3_Object_Tracking/annotations
PY=/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/przeciwstawne_eden_20261017/environment/venv/bin/python
TRYB=${1:-pelny}
OUT=Wyniki/Dane/ablacje/${W}_s$S; [ "$TRYB" = smoke ] && OUT=Wyniki/Dane/ablacje_smoke/${W}_s$S
mkdir -p Wyniki/Dane/logi "$OUT"
exec > "Wyniki/Dane/logi/ablacje_${W}_s${S}_${SLURM_JOB_ID}.log" 2>&1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "wariant $W seed $S tryb $TRYB, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
ARG=""; [ "$TRYB" = smoke ] && ARG="--smoke"; [ "$TRYB" = wznow ] && ARG="--resume"
cd Kod_eksperymentu
"$PY" -B trening_ablacje.py --manifest "$KAMPANIA/input/training_manifest.jsonl" --archives "$ZIPY" --out "../$OUT" --seed $S --wariant $W $ARG
cd ..
[ -f "$OUT/COMPLETE.json" ] || { echo "trening bez COMPLETE.json"; exit 3; }
EXTRA=""; [ "$TRYB" = smoke ] && EXTRA="--kroki 0 4 --proba 6"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" -B Kod_eksperymentu/pomiar_mosty_eden.py --wagi "$OUT/weights" --dane "$KAMPANIA/input" --out "$OUT/pomiar" --wariant $W \
      --rola evaluation --cpus 16 $EXTRA
sha256sum "$OUT"/weights/k*.pt > "$OUT/WAGI_SHA256.txt"
echo "KONIEC $W s$S $(date)"
