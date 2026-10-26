#!/bin/bash
#SBATCH --job-name=lodyga_abl
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --signal=B:TERM@300
# Kontrola „oba zamrożone”: trening wariantu frozen_both (uczy się łodyga, norma końcowa i głowa) na seedzie 3047/4047/5047
# i pomiar mostów na 12 checkpointach (jak ablacje.slurm rodzica mosty_fizyki_24seedy_ablacje_20261026__13-04-31).
#   sbatch Kod_eksperymentu/ablacje_lodyga.slurm           # 3 biegi
#   sbatch --array=0 Kod_eksperymentu/ablacje_lodyga.slurm smoke
set -euo pipefail
SEEDS=(3047 4047 5047)
W=frozen_both; S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
KAMPANIA=/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/kompensacja_funkcjonalna_dgx2_20261017
ZIPY=/mnt/evafs/faculty/home/bpiotrowski/smallvit_data/cataract_lmm/3_Object_Tracking/annotations
PY=/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/przeciwstawne_eden_20261017/environment/venv/bin/python
TRYB=${1:-pelny}
OUT=Wyniki/Dane/ablacje/${W}_s$S; [ "$TRYB" = smoke ] && OUT=Wyniki/Dane/ablacje_smoke/${W}_s$S
mkdir -p Wyniki/Dane/logi "$OUT"
exec > "Wyniki/Dane/logi/ablacje_${W}_s${S}_${SLURM_JOB_ID}.log" 2>&1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "wariant $W seed $S tryb $TRYB, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
ARG=""; [ "$TRYB" = smoke ] && ARG="--smoke"; [ "$TRYB" = wznow ] && ARG="--resume"
cd Kod_eksperymentu
"$PY" -B trening_ablacje.py --manifest "$KAMPANIA/input/training_manifest.jsonl" --archives "$ZIPY" --out "../$OUT" --seed $S --wariant $W $ARG
cd ..
[ -f "$OUT/COMPLETE.json" ] || { echo "trening bez COMPLETE.json"; exit 3; }
EXTRA=""; [ "$TRYB" = smoke ] && EXTRA="--kroki 0 4 --proba 6"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" -B Kod_eksperymentu/pomiar_mosty_eden.py --wagi "$OUT/weights" --dane "$KAMPANIA/input" --out "$OUT/pomiar" --wariant $W \
      --rola evaluation --cpus 3 $EXTRA
sha256sum "$OUT"/weights/k*.pt > "$OUT/WAGI_SHA256.txt"
echo "KONIEC $W s$S $(date)"
