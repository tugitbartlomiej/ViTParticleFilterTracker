#!/bin/bash
#SBATCH --job-name=znak_normy
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-4
# Znak kanału normy: czy MLP przy pchnięciu tokenu w stronę odczytu zwiększa czy zmniejsza normę tokenu i czy to stabilizuje czy utrwala skutek (cwd = folder eksperymentu):
#   0–2: detekcja, seedy 3047/4047/5047 kampanii kompensacji (k0, 3000, 65640); 3: MałyViT CIFAR s47; 4: DeiT CIFAR s47 (k0, 3000, 93780).
#   sbatch --nodelist=dgx-X Kod_eksperymentu/znak_normy.slurm
#   sbatch --nodelist=dgx-X --array=0,4 Kod_eksperymentu/znak_normy.slurm proba   # 2 klatki/obrazy, 2 checkpointy
# Kod importuje (importlib) odbiorca_eden.py rodzica (a przez niego balanced_eden.py i wzmocnienie_eden.py) z sąsiednich folderów experiments/.
set -euo pipefail
UKL=(detekcja detekcja detekcja cifar_maly cifar_deit)
KAMPANIA=/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/kompensacja_funkcjonalna_dgx2_20261017
CIFAR=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-13_deit_cifar10
PY=/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/przeciwstawne_eden_20261017/environment/venv/bin/python
U=${UKL[$SLURM_ARRAY_TASK_ID]}
case $SLURM_ARRAY_TASK_ID in
  0) NAZWA=detekcja_s3047; WAGI=$KAMPANIA/replications/s3047/weights; DANE=$KAMPANIA/input; KROKI="0 3000 65640";;
  1) NAZWA=detekcja_s4047; WAGI=$KAMPANIA/replications/s4047/weights; DANE=$KAMPANIA/input; KROKI="0 3000 65640";;
  2) NAZWA=detekcja_s5047; WAGI=$KAMPANIA/replications/s5047/weights; DANE=$KAMPANIA/input; KROKI="0 3000 65640";;
  3) NAZWA=cifar_maly_s47; WAGI=$CIFAR/wyniki/maly_s47_obciecie/wagi; DANE=$CIFAR/dane/cifar10_uint8.npz; KROKI="0 3000 93780";;
  4) NAZWA=cifar_deit_s47; WAGI=$CIFAR/wyniki/s47_obciecie/wagi; DANE=$CIFAR/dane/cifar10_uint8.npz; KROKI="0 3000 93780";;
esac
OUT=Wyniki/Dane/pomiar/$NAZWA; [ "${1:-}" = proba ] && OUT=Wyniki/Dane/proby_dymne/$NAZWA
mkdir -p Wyniki/Dane/logi "$OUT"
exec > "Wyniki/Dane/logi/znak_normy_${NAZWA}_${SLURM_JOB_ID}.log" 2>&1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
echo "układ $U ($NAZWA), węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
EXTRA=""; [ "${1:-}" = proba ] && { EXTRA="--proba 2"; KROKI="0 ${KROKI##* }"; }
for k in $KROKI; do f=$(printf "%s/k%05d.pt" "$WAGI" "$k"); [ -f "$f" ] || { echo "BRAK $f"; exit 2; }; sha256sum "$f"; done > "$OUT/WAGI_SHA256.txt"
"$PY" -B Kod_eksperymentu/znak_normy_eden.py --uklad $U --wagi "$WAGI" --dane "$DANE" --out "$OUT" --kroki $KROKI --cpus "${SLURM_CPUS_PER_TASK:-4}" $EXTRA
echo "{\"nazwa\": \"$NAZWA\", \"uklad\": \"$U\", \"status\": \"complete\", \"job\": \"$SLURM_JOB_ID\", \"koniec\": \"$(date -Is)\"}" > "$OUT/COMPLETE.json"
echo "KONIEC $NAZWA $(date)"
