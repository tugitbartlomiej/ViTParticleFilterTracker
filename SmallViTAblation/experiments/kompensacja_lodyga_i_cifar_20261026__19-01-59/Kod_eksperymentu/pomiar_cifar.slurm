#!/bin/bash
#SBATCH --job-name=mosty_cifar
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
# Drugi zbiór i druga architektura: pomiar mostów A, B, C na 14 checkpointach biegów CIFAR-10 z 2026-10-13_deit_cifar10
# (DeiT-Tiny s47 z obcięciem i bez; MałyViT s47, s48, wd0, losowe etykiety). cwd = folder eksperymentu.
#   sbatch Kod_eksperymentu/pomiar_cifar.slurm                 # 6 biegów
#   sbatch --array=0 Kod_eksperymentu/pomiar_cifar.slurm proba # próba: 2 checkpointy, 20 obrazów
set -euo pipefail
BIEGI=(s47_obciecie s47_bez_obciecia maly_s47_obciecie maly_s48_obciecie maly_s47_wd0 maly_s47_losowe_etykiety)
MODELE=(deit deit malyvit malyvit malyvit malyvit)
B=${BIEGI[$SLURM_ARRAY_TASK_ID]}; MOD=${MODELE[$SLURM_ARRAY_TASK_ID]}
ZR=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-13_deit_cifar10
PY=/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/przeciwstawne_eden_20261017/environment/venv/bin/python
OUT=Wyniki/Dane/cifar/$B
mkdir -p Wyniki/Dane/logi "$OUT"
exec > "Wyniki/Dane/logi/cifar_${B}_${SLURM_JOB_ID}.log" 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
echo "bieg $B model $MOD, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
EXTRA=""
[ "${1:-}" = proba ] && EXTRA="--kroki 0 93780 --proba 20"
for k in 00000 00300 01000 03000 06000 10000 16000 24000 33000 44000 55000 65000 80000 93780; do
  [ -f "$ZR/wyniki/$B/wagi/k$k.pt" ] || { echo "BRAK checkpointu k$k biegu $B"; exit 2; }
done
sha256sum "$ZR/wyniki/$B/wagi/"k{00000,00300,01000,03000,06000,10000,16000,24000,33000,44000,55000,65000,80000,93780}.pt > "$OUT/WAGI_SHA256.txt"
cp "$ZR/wyniki/$B/dane/OCENY.json" "$OUT/OCENY_treningu.json" 2>/dev/null || true
"$PY" -B Kod_eksperymentu/pomiar_mosty_cifar.py --wagi "$ZR/wyniki/$B/wagi" --dane "$ZR/dane/cifar10_uint8.npz" --out "$OUT" --model $MOD --cpus 3 $EXTRA
echo "{\"bieg\": \"$B\", \"model\": \"$MOD\", \"status\": \"complete\", \"job\": \"$SLURM_JOB_ID\", \"koniec\": \"$(date -Is)\"}" > "$OUT/COMPLETE.json"
echo "KONIEC $B $(date)"
