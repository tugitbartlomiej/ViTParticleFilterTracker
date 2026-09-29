#!/bin/bash
#SBATCH --job-name=rownlat
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-29_rownanie_latentu/logi/%x_%A_%a.out
# Równanie Langevina latentu małego ViT ze sterowaniem danymi (maly_vit_rownanie_latentu/PLAN.md): nagranie uczenia
# krok po kroku z siłami paczek (nagranie.py --sily) i ocena równania (t2_sterowanie.py). To samo co lokalnie dla
# ziarna 47, tylko na kolejnych ziarnach. Nagranie (~6 GB) tylko w /dev/shm węzła, kasowane pułapką; do katalogu
# wyników trafia jeden JSON na ziarno.
# Próba kodu:  sbatch --array=0-0 rownanie_eden.slurm 47 proba      (1300 kroków, dopasowanie 1000–1150, test 1150–1299)
# Kampania:    sbatch --array=0-9 rownanie_eden.slurm 47            (ziarna 47–56, 3000 kroków, 1000–2000 / 2000–2999)
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
BAZA=${1:?pierwsze ziarno}; TRYB=${2:-pelny}
S=$((BAZA + SLURM_ARRAY_TASK_ID))
B=/home2/faculty/bpiotrowski
MY="$B/smallvit_runs/2026-09-29_rownanie_latentu"
V1="$B/smallvit_runs/2026-09-28_maly_vit_od_zera"
PY="$B/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$B/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
for f in "$V1/deps/timm/__init__.py" "$MY/kod/nagranie.py" "$MY/kod/t2_sterowanie.py" "$MY/kod/t1_bez_sterowania.py" \
         "$MY/kod/dane.py" "$MY/kod/kod_vit/dane.py" "$MY/kod/kod_vit/model.py" "$MY/kod/kod_vit/trening.py" \
         "$V1/wejscia/probe/manifest.json" "$V1/wejscia/pretrain/train/_annotations.coco.json" "$ARCHIVE"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
if [ "$TRYB" = proba ]; then
  KROKI=1300; FIT="1000 1150"; TEST="1150 1299"; WYN="$MY/wyniki/proba_s$S.json"
else
  KROKI=3000; FIT="1000 2000"; TEST="2000 2999"; WYN="$MY/wyniki/T2_s$S.json"
fi
[ -f "$WYN" ] && { echo "wynik już jest: $WYN"; exit 0; }
SCRATCH=$(mktemp -d "/dev/shm/rownlat-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/rownlat-*) rm -rf -- "$SCRATCH";; *) exit 91;; esac; }
trap cleanup EXIT INT TERM
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE
export MALY_VIT_WEJSCIA="$V1/wejscia" MALY_VIT_DANE="$SCRATCH/data" MALY_VIT_CACHE="$SCRATCH/cache" MALY_VIT_KOD="$MY/kod/kod_vit"
export PYTHONPATH="$V1/deps"   # timm 0.9.16 (brak w venv Edenu; kopia z lokalnego środowiska)
export MPLCONFIGDIR="$SCRATCH/mpl" TORCH_HOME="$SCRATCH/torch" XDG_CACHE_HOME="$SCRATCH/xdg"
mkdir -p "$SCRATCH/data" "$SCRATCH/cache" "$MY/wyniki"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
[ -f "$SCRATCH/data/pool_manifest.jsonl" ] && [ -d "$SCRATCH/data/images" ] || { echo "złe archiwum"; exit 3; }
echo "ziarno $S, tryb $TRYB, węzeł $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
cd "$MY/kod"
"$PY" -u -B nagranie.py --ziarno "$S" --kroki "$KROKI" --sily --out "$SCRATCH/nagranie"
echo "nagranie gotowe $(date), $(du -sh "$SCRATCH/nagranie" | cut -f1)"
"$PY" -u -B t2_sterowanie.py --katalog "$SCRATCH/nagranie" --ziarno "$S" --fit $FIT --test $TEST --wynik "$WYN"
echo "KONIEC ziarno $S $(date)"
