#!/bin/bash
#SBATCH --job-name=plarge
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper,short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --array=0-2
#SBATCH --signal=B:TERM@120
# Jeden obrot petli od poczatku dostrajania, RF-DETR LargeDeprecated (PLAN.md petla_od_poczatku_large).
# Argumenty: $1 = obrot K (0 = wspolna rozgrzewka od wag COCO, tylko zadanie 0), $2 = ziarno S.
# Tablica 0-2 = petle koopman, losowe, bez (dla K >= 1). Zgloszenie: sbatch --array=0-0 obrot_large.slurm 0 42
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
K=${1:?obrot}; S=${2:?ziarno}
BASE=/home2/faculty/bpiotrowski
ROOT="$BASE/smallvit_runs/2026-09-24_koopman_gesta_historia"
OLD="$BASE/smallvit_runs/2026-09-24_koopman_fastgan_petla"
MY="$BASE/smallvit_runs/2026-09-27_petla_large"
PY="$BASE/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$BASE/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PETLE=(koopman losowe bez)
KK=$(printf '%02d' "$K")
if [ "$K" -eq 0 ]; then P=wspolny; else P=${PETLE[$SLURM_ARRAY_TASK_ID]}; fi
OUT="$MY/results/s$S/$P/k$KK"
if [ -f "$OUT/REPORT.json" ]; then echo "pomijam gotowe $OUT"; exit 0; fi
if [ "$K" -eq 0 ]; then
  WAGI="$MY/modele/rf-detr-large.pth"; SEL="$MY/selekcje/baza.json"
elif [ "$K" -eq 1 ]; then
  WAGI="$MY/results/s$S/wspolny/k00/trening/model_0000160.pth"; SEL="$MY/selekcje/s${S}_${P}_k$KK.json"
else
  WAGI="$MY/results/s$S/$P/k$(printf '%02d' $((K - 1)))/trening/model_0000160.pth"; SEL="$MY/selekcje/s${S}_${P}_k$KK.json"
fi
for f in "$WAGI" "$SEL" "$MY/config_seed$S.json" "$MY/code/kontynuacja_obrot.py" "$MY/code/trening.py" "$MY/code/czastki.py" "$ROOT/probe/manifest.json" "$ARCHIVE"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
SCRATCH=$(mktemp -d "/dev/shm/plarge-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/plarge-*) rm -rf -- "$SCRATCH";; *) exit 91;; esac; }
trap cleanup EXIT INT TERM
export MPLCONFIGDIR="$SCRATCH/mpl"
cp -a "$ROOT/code" "$SCRATCH/code"
cp "$MY/code/trening.py" "$MY/code/czastki.py" "$MY/code/deterministyka.py" "$SCRATCH/code/"
export PYTHONPATH="$SCRATCH/code:$OLD/deps:$BASE/smallvit_runs/2026-09-23_fastgan/deps"
"$PY" -c "import json,sys; c=json.load(open(sys.argv[1])); c['obrot']=int(sys.argv[2]); json.dump(c,open(sys.argv[3],'w'),indent=1)" \
  "$MY/config_seed$S.json" "$K" "$SCRATCH/config.json"
mkdir -p "$SCRATCH/data" "$MY/results/s$S/$P"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
echo "obrot $K, petla $P, ziarno $S, wagi $WAGI, node $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), start $(date)"
rm -rf "$OUT"
"$PY" -u -B "$MY/code/kontynuacja_obrot.py" --code "$SCRATCH/code" --config "$SCRATCH/config.json" \
  --pool "$SCRATCH/data/pool_manifest.jsonl" --data-root "$SCRATCH/data" --selection "$SEL" \
  --weights "$WAGI" --probe "$ROOT/probe/manifest.json" --out "$OUT" --ramie "$P"
cp "$SCRATCH/config.json" "$OUT/config_obrotu.json"
echo "KONIEC obrot $K petla $P ziarno $S $(date)"
