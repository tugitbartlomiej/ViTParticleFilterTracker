#!/bin/bash
#SBATCH --job-name=petla
#SBATCH --account=transformers_vsc
#SBATCH --partition=short,hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --array=0-2
#SBATCH --signal=B:TERM@120
# Jeden obrot petli uczenia (PLAN.md): tablica 0-2 = petle koopman, losowe, bez; argument $1 = numer obrotu K (1..8).
# Wagi startowe: K = 1 model_0000245.pth biegu 1810662, dalej model_0000165.pth poprzedniego obrotu tej samej petli.
# Zgloszenie: sbatch obrot.slurm K
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
K=${1:?numer obrotu}
BASE=/home2/faculty/bpiotrowski
ROOT="$BASE/smallvit_runs/2026-09-24_koopman_gesta_historia"
OLD="$BASE/smallvit_runs/2026-09-24_koopman_fastgan_petla"
MY="$BASE/smallvit_runs/2026-09-26_petla_uczenia"
PY="$BASE/venvs/rfdetr_nano_clean_1_10_1/bin/python"
ARCHIVE="$BASE/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 USE_TF=0 USE_FLAX=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PETLE=(koopman losowe bez)
P=${PETLE[$SLURM_ARRAY_TASK_ID]}
KK=$(printf '%02d' "$K")
OUT="$MY/results/$P/k$KK"
if [ -f "$OUT/REPORT.json" ]; then echo "pomijam gotowe $OUT"; exit 0; fi
if [ "$K" -eq 1 ]; then
  WAGI="$ROOT/results/run_1810662/pretrain/model_0000245.pth"
else
  WAGI="$MY/results/$P/k$(printf '%02d' $((K - 1)))/trening/model_0000165.pth"
fi
SEL="$MY/selekcje/${P}_k$KK.json"
for f in "$WAGI" "$SEL" "$MY/config_seed42.json" "$MY/code/kontynuacja_obrot.py" "$ROOT/probe/manifest.json" "$ARCHIVE"; do
  [ -f "$f" ] || { echo "BRAK $f"; exit 2; }
done
SCRATCH=$(mktemp -d "/dev/shm/petla-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/petla-*) rm -rf -- "$SCRATCH";; *) exit 91;; esac; }
trap cleanup EXIT INT TERM
export MPLCONFIGDIR="$SCRATCH/mpl"
cp -a "$ROOT/code" "$SCRATCH/code"
export PYTHONPATH="$SCRATCH/code:$OLD/deps:$BASE/smallvit_runs/2026-09-23_fastgan/deps"
mkdir -p "$SCRATCH/data" "$MY/results/$P"
tar -xf "$ARCHIVE" -C "$SCRATCH/data"
echo "obrot $K, petla $P, wagi $WAGI, node $(hostname), scratch $SCRATCH, start $(date)"
rm -rf "$OUT"
"$PY" -u -B "$MY/code/kontynuacja_obrot.py" --code "$SCRATCH/code" --config "$MY/config_seed42.json" \
  --pool "$SCRATCH/data/pool_manifest.jsonl" --data-root "$SCRATCH/data" --selection "$SEL" \
  --weights "$WAGI" --probe "$ROOT/probe/manifest.json" --out "$OUT" --ramie "$P"
echo "KONIEC obrot $K petla $P $(date)"
