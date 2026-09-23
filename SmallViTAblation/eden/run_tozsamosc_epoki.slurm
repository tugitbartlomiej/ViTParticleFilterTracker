#!/bin/bash
# Test obrotu, wydluzenia i tla na wagach z kolejnych epok (3 ziarna, B2 1500 klatek, co 2 epoki)
# oraz na wagach biegu z sonda (250 klatek). Z kontrolami wklejania (tlo_swoje, tlo_film).
# Wymagane: ROOT, PROJECT_DIR, DATA_TAR, PY.
#SBATCH -A transformers_vsc
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=kl_epoki
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${ROOT:?}" "${PROJECT_DIR:?}" "${DATA_TAR:?}" "${PY:?}"
if [[ -d /raid && -w /raid ]]; then SCRATCH="/raid/${USER}_${SLURM_JOB_ID:-$$}_kl_epoki"
else SCRATCH="/dev/shm/${USER}_${SLURM_JOB_ID:-$$}_kl_epoki"; fi
cleanup() { rm -rf -- "$SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
OUT="$ROOT/results_obrot_epoki"
mkdir -p "$SCRATCH/data" "$OUT" slurm_logs
export TMPDIR="$SCRATCH" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor" MPLCONFIGDIR="$SCRATCH/mpl"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
unset PYTHONPATH
test -s "$DATA_TAR"
ARGS=()
for s in 42 1337 2024; do
  R="$ROOT/results_telemetria/telemetria_uniform_time_b1500_s$s"
  for f in "$R"/epoch_checkpoints/epoch_*.pth; do
    test -s "$f"; e=$(basename "$f" .pth); ARGS+=(--detektor "b1500_s${s}_${e}=$f")
  done
done
SONDA="$ROOT/results_sonda/sonda_B1m_random_b250_s42"
for f in best last; do test -s "$SONDA/$f.pth" && ARGS+=(--detektor "sonda250_${f}=$SONDA/$f.pth"); done
echo "[kl_epoki] detektorow: $(( ${#ARGS[@]} / 2 ))"
tar -xf "$DATA_TAR" -C "$SCRATCH/data" --wildcards 'annotations_val.json' 'images/*'
cd "$PROJECT_DIR"
"$PY" -B -P pipeline/koopman_langevin/tozsamosc_obrot.py --data "$SCRATCH/data" --lekko "${ARGS[@]}" --out "$OUT"
echo "[kl_epoki] gotowe"
