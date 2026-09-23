#!/bin/bash
# Test obrotu, wydluzenia i tla na gotowych detektorach (bez treningu).
# Wymagane: ROOT, PROJECT_DIR, DATA_TAR, PY.
#SBATCH -A transformers_vsc
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=kl_obrot
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${ROOT:?}" "${PROJECT_DIR:?}" "${DATA_TAR:?}" "${PY:?}"
if [[ -d /raid && -w /raid ]]; then SCRATCH="/raid/${USER}_${SLURM_JOB_ID:-$$}_kl_obrot"
else SCRATCH="/dev/shm/${USER}_${SLURM_JOB_ID:-$$}_kl_obrot"; fi
cleanup() { rm -rf -- "$SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
mkdir -p "$SCRATCH/data" "$ROOT/results_obrot" slurm_logs
export TMPDIR="$SCRATCH" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor" MPLCONFIGDIR="$SCRATCH/mpl"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
unset PYTHONPATH
CK="$ROOT/checkpoints"
MB="$ROOT/results_maly_budzet"
for f in "$DATA_TAR" "$CK/ut_b0500_best.pth" "$CK/ut_b1500_best.pth" \
         "$MB/maly_B1m_random_b250_s42/best.pth" "$MB/maly_K6_identity_cells_b250_s42/best.pth" \
         "$ROOT/czastki/CZASTKI_pool_b0500/baza.npz"; do test -s "$f"; done
tar -xf "$DATA_TAR" -C "$SCRATCH/data" --wildcards 'annotations_val.json' 'images/*'
cd "$PROJECT_DIR"
"$PY" -B -P pipeline/koopman_langevin/tozsamosc_obrot.py --data "$SCRATCH/data" \
    --detektor ut_b0500="$CK/ut_b0500_best.pth" \
    --detektor ut_b1500="$CK/ut_b1500_best.pth" \
    --detektor B1m_250="$MB/maly_B1m_random_b250_s42/best.pth" \
    --detektor K6_250="$MB/maly_K6_identity_cells_b250_s42/best.pth" \
    --baza-koopmana "$ROOT/czastki/CZASTKI_pool_b0500/baza.npz" \
    --out "$ROOT/results_obrot"
echo "[kl_obrot] gotowe"
