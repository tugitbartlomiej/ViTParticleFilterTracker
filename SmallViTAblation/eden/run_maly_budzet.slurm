#!/bin/bash
# Jeden bieg RF-DETR Nano na malym budzecie (250 klatek) z wczesnym zatrzymaniem.
# Przepis: configs/protocol_maly_budzet.yaml (sufit 60 epok, patience 15, min_delta 0,001),
# profil nano_maly_budzet. Strata na walidacji zapisywana co epoke.
# Wymagane: ROOT, PROJECT_DIR, DATA_TAR, PROTOCOL, PY, VARIANT_PATH, RUN_ID, TRAIN_SEED.
#SBATCH -A transformers_vsc
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=kl_maly
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${ROOT:?}" "${PROJECT_DIR:?}" "${DATA_TAR:?}" "${PROTOCOL:?}" "${PY:?}" "${VARIANT_PATH:?}" "${RUN_ID:?}" "${TRAIN_SEED:?}"
if [[ -d /raid && -w /raid ]]; then SCRATCH="/raid/${USER}_${SLURM_JOB_ID:-$$}_kl_maly"
else SCRATCH="/dev/shm/${USER}_${SLURM_JOB_ID:-$$}_kl_maly"; fi
cleanup() { rm -rf -- "$SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
mkdir -p "$SCRATCH/data" "$ROOT/results_maly_budzet" slurm_logs
export TMPDIR="$SCRATCH" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor" MPLCONFIGDIR="$SCRATCH/mpl"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
unset PYTHONPATH
test -s "$DATA_TAR" && test -s "$VARIANT_PATH" && test -s "$PROTOCOL"
test ! -e "$ROOT/results_maly_budzet/$RUN_ID"
tar -xf "$DATA_TAR" -C "$SCRATCH/data"
DATA="$(dirname "$(find "$SCRATCH/data" -maxdepth 2 -name pool_manifest.jsonl | head -1)")"
test -n "$DATA"
cd "$PROJECT_DIR"

"$PY" -B pipeline/train/run_with_validation.py --run-id "$RUN_ID" --data-root "$DATA" \
    --out "$ROOT/results_maly_budzet" --protocol "$PROTOCOL" --variant-file "$VARIANT_PATH" \
    --train-seed "$TRAIN_SEED" --profile nano_maly_budzet --expected-epochs 0 \
    --compress-predictions --val-loss --early-stop

RUN="$ROOT/results_maly_budzet/$RUN_ID"
"$PY" -B - "$RUN" <<'PYEOF'
import json, sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
run = Path(sys.argv[1])
ea = EventAccumulator(str(run / "training_logs"), size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]
(run / "TELEMETRIA.json").write_text(json.dumps(
    {t: [[s.step, float(s.value)] for s in ea.Scalars(t)] for t in tags}))
mp = [s.value for s in ea.Scalars("val/mAP_50_95")] if "val/mAP_50_95" in tags else []
print("epok:", len(mp), "najlepsze mAP:", max(mp) if mp else None,
      "w epoce:", mp.index(max(mp)) if mp else None)
PYEOF
echo "[kl_maly] gotowe: $RUN"
