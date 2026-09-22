#!/bin/bash
# Jeden bieg RF-DETR Nano wedlug tego samego przepisu co kampania, ale z PELNA telemetria:
#   - strata na walidacji co epoke (RF-DETR compute_val_loss; w kampanii bylo "auto", czyli pomijane)
#   - wagi zachowane co EVERY epok (domyslnie 2), zeby zegar treningu mial gestsza siatke
#   - latent kazdego zachowanego checkpointu na klipach sondujacych
#   - zrzut wszystkich skalarow TensorBoard do jednego JSON-a
# Przepis optymalizacji NIE ulega zmianie: compute_val_loss dokłada tylko przebieg po walidacji.
# Wymagane: ROOT, PROJECT_DIR, DATA_TAR, PROTOCOL, PY, VARIANT_PATH, RUN_ID, TRAIN_SEED.
# Opcjonalne: EVERY (domyslnie 2).
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodelist=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=kl_telem
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${ROOT:?}" "${PROJECT_DIR:?}" "${DATA_TAR:?}" "${PROTOCOL:?}" "${PY:?}" "${VARIANT_PATH:?}" "${RUN_ID:?}" "${TRAIN_SEED:?}"
EVERY="${EVERY:-2}"
if [[ -d /raid && -w /raid ]]; then SCRATCH="/raid/${USER}_${SLURM_JOB_ID:-$$}_kl_telem"
else SCRATCH="/dev/shm/${USER}_${SLURM_JOB_ID:-$$}_kl_telem"; fi
cleanup() { rm -rf -- "$SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
mkdir -p "$SCRATCH/data" "$ROOT/results_telemetria" "$ROOT/latents_telemetria" slurm_logs
export TMPDIR="$SCRATCH" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor" MPLCONFIGDIR="$SCRATCH/mpl"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
unset PYTHONPATH
test -s "$DATA_TAR" && test -s "$ROOT/package/tracks.json" && test -s "$VARIANT_PATH"
test ! -e "$ROOT/results_telemetria/$RUN_ID"
tar -xf "$DATA_TAR" -C "$SCRATCH/data"
DATA="$(dirname "$(find "$SCRATCH/data" -maxdepth 2 -name pool_manifest.jsonl | head -1)")"
test -n "$DATA"
cd "$PROJECT_DIR"

"$PY" -B pipeline/train/run_with_validation.py --run-id "$RUN_ID" --data-root "$DATA" \
    --out "$ROOT/results_telemetria" --protocol "$PROTOCOL" --variant-file "$VARIANT_PATH" \
    --train-seed "$TRAIN_SEED" --profile nano_static_pilot --expected-epochs 30 \
    --compress-predictions --keep-epoch-checkpoints "$EVERY" --val-loss

RUN="$ROOT/results_telemetria/$RUN_ID"
OUT="$ROOT/latents_telemetria/$RUN_ID"
mkdir -p "$OUT"
COCO="$HOME/.roboflow/models/rf-detr-nano.pth"
"$PY" -B "$ROOT/code/detector_latent.py" --package "$ROOT/package" --checkpoint "$COCO" \
    --tag ep000_coco --out "$OUT"
for ckpt in "$RUN"/epoch_checkpoints/epoch_*.pth; do
    tag="$(basename "$ckpt" .pth)"
    "$PY" -B "$ROOT/code/detector_latent.py" --package "$ROOT/package" --checkpoint "$ckpt" \
        --tag "$tag" --out "$OUT"
done

"$PY" -B - "$RUN" <<'PYEOF'
import json, sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
run = Path(sys.argv[1])
ea = EventAccumulator(str(run / "training_logs"), size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]
out = {t: [[s.step, float(s.value)] for s in ea.Scalars(t)] for t in tags}
(run / "TELEMETRIA.json").write_text(json.dumps(out))
print("tagow:", len(tags))
print("val loss:", "TAK" if any(t.startswith("val/loss") for t in tags) else "BRAK")
PYEOF

ls -la "$OUT" | head
echo "[kl_telem] gotowe: $RUN"
