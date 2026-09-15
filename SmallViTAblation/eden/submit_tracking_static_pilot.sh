#!/bin/bash
# Submit immutable preparation and, only after it succeeds, the 36-run campaign.
set -euo pipefail
ROOT="${ROOT:-$HOME/smallvit_runs/2026-09-15_tracking_static36}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/smallvit_runs/2026-09-14_tracking_nano/SmallViTAblation}"
INDEX_DIR="${INDEX_DIR:-$HOME/smallvit_runs/2026-09-14_tracking_nano/index}"
PY="${PY:-$HOME/venvs/rfdetr_nano_clean_1_10_1/bin/python}"
PROTOCOL="${PROTOCOL:-$PROJECT_DIR/configs/protocol_tracking_calibration.yaml}"
PREPARED_DIR="$ROOT/data/tracking_static36"
CACHE_ROOT="$ROOT/cache/dinov2"
QUEUE_DIR="$ROOT/queue/static36"
RESULTS_DIR="$ROOT/results"
DINO_HUB_REPO="facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
export ROOT PROJECT_DIR INDEX_DIR PY PROTOCOL PREPARED_DIR CACHE_ROOT QUEUE_DIR RESULTS_DIR DINO_HUB_REPO
mkdir -p "$ROOT/data" "$ROOT/cache" "$ROOT/queue" "$ROOT/results" "$PROJECT_DIR/slurm_logs"
test ! -e "$PREPARED_DIR" -a ! -e "${PREPARED_DIR}.tar" -a ! -e "$QUEUE_DIR"
prep_job=$(sbatch --parsable --export=ALL "$PROJECT_DIR/eden/prepare_tracking_static_pilot.slurm")
train_job=$(sbatch --parsable --dependency="afterok:$prep_job" --kill-on-invalid-dep=yes \
  --export=ALL "$PROJECT_DIR/eden/run_tracking_static_pilot.slurm")
printf '%s\n' "$prep_job" > "$ROOT/prep_job_id.txt"
printf '%s\n' "$train_job" > "$ROOT/train_job_id.txt"
"$PY" - <<PY
import json
from pathlib import Path
Path('$ROOT/submission.json').write_text(json.dumps({
  'prep_job_id': '$prep_job', 'train_job_id': '$train_job',
  'dependency': 'afterok', 'dino_hub_repo': '$DINO_HUB_REPO',
  'selectors': ['random','uniform_time','dino_coverage','dino_temporal_hybrid'],
  'budgets': [500,1500,2500], 'selection_seed': 42,
  'training_seeds': [42,1337,2024], 'test_evaluated': False,
}, indent=2))
PY
echo "prep_job=$prep_job train_job=$train_job root=$ROOT"
