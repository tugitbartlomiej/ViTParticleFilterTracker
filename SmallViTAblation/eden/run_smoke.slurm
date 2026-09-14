#!/bin/bash
# Short, single-GPU entry point. Uses the existing staging and cleanup runner.
# Create $PROJECT_DIR/slurm_logs before sbatch, and submit from $PROJECT_DIR.
# Required exported variables: PROJECT_DIR, QUEUE_DIR, DATA_ARCHIVE, PY.

#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=nano_smoke
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${PROJECT_DIR:?Set the absolute SmallViTAblation directory}"
: "${QUEUE_DIR:?Set an initialized smoke queue directory}"
: "${DATA_ARCHIVE:?Set a prepared smoke data tar archive}"
: "${PY:?Set the absolute Python executable in the RF-DETR environment}"

[[ "$PROJECT_DIR" == /* && -d "$PROJECT_DIR" ]]
[[ "$PY" == /* && -x "$PY" ]]
[[ -f "$DATA_ARCHIVE" && -f "$QUEUE_DIR/queue.jsonl" ]]
export N_WORKERS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/results/nano_smoke}"
cd "$PROJECT_DIR"

"$PY" - <<'PY'
import importlib.metadata as metadata
import torch
import rfdetr
import pycocotools
import yaml
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device visible in the allocated SLURM job")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("nano_smoke requires BF16 support")
print({"torch": torch.__version__, "rfdetr": metadata.version("rfdetr"),
       "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}, flush=True)
PY

bash "$PROJECT_DIR/eden/run_node.slurm"
# The generic worker continues after a failed task. A smoke job must report it.
"$PY" "$PROJECT_DIR/eden/queue.py" status "$QUEUE_DIR" --json | "$PY" -c '
import json, sys
status = json.load(sys.stdin)
print(json.dumps(status, indent=2))
counts = status["counts"]
sys.exit(0 if counts["done"] > 0 and all(counts[key] == 0 for key in ("pending", "running", "failed")) else 1)
'
