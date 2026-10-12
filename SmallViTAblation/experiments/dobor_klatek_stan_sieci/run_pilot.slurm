#!/bin/bash
# Five independent Nano runs; the queue must already be initialized by the caller.
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --job-name=network_state_pilot
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${ROOT:?}" "${PROJECT_DIR:?}" "${DATA_ARCHIVE:?}" "${PROTOCOL:?}" "${PY:?}" "${QUEUE_DIR:?}" "${RESULTS_DIR:?}"
test -d "$ROOT" && test -d "$PROJECT_DIR" && test -d "$QUEUE_DIR"
test -s "$DATA_ARCHIVE" && test -s "$PROTOCOL" && test -x "$PY"
test -s "$(dirname "$PROTOCOL")/profiles.yaml"
test -s "$PROJECT_DIR/eden/run_node.slurm"
"$PY" -B "$PROJECT_DIR/eden/queue.py" status "$QUEUE_DIR" --json | \
  "$PY" -c 'import json,sys; s=json.load(sys.stdin); c=s["counts"]; assert c["pending"] == 5 and all(c[k] == 0 for k in ("running","done","failed")), "pilot requires exactly five fresh pending runs"'
export N_WORKERS=5
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONPATH
cd "$PROJECT_DIR"
# run_node provides node-local extraction and EXIT/INT/TERM cleanup.
bash "$PROJECT_DIR/eden/run_node.slurm"
"$PY" -B "$PROJECT_DIR/eden/queue.py" status "$QUEUE_DIR" --json | \
  "$PY" -c 'import json,sys; s=json.load(sys.stdin); print(json.dumps(s,indent=2)); c=s["counts"]; sys.exit(0 if c["done"] == 5 and all(c[k] == 0 for k in ("running","pending","failed")) else 1)'
