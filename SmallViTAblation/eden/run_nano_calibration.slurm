#!/bin/bash
# Three independent Nano trainings, one GPU per worker; no automatic job chain.
# Required: PROJECT_DIR, QUEUE_DIR, DATA_ARCHIVE, PY, PROTOCOL.
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=nano_calibration
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${PROJECT_DIR:?}" "${QUEUE_DIR:?}" "${DATA_ARCHIVE:?}" "${PY:?}" "${PROTOCOL:?}"
export N_WORKERS=3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
cd "$PROJECT_DIR"
"$PY" -B - <<'PY'
import torch, rfdetr.training
if torch.cuda.device_count() != 3 or not torch.cuda.is_bf16_supported():
    raise SystemExit('Expected three CUDA devices supporting BF16')
print({'cuda': torch.version.cuda, 'torch': torch.__version__,
       'gpus': [torch.cuda.get_device_name(i) for i in range(3)]}, flush=True)
PY
# Header directives inside run_node.slurm are ignored by bash; this allocation
# remains three GPUs, shared node, 24 CPUs and 128 GiB.
bash "$PROJECT_DIR/eden/run_node.slurm"
"$PY" "$PROJECT_DIR/eden/queue.py" status "$QUEUE_DIR" --json | "$PY" -c '
import json, sys
s=json.load(sys.stdin); print(json.dumps(s, indent=2))
c=s["counts"]
sys.exit(0 if c["done"] == 3 and all(c[k] == 0 for k in ("failed","pending","running")) else 1)
'
