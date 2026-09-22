#!/bin/bash
# Patch-token particles of the L0 detector on the label-eligible POOL (sparse frames of the static
# campaign archive). One GPU, minutes. Required: ROOT, PY, DATA_ARCHIVE, L0_CHECKPOINT.
#SBATCH -A transformers_vsc
#SBATCH -p hopper
#SBATCH --nodelist=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=kl_pool_czastki
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
: "${ROOT:?}" "${PY:?}" "${DATA_ARCHIVE:?}" "${L0_CHECKPOINT:?}"
SCRATCH="/dev/shm/${USER}_${SLURM_JOB_ID:-$$}_kl_pool"
cleanup() { rm -rf -- "$SCRATCH" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
mkdir -p "$SCRATCH/data" "$ROOT/czastki"
export TMPDIR="$SCRATCH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
unset PYTHONPATH
test -s "$DATA_ARCHIVE" && test -s "$L0_CHECKPOINT"
tar -xf "$DATA_ARCHIVE" -C "$SCRATCH/data"
DATA="$(dirname "$(find "$SCRATCH/data" -maxdepth 2 -name pool_manifest.jsonl | head -1)")"
"$PY" -B -P - "$DATA/pool_manifest.jsonl" "$SCRATCH/pool_train.jsonl" <<'PYEOF'
import json, sys
n = 0
with open(sys.argv[1], encoding='utf-8') as src, open(sys.argv[2], 'w', encoding='utf-8') as dst:
    for line in src:
        r = json.loads(line)
        if r.get('split', 'train') == 'train':
            dst.write(json.dumps(dict(frame_id=r['frame_id'], path=r['path'])) + '\n'); n += 1
print('pool train frames', n)
PYEOF
"$PY" -B -P "$ROOT/code/patch_particles.py" --frame-list "$SCRATCH/pool_train.jsonl" --data-root "$DATA" \
    --checkpoint "$L0_CHECKPOINT" --tag pool_b0500 --out "$ROOT/czastki" --full-every 1000000
du -sh "$ROOT/czastki/CZASTKI_pool_b0500"
