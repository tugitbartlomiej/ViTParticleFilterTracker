#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper,short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=fastgan_tool
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
ROOT="$HOME/smallvit_runs/2026-09-23_fastgan"
PY="$HOME/venvs/rfdetr_nano_clean_1_10_1/bin/python"
OUT="$ROOT/results_50k"
SCRATCH="$(mktemp -d "/dev/shm/${USER}_fastgan_${SLURM_JOB_ID}_XXXXXX")"
child=''
cleanup() {
    case "$SCRATCH" in /dev/shm/"${USER}"_fastgan_"${SLURM_JOB_ID}"_*) rm -rf -- "$SCRATCH" ;; esac
}
request_stop() { mkdir -p "$OUT"; touch "$OUT/STOP"; }
trap cleanup EXIT
trap request_stop TERM INT
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
export TMPDIR="$SCRATCH" TORCH_HOME="$ROOT/torch-cache" PYTHONPATH="$ROOT/deps"
cd "$ROOT"
sha256sum -c INPUT_SHA256SUMS
tar -xf eden_inputs.tar -C "$SCRATCH"
test ! -e "$OUT"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$ROOT/GPU.csv"
"$PY" -B -m pip freeze > "$ROOT/ENVIRONMENT.txt"
"$PY" -B "$SCRATCH/trening_lokalny.py" --steps 50000 --batch 2 --preview-every 1000 --checkpoint-every 5000 --resume "$ROOT/resume.pth" --out "$OUT" &
child=$!
set +e
wait "$child"
rc=$?
# A trapped signal can interrupt wait while Python is saving the requested checkpoint.
if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
set -e
printf '%s\n' "$rc" > "$ROOT/EXIT_CODE.txt"
if [ "$rc" -eq 0 ]; then "$PY" -B "$ROOT/pakuj_wynik.py" "$ROOT"; fi
exit "$rc"
