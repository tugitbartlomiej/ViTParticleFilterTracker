#!/bin/bash
#SBATCH --job-name=sd0k8000
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=23:30:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy/logi/%x_%j.out
set -euo pipefail
B=/home2/faculty/bpiotrowski
MY=$B/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy
OLD=$B/smallvit_runs/2026-09-30_dyfuzja_sd_filmy
PREVIOUS=$B/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy
PY=$B/venvs/rfdetr_nano_clean_1_10_1/bin/python
MODE=${1:-train}
SEGMENT=${2:-1}
MAX_SEGMENTS=${3:-10}
[[ "$MODE" == train || "$MODE" == smoke ]]
[[ "$SEGMENT" =~ ^[1-9][0-9]*$ && "$MAX_SEGMENTS" =~ ^[1-9][0-9]*$ ]]
(( MAX_SEGMENTS <= 10 && SEGMENT <= MAX_SEGMENTS ))
mkdir -p "$MY/wyniki" "$MY/logi"
OUT="$MY/wyniki/trening"
if [[ "$MODE" == smoke ]]; then OUT="$MY/wyniki/smoke_${SLURM_JOB_ID}"; fi
mkdir -p "$OUT"
# Keep the lock for the entire execution, including checkpoint publication.
exec 8>"$MY/.training.lock"
flock -n 8 || { echo 'Another execution holds the training lock' >&2; exit 2; }
SCRATCH=$(mktemp -d "/dev/shm/sd0k8000-${SLURM_JOB_ID}-XXXXXX")
cleanup() { case "$SCRATCH" in /dev/shm/sd0k8000-*) rm -rf -- "$SCRATCH";; esac; }
trap cleanup EXIT
TIME_REQUESTED=0
WAIT_INTERRUPTED=0
PID=
time_stop() {
    TIME_REQUESTED=1
    WAIT_INTERRUPTED=1
    # run.py recognizes STOP_TIME, synchronizes ranks, saves, and exits 75.
    touch "$OUT/STOP_TIME"
}
cancel_stop() {
    # Cancellation is never eligible for automatic continuation.
    touch "$OUT/STOP"
    if [[ -n "$PID" ]]; then kill -TERM "$PID" 2>/dev/null || true; fi
    exit 143
}
trap time_stop USR1
trap cancel_stop TERM INT
# Only stale time requests from the previous segment are removed. A manual
# STOP is deliberately preserved. Checkpoints and training outputs are untouched.
rm -f -- "$OUT/STOP_TIME"
export TMPDIR="$SCRATCH" HF_HOME="$SCRATCH/hf" TRITON_CACHE_DIR="$SCRATCH/triton"
export PYTHONPATH="$MY/deps:$PREVIOUS/deps:$OLD/deps"
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
cd "$MY/kod"
test "$(sha256sum "$MY/wejscia/MALY_VIT_K8000_s47_losowe.pth" | cut -d' ' -f1)" = 0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b
nvidia-smi
COMMON=(--manifest "$MY/wejscia/FILMY.json" --vit "$MY/wejscia/MALY_VIT_K8000_s47_losowe.pth" --output "$OUT" --resume auto --frame-stride 600)
if [[ "$MODE" == smoke ]]; then
    ARGS=(--smoke --vae-steps 2 --sd-steps 2 --eval-every 1 --batch-size 1 --workers 2)
else
    ARGS=(--vae-steps 30000 --sd-steps 100000 --eval-every 500 --batch-size 4 --workers 2)
fi
"$PY" -B deploy.py launch-ddp --standalone --nproc_per_node=4 run.py "${COMMON[@]}" "${ARGS[@]}" &
PID=$!
# A trapped USR1 interrupts wait. Re-enter wait until the child really exits.
RC=0
while true; do
    WAIT_INTERRUPTED=0
    if wait "$PID"; then RC=0; else RC=$?; fi
    if (( WAIT_INTERRUPTED == 1 )); then continue; fi
    break
done
printf 'mode=%s segment=%s/%s rc=%s time_signal=%s\n' "$MODE" "$SEGMENT" "$MAX_SEGMENTS" "$RC" "$TIME_REQUESTED"
if [[ "$MODE" == train && "$RC" == 75 && "$TIME_REQUESTED" == 1 ]]; then
    # Require a published checkpoint after the request, never just an old file.
    FRESH=$(find "$OUT" -maxdepth 1 -type f -name 'latest_checkpoint.pt' -newer "$OUT/STOP_TIME" -print -quit)
    [[ -n "$FRESH" ]] || { echo 'No fresh checkpoint; refusing continuation' >&2; exit 75; }
    if (( SEGMENT >= MAX_SEGMENTS )); then
        echo 'Segment limit reached; checkpoints retained for manual resume.'
        exit 75
    fi
    exec 9>"$MY/.submission.lock"
    flock -x -w 30 9
    QUEUE=$(squeue -h -u "$(id -un)" -o '%A|%j')
    DUPLICATE=$(printf '%s\n' "$QUEUE" | awk -F'|' -v current="$SLURM_JOB_ID" '$2 == "sd0k8000" && $1 != current {print $1}')
    [[ -z "$DUPLICATE" ]] || { echo "Duplicate sd0k8000 job: $DUPLICATE; refusing continuation" >&2; exit 75; }
    NEXT=$(( SEGMENT + 1 ))
    JOB=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" "$MY/kod/trening.slurm" train "$NEXT" "$MAX_SEGMENTS")
    printf '%s train segment=%s job=%s parent=%s\n' "$(date -Is)" "$NEXT" "$JOB" "$SLURM_JOB_ID" >> "$MY/logi/submissions.log"
    echo "Checkpoint continuation queued: $JOB"
fi
# Zero (completed/plateau/manual stop), failures and smoke never self-submit.
exit "$RC"
