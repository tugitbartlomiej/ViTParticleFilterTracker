#!/bin/bash
#SBATCH --job-name=sdimpk8
#SBATCH --account=transformers_vsc
#SBATCH --partition=hopper
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=23:30:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_vit_sd_k8000_ulepszenia/logi/%x_%j.out
set -euo pipefail
B=/home2/faculty/bpiotrowski
MY=$B/smallvit_runs/2026-10-07_vit_sd_k8000_ulepszenia
BASE=$B/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy
OLD=$B/smallvit_runs/2026-09-30_dyfuzja_sd_filmy
PREVIOUS=$B/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy
PY=$B/venvs/rfdetr_nano_clean_1_10_1/bin/python
MODE=${1:-train}
SEGMENT=${2:-1}
[[ "$MODE" == train || "$MODE" == smoke ]]
[[ "$SEGMENT" =~ ^[1-9][0-9]*$ ]]
(( SEGMENT <= 10 ))
OUT="$MY/wyniki/porownanie"
if [[ "$MODE" == smoke ]]; then OUT="$MY/wyniki/smoke_$SLURM_JOB_ID"; fi
mkdir -p "$OUT"
exec 8>"$MY/.training.lock"
flock -n 8 || { echo 'Own experiment already executing'; exit 2; }
SCRATCH=$(mktemp -d "/dev/shm/sdimpk8-${SLURM_JOB_ID}-XXXXXX")
cleanup() {
  local resolved
  resolved=$(realpath -m -- "$SCRATCH")
  case "$resolved" in /dev/shm/sdimpk8-${SLURM_JOB_ID}-*) rm -rf -- "$resolved";; *) echo "Unsafe scratch path retained: $resolved" >&2;; esac
}
trap cleanup EXIT
TIME_REQUESTED=0
CANCEL_REQUESTED=0
WAIT_INTERRUPTED=0
PID=
time_stop() { TIME_REQUESTED=1; WAIT_INTERRUPTED=1; touch "$OUT/STOP_TIME"; }
cancel_stop() { CANCEL_REQUESTED=1; WAIT_INTERRUPTED=1; touch "$OUT/STOP"; }
trap time_stop USR1
trap cancel_stop TERM INT
if [[ -f "$OUT/STOP" ]]; then echo 'Manual STOP preserved'; exit 0; fi
rm -f -- "$OUT/STOP_TIME"
export TMPDIR="$SCRATCH" HF_HOME="$SCRATCH/hf" TRITON_CACHE_DIR="$SCRATCH/triton"
export PYTHONPATH="$BASE/deps:$PREVIOUS/deps:$OLD/deps"
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
cd "$MY/kod"
ARGS=(--manifest "$BASE/wejscia/FILMY.json" --vit "$BASE/wejscia/MALY_VIT_K8000_s47_losowe.pth"
      --checkpoint "$BASE/wyniki/trening/best_sd.pt" --output "$OUT")
if [[ "$MODE" == smoke ]]; then
  "$PY" -B -m unittest discover -s . -p 'test_*.py' -v
  ARGS+=(--smoke)
fi
nvidia-smi
"$PY" -B orchestrate.py "${ARGS[@]}" &
PID=$!
RC=0
while true; do
  WAIT_INTERRUPTED=0
  if wait "$PID"; then RC=0; else RC=$?; fi
  if (( WAIT_INTERRUPTED == 1 )); then continue; fi
  break
done
echo "mode=$MODE segment=$SEGMENT rc=$RC"
if (( CANCEL_REQUESTED == 1 )); then exit 143; fi
if [[ "$MODE" == train && "$RC" == 75 && "$TIME_REQUESTED" == 1 ]]; then
  (( SEGMENT < 10 )) || exit 75
  FRESH=$(find "$OUT" -mindepth 2 -maxdepth 2 -name latest_checkpoint.pt -newer "$OUT/STOP_TIME" -print -quit)
  [[ -n "$FRESH" || "$OUT/RESUME_BOUNDARY.json" -nt "$OUT/STOP_TIME" ]] || { echo 'No fresh recovery checkpoint or safe boundary'; exit 75; }
  exec 9>"$MY/.submission.lock"; flock -x -w 30 9
  DUP=$(squeue -h -u "$(id -un)" -o '%A|%j' | awk -F'|' -v current="$SLURM_JOB_ID" '$2=="sdimpk8" && $1!=current {print $1}')
  [[ -z "$DUP" ]] || exit 75
  NEXT=$((SEGMENT+1))
  J=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" "$MY/kod/trening.slurm" train "$NEXT")
  printf '%s train segment=%s job=%s parent=%s\n' "$(date -Is)" "$NEXT" "$J" "$SLURM_JOB_ID" >> "$MY/logi/submissions.log"
fi
exit "$RC"
