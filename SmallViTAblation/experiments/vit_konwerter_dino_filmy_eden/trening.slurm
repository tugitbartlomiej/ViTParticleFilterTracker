#!/bin/bash
#SBATCH --job-name=vit_dino_konwerter
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_konwerter_dino_filmy/logi/%x_%j.out
# Konwerter latent ViT -> DINOv2 na strumieniu filmów MP4. Żadne klatki, latenty ani cechy nie trafiają na dysk.
set -uo pipefail
B=/home2/faculty/bpiotrowski
MY=$B/smallvit_runs/2026-10-06_vit_konwerter_dino_filmy
PY=$B/venvs/rfdetr_nano_clean_1_10_1/bin/python
DINO_REPO=$B/.cache/torch/hub/facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8
DINO_WAGI=$B/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
OUT=$MY/wyniki/trening

ROBOCZY=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$ROBOCZY"
ZNACZNIK="$ROBOCZY/.start"; touch "$ZNACZNIK"
sprzataj() {
    find /tmp /dev/shm -maxdepth 1 -user "$(id -u)" \( -name 'tmp*' -o -name 'pymp-*' -o -name 'torch_*' -o -name 'matplotlib-*' \) \
        -newer "$ZNACZNIK" -exec rm -rf -- {} + 2>/dev/null
    rm -rf -- "$ROBOCZY"
    echo "kontrola /tmp po sprzątaniu:"; find /tmp /dev/shm -maxdepth 1 -user "$(id -u)"
    df -h /tmp | tail -1
}
trap sprzataj EXIT INT
trap 'touch "$OUT/STOP"' TERM
export TMPDIR="$ROBOCZY" TORCH_HOME="$ROBOCZY/torch" TRITON_CACHE_DIR="$ROBOCZY/triton"
export PYTHONPATH="$B/smallvit_runs/2026-09-28_maly_vit_od_zera/deps" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$MY/wyniki/smoke"
cd "$MY/kod"
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /tmp | tail -1

WSPOLNE=(--manifest "$MY/wejscia/FILMY.json" --vit-checkpoint "$MY/wejscia/MODEL_VIT_K1000.pt"
         --dino-repo "$DINO_REPO" --dino-weights "$DINO_WAGI")

uruchom() {  # torchrun w tle, żeby trap TERM zadziałał w trakcie biegu
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 run.py "$@" &
    local PID=$! RC=0
    while kill -0 "$PID" 2>/dev/null; do wait "$PID"; RC=$?; done
    return $RC
}

if [ ! -f "$MY/wyniki/smoke/SMOKE.json" ]; then
    uruchom --config config.json "${WSPOLNE[@]}" --output "$MY/wyniki/smoke" --smoke --steps 40 \
        --stats-batches 3 --eval-max-filmow 4 --eval-krok 60 --log-every 10 --warmup 10 || exit 1
fi
"$PY" -c "import json,sys; s=json.load(open('$MY/wyniki/smoke/SMOKE.json')); print('SMOKE passed', s['passed'], 'klatek/s', s['klatek_na_s']); sys.exit(0 if s['passed'] else 3)" || exit 3
uruchom --config config.json "${WSPOLNE[@]}" --output "$OUT"
