#!/usr/bin/env bash
###############################################################################
#  YOLOv8 – 8×GPU Hooper | klaster Eden | automatyczny resume + requeue
###############################################################################
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-4          # <-- wymusza węzeł dgx-2
#SBATCH --gres=gpu:8              # rezerwuje wszystkie 8 GPU węzła
#SBATCH --mem=400G
#SBATCH --time=5-00:00:00         # 3 dni; zmniejsz gdyby hopper miał twarde 48 h
#SBATCH --job-name=yolo_8_gpu_resume
#SBATCH --chdir=/home2/faculty/bpiotrowski/Yolo
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@90        # 90 s przed TIMEOUT → SIGUSR1

set -euo pipefail -x

echo "=== $(date) | SLURM_JOBID=$SLURM_JOB_ID on $(hostname) ==="

###############################################################################
# 1. Środowisko
###############################################################################
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

###############################################################################
# 2. Tymczasowy katalog + przygotowanie danych
###############################################################################
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR=$TMPDIR"

#ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606.tar.gz
ARCHIVE=/mnt/evafs/faculty/home/bpiotrowski/datasets_20250606_test.tar.gz
echo "📦 Rozpakowuję: $ARCHIVE → $TMPDIR"
tar -xzf "$ARCHIVE" -C "$TMPDIR"

python prepare_dataset.py \
       --tmpdir "$TMPDIR" \
       --unified_dir "$TMPDIR/unified_dataset"

DATASET_YAML="$TMPDIR/unified_dataset/dataset.yaml"
if [[ ! -f "$DATASET_YAML" ]]; then
    echo "❌ Brak $DATASET_YAML – prepare_dataset.py nie utworzył YAML-a"
    exit 1
fi
echo "✅ Znaleziono dataset.yaml: $DATASET_YAML"

###############################################################################
# 3. Ustawienia treningu
###############################################################################
PROJECT_DIR=/home2/faculty/bpiotrowski/Yolo 
CHECKPOINT_DIR="${PROJECT_DIR}/train_out_5gpu_hopper/exp/weights"
BEST_MODEL_DIR="${CHECKPOINT_DIR}/best"
EPOCHS=130
# ------------------ PARAMETRY DOMYŚLNE ------------------
BATCH_SIZE=${BATCH_SIZE:-96}        # domyślnie 96     (lub inna wartość)
IMGSZ=${IMGSZ:-640}                 # rozdzielczość wejściowa
LR=${LR:-0.01}                      # learning-rate
WORKERS=${WORKERS:-24}              # wątki dataloadera
DEVICES="0,1,2,3,4,5,6,7"
SAVE_PERIOD=${SAVE_PERIOD:-10}      # co ile epok checkpoint
PATIENCE=${PATIENCE:-50}            # early-stopping
# ---------------------------------------------------------


# wyszukujemy najnowszy checkpoint (last.pt > epoch*.pt)
latest_ckpt=""
if [[ -d "$CHECKPOINT_DIR" ]]; then
    latest_ckpt=$(ls -t "$CHECKPOINT_DIR"/last.pt "$CHECKPOINT_DIR"/epoch*.pt 2>/dev/null | head -n 1 || true)
fi

###############################################################################
# 4. Łapanie SIGUSR1 – zapis checkpointu przed TIMEOUT-em
###############################################################################
on_timeout() {
    echo "[$(date)] ↳ SIGUSR1 odebrany – wysyłam SIGINT do Pythona, aby zapisać checkpoint"
    pkill -INT -P $$ python || true
}
trap on_timeout SIGUSR1

###############################################################################
# 5. Start treningu (resume | fresh)
###############################################################################
cd "$PROJECT_DIR" || mkdir -p "$PROJECT_DIR" && cd "$PROJECT_DIR"

PY_ARGS=(
  --dataset_yaml_path "${DATASET_YAML}"
  --project_dir       "${PROJECT_DIR}"
  --checkpoint_dir    "${CHECKPOINT_DIR}"
  --best_model_dir    "${BEST_MODEL_DIR}"
  --model_size        l
  --epochs            "${EPOCHS}"
  --batch_size        "${BATCH_SIZE}"
  --imgsz             "${IMGSZ}"
  --learning_rate     "${LR}"
  --workers           "${WORKERS}"
  --device            "${DEVICES}"
  --save_period       "${SAVE_PERIOD}"
  --patience          "${PATIENCE}"
  --verbose
)

if [[ -n "$latest_ckpt" ]]; then
    echo "🔄 Wznawiam z checkpointu: $latest_ckpt"
    PY_ARGS+=( --resume --resume_path "$latest_ckpt" )
else
    echo "🆕 Start od zera (brak checkpointu)"
fi

python3 "$HOME/Yolo/yolo-train.py" "${PY_ARGS[@]}"

echo "✓ Trening zakończony z kodem: $?"
