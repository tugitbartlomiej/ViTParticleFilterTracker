#!/bin/bash
# Eden — jeden bieg SLURM = caly wezel hopper (8x H100), 8 workerow po jednym
# na karte, wszyscy z tej samej kolejki. Wzorzec TMPDIR i cleanup zgodnie z
# Eden/EDEN_CONSTRAINTS.md (partycja debug ZABLOKOWANA — nie uzywac).
#
# Wymagane zmienne srodowiskowe przy sbatch (lub edycja ponizszych domyslnych):
#   DATA_ARCHIVE   paczka .tar(.gz) z danymi: images/, annotations_val.json,
#                  pool_manifest.jsonl, variants/  (~7 GB)
#   PROJECT_DIR    katalog SmallViTAblation na klastrze
#   QUEUE_DIR      katalog kolejki (zainicjowany wczesniej przez:
#                  py -3.11 eden/queue.py init QUEUE_DIR --queue-file queue.jsonl)
#   RESULTS_DIR    katalog wynikow (domyslnie $PROJECT_DIR/results)

#SBATCH -A transformers_vsc
#SBATCH -p hopper              # H100 x8, MaxTime 5 dni (NIE debug, NIE hopper-2)
#SBATCH --nodelist=hopper
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=5-00:00:00
#SBATCH --job-name=abla_node
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/SmallViTAblation}"
DATA_ARCHIVE="${DATA_ARCHIVE:-$PROJECT_DIR/data/ablation_data.tar}"
QUEUE_DIR="${QUEUE_DIR:-$PROJECT_DIR/eden/qdir}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/results}"
PROTOCOL="${PROTOCOL:-$PROJECT_DIR/configs/protocol.yaml}"
N_WORKERS="${N_WORKERS:-8}"
PY="${PY:-python3.11}"

# --- TMPDIR: wzorzec z EDEN_CONSTRAINTS.md (/tmp za maly; /raid 14TB, /dev/shm 1TB)
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    TMPDIR="$SLURM_TMPDIR"
elif [[ -d "/raid" && -w "/raid" ]] && [[ "$(df /raid --output=avail 2>/dev/null | tail -1)" -gt 52428800 ]]; then
    TMPDIR="/raid/${USER}_${SLURM_JOB_ID:-$$}_ablation"
elif [[ -w "/dev/shm" ]] && [[ "$(df /dev/shm --output=avail 2>/dev/null | tail -1)" -gt 52428800 ]]; then
    TMPDIR="/dev/shm/${USER}_${SLURM_JOB_ID:-$$}_ablation"
else
    echo "[run_node] ERROR: neither /raid nor /dev/shm has at least 50 GiB free" >&2
    exit 2
fi
DATA_DIR="$TMPDIR/ablation_data"

WORKER_PIDS=()
REQUEUE_PID=""

cleanup() {
    echo "Cleaning up TMPDIR=$TMPDIR ..."
    [[ -n "$REQUEUE_PID" ]] && kill "$REQUEUE_PID" 2>/dev/null || true
    for pid in "${WORKER_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    case "$TMPDIR" in
        /raid/*|/dev/shm/*) rm -rf -- "$TMPDIR" 2>/dev/null || true ;;
        *) echo "[run_node] leaving scheduler-managed TMPDIR=$TMPDIR" ;;
    esac
}
trap cleanup EXIT INT TERM

mkdir -p "$TMPDIR" "$RESULTS_DIR" "$(dirname "$QUEUE_DIR")" slurm_logs

# --- dane rozpakowywane RAZ na wezel (nie raz na bieg!)
echo "[run_node] rozpakowuje $DATA_ARCHIVE -> $DATA_DIR"
mkdir -p "$DATA_DIR"
if [[ "$DATA_ARCHIVE" == *.tar.gz || "$DATA_ARCHIVE" == *.tgz ]]; then
    tar -xzf "$DATA_ARCHIVE" -C "$DATA_DIR"
else
    tar -xf "$DATA_ARCHIVE" -C "$DATA_DIR"
fi

# --- straznik requeue-stale: przywraca do kolejki biegi po ubitych workerach
(
    while true; do
        "$PY" "$PROJECT_DIR/eden/queue.py" requeue-stale "$QUEUE_DIR" \
            --timeout 180 >/dev/null 2>&1 || true
        sleep 60
    done
) &
REQUEUE_PID=$!

# --- 8 workerow, po jednym na karte
for ((w = 0; w < N_WORKERS; w++)); do
    echo "[run_node] start worker $w (GPU $w)"
    "$PY" "$PROJECT_DIR/eden/runner.py" \
        --queue "$QUEUE_DIR" \
        --data "$DATA_DIR" \
        --worker "$w" \
        --protocol "$PROTOCOL" \
        --out "$RESULTS_DIR" \
        &
    WORKER_PIDS+=($!)
done

FAIL=0
for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" || FAIL=1
done

echo "[run_node] wszyscy workerzy zakonczeni"
"$PY" "$PROJECT_DIR/eden/queue.py" status "$QUEUE_DIR"
exit "$FAIL"
