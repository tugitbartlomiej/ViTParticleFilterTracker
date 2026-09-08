#!/bin/bash
# =============================================================================
# Submit ablation training jobs (V2-V7) to SLURM
# V1 uses existing checkpoints — not trained here.
#
# Usage:
#   bash run_ablation_all.sh              # Submit V2-V7 as job array
#   bash run_ablation_all.sh --sequential # Submit one at a time (chained)
#   bash run_ablation_all.sh --check      # Check status only
#   bash run_ablation_all.sh v2 v5 v7     # Submit specific variants
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/run_ablation_variant.slurm"
ABLATION_DIR=/mnt/evafs/faculty/home/bpiotrowski/DETR/ablation
CHECKPOINT=/mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints/checkpoint_epoch_170.pth

# === PREFLIGHT CHECKS ===
preflight() {
    echo "=== Preflight Checks ==="
    local ok=true

    # Check SLURM script
    if [[ ! -f "$SLURM_SCRIPT" ]]; then
        echo "  FAIL: SLURM script not found: $SLURM_SCRIPT"
        ok=false
    else
        echo "  OK: SLURM script"
    fi

    # Check checkpoint
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "  FAIL: Epoch 170 checkpoint not found: $CHECKPOINT"
        ok=false
    else
        echo "  OK: Epoch 170 checkpoint"
    fi

    # Check variant data
    for v in v2 v3 v4 v5 v6 v7; do
        train_json="$ABLATION_DIR/$v/annotations_train.json"
        val_json="$ABLATION_DIR/$v/annotations_val.json"
        if [[ -f "$train_json" ]] && [[ -f "$val_json" ]]; then
            echo "  OK: $v annotations"
        else
            echo "  FAIL: $v missing annotations"
            ok=false
        fi
    done

    # Check training script
    if [[ ! -f /mnt/evafs/faculty/home/bpiotrowski/DETR/detr_train_ablation.py ]]; then
        echo "  FAIL: detr_train_ablation.py not found in ~/DETR/"
        echo "        Copy it: cp $SCRIPT_DIR/detr_train_ablation.py ~/DETR/"
        ok=false
    else
        echo "  OK: detr_train_ablation.py"
    fi

    # Check GPU availability
    echo ""
    echo "=== GPU Availability ==="
    sfree 2>/dev/null || sinfo -p debug,long,experimental -o "%P %G %D %t" 2>/dev/null || echo "  (cannot check — run on Eden)"

    if [[ "$ok" == "false" ]]; then
        echo ""
        echo "PREFLIGHT FAILED — fix issues above before submitting"
        exit 1
    fi
    echo ""
    echo "All checks passed."
}

# === STATUS CHECK ===
check_status() {
    echo "=== Ablation Job Status ==="
    squeue -u "$USER" -o "%.10i %.20j %.10P %.8T %.10M %.6D %.4C %R" | grep -i ablation || echo "  No ablation jobs running"
    echo ""
    echo "=== Checkpoint Status ==="
    for v in v2 v3 v4 v5 v6 v7; do
        ckpt_dir=/mnt/evafs/faculty/home/bpiotrowski/DETR/ablation_checkpoints/$v
        if [[ -d "$ckpt_dir" ]]; then
            latest=$(ls -t "$ckpt_dir"/checkpoint_epoch_*.pth 2>/dev/null | head -1)
            if [[ -n "$latest" ]]; then
                epoch=$(basename "$latest" | sed 's/checkpoint_epoch_//;s/.pth//')
                echo "  $v: epoch $epoch"
            else
                echo "  $v: no checkpoints yet"
            fi
        else
            echo "  $v: not started"
        fi
    done
}

# === MAIN ===
MODE="${1:-parallel}"

case "$MODE" in
    --check|check|status)
        check_status
        exit 0
        ;;
    --sequential|sequential)
        preflight
        echo ""
        echo "=== Submitting 6 jobs SEQUENTIALLY (chained, V2-V7) ==="
        PREV_JOB=""
        for v in v2 v3 v4 v5 v6 v7; do
            if [[ -n "$PREV_JOB" ]]; then
                JOB_ID=$(VARIANT=$v sbatch --dependency=afterok:$PREV_JOB "$SLURM_SCRIPT" | awk '{print $4}')
            else
                JOB_ID=$(VARIANT=$v sbatch "$SLURM_SCRIPT" | awk '{print $4}')
            fi
            echo "  $v: Job $JOB_ID (depends on: ${PREV_JOB:-none})"
            PREV_JOB=$JOB_ID
        done
        echo ""
        echo "All 6 jobs submitted in chain."
        ;;
    v[2-7]*)
        # Submit specific variants
        preflight
        echo ""
        echo "=== Submitting selected variants ==="
        for v in "$@"; do
            if [[ "$v" =~ ^v[2-7]$ ]]; then
                JOB_ID=$(VARIANT=$v sbatch "$SLURM_SCRIPT" | awk '{print $4}')
                echo "  $v: Job $JOB_ID"
            else
                echo "  Skipping unknown or V1 (uses existing ckpt): $v"
            fi
        done
        ;;
    *)
        # Default: parallel job array
        preflight
        echo ""
        echo "=== Submitting job array (6 parallel jobs, V2-V7) ==="
        JOB_ID=$(sbatch --array=2-7 "$SLURM_SCRIPT" | awk '{print $4}')
        echo "  Job array: $JOB_ID (tasks 2-7 = V2-V7)"
        echo "  Monitor: squeue -u $USER"
        echo "  Logs: tail -f logs/ablation_*_${JOB_ID}*.log"
        ;;
esac

echo ""
echo "Check status: bash $0 --check"
