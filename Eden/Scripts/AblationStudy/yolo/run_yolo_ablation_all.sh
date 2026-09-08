#!/bin/bash
# =============================================================================
# Submit YOLO ablation training jobs (YE-A, YE-B, YE-C) to SLURM.
#
# YE-B target=DETR -> uzywa detr_train_ablation.py z folderu ../detr/
# YE-A i YE-C target=YOLO -> uzywaja run_yolo_ablation_variant.slurm
#
# Usage:
#   bash run_yolo_ablation_all.sh              # Submit YE-A + YE-C jako array
#   bash run_yolo_ablation_all.sh --check      # Tylko status
#   bash run_yolo_ablation_all.sh ye_a         # Single variant
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_YOLO_SCRIPT="$SCRIPT_DIR/run_yolo_ablation_variant.slurm"
SLURM_DETR_SCRIPT="$SCRIPT_DIR/../detr/run_ablation_variant.slurm"

YOLO_VARIANTS=(ye_a ye_c)
DETR_VARIANTS=(ye_b)

preflight() {
    echo "=== Preflight Checks ==="
    [[ -f "$SLURM_YOLO_SCRIPT" ]] && echo "  OK: YOLO slurm" || { echo "  FAIL: $SLURM_YOLO_SCRIPT"; exit 1; }
    [[ -f "$SLURM_DETR_SCRIPT" ]] && echo "  OK: DETR slurm" || echo "  WARN: $SLURM_DETR_SCRIPT (YE-B will fail)"
    # TODO: dodaj checks dla weights ep170 (YOLO i DETR), dataset yaml-i
}

submit_yolo() {
    local variant="$1"
    echo "Submitting YOLO variant: $variant"
    VARIANT="$variant" sbatch "$SLURM_YOLO_SCRIPT"
}

submit_detr() {
    local variant="$1"
    echo "Submitting DETR-target variant: $variant (uses ../detr/run_ablation_variant.slurm)"
    VARIANT="$variant" sbatch "$SLURM_DETR_SCRIPT"
}

check_status() {
    echo "=== Active YOLO ablation jobs ==="
    squeue -u "$USER" -n "yolo_abl_*" --format="%.10i %.20j %.8T %.10M %.20R" 2>/dev/null || true
}

# === MAIN ===
case "${1:-}" in
    --check)
        check_status
        ;;
    --help|-h)
        sed -n '1,15p' "$0"
        ;;
    "")
        preflight
        for v in "${YOLO_VARIANTS[@]}"; do submit_yolo "$v"; done
        for v in "${DETR_VARIANTS[@]}"; do submit_detr "$v"; done
        ;;
    *)
        # Specific variants on command line
        preflight
        for v in "$@"; do
            case "$v" in
                ye_a|ye_c) submit_yolo "$v" ;;
                ye_b) submit_detr "$v" ;;
                *) echo "Unknown variant: $v"; exit 1 ;;
            esac
        done
        ;;
esac
