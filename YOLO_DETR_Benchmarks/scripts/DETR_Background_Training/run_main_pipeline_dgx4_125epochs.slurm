#!/bin/bash
###############################################################################
#  DETR BACKGROUND TRAINING MAIN PIPELINE – 125 epochs | DGX-4
###############################################################################
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --nodelist=dgx-4
#SBATCH --gres=gpu:2
#SBATCH --mem=150G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=detr_main_pipeline_125e
#SBATCH --chdir=~/ViT/YOLO_DETR_Benchmarks/scripts/DETR_Background_Training
#SBATCH --output=logs/main_pipeline_125epochs_dgx4_%j.log
#SBATCH --error=logs/main_pipeline_125epochs_dgx4_%j.log
#SBATCH --open-mode=append

set -euo pipefail
echo "=== DETR MAIN PIPELINE 125 EPOCHS on $(hostname) | $(date) ==="
echo "=== DINO → YOLO → DETR Mixed Classification → Gentle Training ==="

# ───────────────────── TMPDIR (NVMe węzła) ───────────────────────────────────
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"
echo "TMPDIR → $TMPDIR"

# ───────────────────── Conda env ──────────────────────────────────────────────
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310
echo "[Env] Aktywowane środowisko: $(conda env list | grep '\*')"

# ───────────────────── Setup model paths ──────────────────────────────────────
echo "📦 Setting up model paths..."

# Create models directory structure if needed
mkdir -p ../../models/YOLO/yolo_inference_model_final
mkdir -p ../../models/DETR

# Copy/link YOLO model to expected relative path
YOLO_SOURCE="~/models/YOLO/yolo_inference_model_final/yolo_inference_model.pt"
YOLO_TARGET="../../models/YOLO/yolo_inference_model_final/yolo_inference_model.pt"

if [[ -f "$YOLO_SOURCE" ]]; then
    if [[ ! -f "$YOLO_TARGET" ]]; then
        ln -sf "$YOLO_SOURCE" "$YOLO_TARGET"
        echo "✅ YOLO model linked: $YOLO_SOURCE → $YOLO_TARGET"
    else
        echo "✅ YOLO model already exists: $YOLO_TARGET"
    fi
else
    echo "❌ YOLO model not found: $YOLO_SOURCE"
    exit 1
fi

# Copy/link DETR inference model to expected relative path  
DETR_SOURCE="~/models/DETR/detr_inference_model.pth"
DETR_TARGET="../../models/DETR/detr_inference_model.pth"

if [[ -f "$DETR_SOURCE" ]]; then
    if [[ ! -f "$DETR_TARGET" ]]; then
        ln -sf "$DETR_SOURCE" "$DETR_TARGET"
        echo "✅ DETR inference model linked: $DETR_SOURCE → $DETR_TARGET"
    else
        echo "✅ DETR inference model already exists: $DETR_TARGET"
    fi
else
    echo "❌ DETR inference model not found: $DETR_SOURCE"
    exit 1
fi

# ───────────────────── Verify checkpoint ──────────────────────────────────────
CHECKPOINT_PATH="~/DETR/Checkpoints/checkpoint_epoch_125.pth"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "❌ Checkpoint not found: $CHECKPOINT_PATH"
    echo "📋 Available checkpoints:"
    ls -la ~/DETR/Checkpoints/checkpoint_epoch_*.pth || echo "No checkpoints found"
    exit 1
fi
echo "✅ Training checkpoint ready: $CHECKPOINT_PATH"

# ───────────────────── Config file setup ──────────────────────────────────────
CONFIG_FILE="pipeline_config_eden_125epochs.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo "📋 Available config files:"
    ls -la pipeline_config*.yaml
    exit 1
fi
echo "✅ Pipeline configuration: $CONFIG_FILE"

# ───────────────────── Create logs directory ──────────────────────────────────
mkdir -p logs
mkdir -p pipeline_output_eden_125epochs

# ───────────────────── Launch Main Pipeline ───────────────────────────────────
echo "🚀 Starting DETR Background Training Main Pipeline..."
echo "📊 Configuration: 125 epochs, 4 GPUs, gentle learning rate 1e-6"
echo "📁 Output directory: pipeline_output_eden_125epochs"
echo "🎯 Checkpoint: $CHECKPOINT_PATH"
echo "🔄 Pipeline stages:"
echo "   1. DINO Frame Extraction from videos"
echo "   2. YOLO + DETR Mixed Classification"  
echo "   3. Dataset Mixing (tooltip 70% + background 30%)"
echo "   4. Model Preparation"
echo "   5. Mixed Gentle Training (125 epochs)"
echo "   6. Results Validation"

# Launch main pipeline
python3 main_pipeline.py --config "$CONFIG_FILE"

PIPELINE_EXIT_CODE=$?

echo -e "\n✓ Main Pipeline completed with exit code: $PIPELINE_EXIT_CODE"

if [[ $PIPELINE_EXIT_CODE -eq 0 ]]; then
    echo "✅ DETR Background Training Pipeline successful!"
    echo "📁 Check results in: pipeline_output_eden_125epochs/"
    echo "📊 Training logs in: logs/"
else
    echo "❌ Pipeline failed with code: $PIPELINE_EXIT_CODE"
    echo "📋 Check logs for details: logs/main_pipeline_125epochs_dgx4_${SLURM_JOB_ID}.log"
fi

# ───────────────────── Cleanup ─────────────────────────────────────────────────
echo "🧹 Cleaning up TMPDIR..."
rm -rf "$TMPDIR"

echo "✅ Job completed: $(date)"
exit $PIPELINE_EXIT_CODE