#!/bin/bash
# TensorBoard launch script for YOLO 20k finetune monitoring

echo "============================================"
echo "  YOLO 20k Finetune - TensorBoard Launcher"
echo "============================================"

# Default port
PORT=${1:-6007}

# Log directories to monitor
PROJECT_DIR=/mnt/evafs/faculty/home/bpiotrowski/YOLO_20k
LOGDIR="$PROJECT_DIR/train_out_20k"

# Alternative: monitor multiple runs
# LOGDIR="$PROJECT_DIR/train_out_20k,$PROJECT_DIR/exp"

echo ""
echo "Configuration:"
echo "  Port: $PORT"
echo "  Log directory: $LOGDIR"
echo ""

# Check if directory exists
if [[ ! -d "$LOGDIR" ]]; then
    echo "WARNING: Log directory does not exist yet: $LOGDIR"
    echo "TensorBoard will wait for logs to appear..."
    mkdir -p "$LOGDIR"
fi

# Activate conda environment
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

echo "Starting TensorBoard..."
echo "Access via: http://localhost:$PORT"
echo ""
echo "If running on Eden, use SSH tunnel:"
echo "  ssh -L $PORT:localhost:$PORT bpiotrowski@eden.ue.katowice.pl"
echo ""

# Launch TensorBoard
tensorboard --logdir="$LOGDIR" --port=$PORT --bind_all

# Alternative with more options:
# tensorboard \
#     --logdir="$LOGDIR" \
#     --port=$PORT \
#     --bind_all \
#     --reload_interval=30 \
#     --samples_per_plugin=images=100
