#!/bin/bash
###############################################################################
#  TensorBoard Launch Script for DETR Training Monitoring
#
#  Usage:
#    1. SSH to Eden: ssh eden
#    2. Run: bash launch_tensorboard.sh
#    3. Open in browser: http://localhost:6006
#
#  For remote access (from Windows):
#    ssh -L 6006:localhost:6006 eden
#    Then open: http://localhost:6006
###############################################################################

# Configuration
LOGDIR="/mnt/evafs/faculty/home/bpiotrowski/DETR/train_out_20k_finetune/logs"
PORT=6006

echo "============================================"
echo "  TensorBoard for DETR Training"
echo "============================================"
echo ""

# Activate conda environment
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# Check if log directory exists
if [ ! -d "$LOGDIR" ]; then
    echo "WARNING: Log directory not found: $LOGDIR"
    echo "Creating directory..."
    mkdir -p "$LOGDIR"
    echo ""
    echo "Note: Logs will appear here after training starts."
    echo ""
fi

# Show existing logs
echo "Log directory: $LOGDIR"
if [ -d "$LOGDIR" ]; then
    echo "Contents:"
    ls -la "$LOGDIR" 2>/dev/null | head -10
fi
echo ""

# Check if port is already in use
if netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
    echo "WARNING: Port $PORT is already in use!"
    echo "Trying to find another port..."
    PORT=$((PORT + 1))
fi

echo "Starting TensorBoard on port $PORT..."
echo ""
echo "============================================"
echo "  Access TensorBoard:"
echo "  - Local:  http://localhost:$PORT"
echo "  - Remote: ssh -L $PORT:localhost:$PORT eden"
echo "             then open http://localhost:$PORT"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""

# Launch TensorBoard
tensorboard --logdir="$LOGDIR" --port=$PORT --bind_all

