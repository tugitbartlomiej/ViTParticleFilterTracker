@echo off
echo ============================================================
echo    DETR Quick Start - Background Fine-tuning
echo ============================================================

REM Activate conda environment (adjust path as needed)
call conda activate base

echo Starting DETR training with your checkpoint...
echo Checkpoint: checkpoint_epoch_100.pth
echo Images: 55 background images
echo.

REM Quick training with good defaults
python detr_local_train.py ^
    --epochs 5 ^
    --batch_size 1 ^
    --gradient_accumulation_steps 8 ^
    --lr 1e-5 ^
    --lr_backbone 1e-6 ^
    --save_interval 2 ^
    --patience 3 ^
    --augment ^
    --use_amp

echo.
echo ============================================================
echo Training completed! Check these folders:
echo - checkpoints_local/ - saved models
echo - output_local/logs/ - training logs
echo.
echo To view training progress:
echo tensorboard --logdir output_local/logs
echo ============================================================
pause