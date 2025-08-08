@echo off
echo ============================================================
echo    DETR Local Training - With Checkpoint
echo ============================================================

REM Activate conda environment (adjust path as needed)
call conda activate base

REM Set checkpoint path 
set CHECKPOINT_PATH="..\..\..\..\models\DETR\checkpoint_epoch_100.pth"

REM Check if checkpoint exists
if not exist %CHECKPOINT_PATH% (
    echo Warning: Checkpoint not found at %CHECKPOINT_PATH%
    echo Please download checkpoint from cluster first
    echo Continuing with pretrained model...
    set CHECKPOINT_PATH=""
)

REM Create output directories
if not exist "checkpoints_local" mkdir checkpoints_local
if not exist "output_local" mkdir output_local

echo.
echo Starting DETR training with checkpoint...
echo.

REM Run training with checkpoint
python detr_local_train.py ^
    --checkpoint_path %CHECKPOINT_PATH% ^
    --background_images_dir "..\..\..\..\Datasets\Detr\Background\train" ^
    --background_annotations_path "..\..\..\..\Datasets\Detr\Background\annotations\train_annotations.json" ^
    --checkpoint_dir ".\checkpoints_local" ^
    --output_dir ".\output_local" ^
    --epochs 10 ^
    --batch_size 2 ^
    --num_workers 0 ^
    --gradient_accumulation_steps 4 ^
    --lr 2e-5 ^
    --lr_backbone 2e-6 ^
    --save_interval 2 ^
    --patience 5 ^
    --augment ^
    --use_amp ^
    --resume_training

echo.
echo Training completed!
echo Check output_local folder for results
pause