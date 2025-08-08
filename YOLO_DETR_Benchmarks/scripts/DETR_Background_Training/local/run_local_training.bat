@echo off
echo ============================================================
echo    DETR Local Training - Background Fine-tuning
echo ============================================================

REM Activate conda environment (adjust path as needed)
call conda activate base

REM Check if CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
if %errorlevel% neq 0 (
    echo Error: PyTorch not installed properly
    pause
    exit /b 1
)

REM Create output directories
if not exist "checkpoints_local" mkdir checkpoints_local
if not exist "output_local" mkdir output_local

echo.
echo Starting DETR training...
echo.

REM Run training with basic settings
python detr_local_train.py ^
    --background_images_dir "..\..\..\..\Datasets\Detr\Background\train" ^
    --background_annotations_path "..\..\..\..\Datasets\Detr\Background\annotations\train_annotations.json" ^
    --checkpoint_dir ".\checkpoints_local" ^
    --output_dir ".\output_local" ^
    --epochs 5 ^
    --batch_size 1 ^
    --num_workers 0 ^
    --gradient_accumulation_steps 8 ^
    --lr 5e-5 ^
    --lr_backbone 5e-6 ^
    --save_interval 2 ^
    --patience 3 ^
    --augment ^
    --use_amp

echo.
echo Training completed!
echo Check output_local folder for results
pause