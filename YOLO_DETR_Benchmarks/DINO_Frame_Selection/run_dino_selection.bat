@echo off
echo ===================================================
echo DINO-Based Intelligent Frame Selection for DETR
echo ===================================================

cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo ===================================================
echo Step 1: DINO Frame Selection
echo ===================================================
echo Running DINO-based frame selection...
echo.

py -3.11 dino_frame_selector.py ^
    --video_dir "E:\Cataract\videos\micro\one_video" ^
    --output_dir "dino_selection_results" ^
    --frame_interval 60 ^
    --dino_model "dino_vits16" ^
    --n_clusters 20 ^
    --frames_per_cluster 5 ^
    --selection_method "centroid" ^
    --create_detr_dataset ^
    --max_images 1000

if %errorlevel% neq 0 (
    echo Error in DINO frame selection!
    pause
    exit /b 1
)

echo.
echo ===================================================
echo Step 2: Integrated Dataset Creation
echo ===================================================
echo Creating integrated dataset with background frames...
echo.

py -3.11 integrated_detr_dataset_creator.py ^
    --dino_frames "dino_selection_results\clustering_results\selected_frames" ^
    --background_frames "..\Background_Extraction\background_frames" ^
    --output_dir "complete_detr_dataset" ^
    --balance_ratio 0.7 ^
    --train_ratio 0.8

if %errorlevel% neq 0 (
    echo Error in integrated dataset creation!
    pause
    exit /b 1
)

echo.
echo ===================================================
echo DINO Frame Selection Completed Successfully!
echo ===================================================
echo.
echo Results:
echo - DINO selection: dino_selection_results\
echo - Selected frames: dino_selection_results\clustering_results\selected_frames\
echo - Complete dataset: complete_detr_dataset\
echo - Visualizations: dino_selection_results\clustering_results\
echo.
echo Next steps:
echo 1. Review selected frames and clustering results
echo 2. Add tool annotations to selected frames
echo 3. Train DETR model with integrated dataset
echo 4. Compare with original DETR performance
echo.

pause