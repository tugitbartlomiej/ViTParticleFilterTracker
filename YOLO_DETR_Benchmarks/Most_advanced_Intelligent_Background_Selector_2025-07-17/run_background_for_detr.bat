@echo off
echo =========================================================
echo INTELLIGENT BACKGROUND FRAME SELECTOR FOR DETR TRAINING
echo =========================================================
echo.
echo This script will:
echo - Process ALL videos from E:/Cataract/videos/micro
echo - Extract maximally informative background frames
echo - Create DETR-ready dataset in Datasets/Detr/Background
echo - Use optimized parameters for best results
echo.
echo Expected processing time: 30-120 minutes (depending on hardware)
echo.
pause

cd /d "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\Most_advanced_Intelligent_Background_Selector_2025-07-17"

echo Starting background frame selection...
python intelligent_background_frame_selector.py

echo.
echo =========================================================
echo BACKGROUND FRAME SELECTION COMPLETED!
echo =========================================================
echo.
echo Results saved to:
echo F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\Datasets\Detr\Background
echo.
echo Next step: Use this dataset for DETR fine-tuning with detr_train_optimized.py
echo.
pause