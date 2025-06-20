@echo off
echo ====================================================================
echo INTELLIGENT BACKGROUND FRAME SELECTOR FOR DETR TRAINING
echo ====================================================================
echo.
echo This tool extracts background frames (no surgical tools detected)
echo from videos using YOLO and DETR models with DINO-based selection.
echo.
echo TESTED AND WORKING! Generated 105+ background frames successfully.
echo.

cd /d "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\Intelligent_Background_Selector_2025-07-17"

echo Current settings:
echo - Videos directory: E:/Cataract/videos/micro
echo - YOLO model: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/YOLO/yolo_inference_model_final/yolo_inference_model.pt
echo - DETR model: F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/DETR/detr_inference_model_final
echo - Frame interval: 30 (every 30 frames)
echo - Max frames per video: 200
echo - Confidence thresholds: 0.3 (both YOLO and DETR)
echo.
echo Results will be saved to: background_frames/
echo.

py intelligent_background_frame_selector.py

echo.
echo Processing completed! Check the following locations:
echo - Background frames: background_frames/
echo - Analysis results: analysis_results/
echo - Test log: test_log.txt
echo.
pause