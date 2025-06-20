@echo off
echo ===================================================
echo Background Frame Extraction for DETR Training
echo ===================================================

cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo Running background frame extraction...
python extract_background_frames.py --video_dir "E:\Cataract\videos\micro" --max_frames 1000 --clean_temp

echo.
echo ===================================================
echo Process completed!
echo ===================================================
echo.
echo Check the following directories:
echo - background_frames\          : Selected background frames
echo - detr_background_dataset\    : DETR-ready dataset
echo.

pause