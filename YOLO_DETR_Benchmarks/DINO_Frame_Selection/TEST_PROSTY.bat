@echo off
echo ===================================================
echo PROSTY TEST SYSTEMU DINO
echo ===================================================

cd /d "%~dp0"

echo Sprawdzanie Python...
py -3.11 --version
if %errorlevel% neq 0 (
    echo BŁĄD: Python 3.11 nie jest dostępny!
    pause
    exit /b 1
)

echo.
echo Sprawdzanie bibliotek...
py -3.11 -c "import torch; print('PyTorch:', torch.__version__)"
if %errorlevel% neq 0 (
    echo Instalowanie PyTorch...
    py -3.11 -m pip install torch torchvision
)

echo.
echo Sprawdzanie video...
if not exist "E:\Cataract\videos\micro\one_video" (
    echo BŁĄD: Folder z video nie istnieje!
    echo Sprawdź: E:\Cataract\videos\micro\one_video
    pause
    exit /b 1
)

echo.
echo ===================================================
echo URUCHAMIANIE MINI TESTU
echo ===================================================

py -3.11 dino_frame_selector.py ^
    --video_dir "E:\Cataract\videos\micro\one_video" ^
    --output_dir "mini_test" ^
    --frame_interval 180 ^
    --n_clusters 5 ^
    --frames_per_cluster 2 ^
    --max_images 50

echo.
echo ===================================================
echo SPRAWDZANIE WYNIKÓW
echo ===================================================

if exist "mini_test\clustering_results\selected_frames" (
    echo ✅ SUKCES! Znaleziono wybrane ramki:
    dir "mini_test\clustering_results\selected_frames"
    echo.
    echo Pełne wyniki w folderze: mini_test\
) else (
    echo ❌ BŁĄD: Nie znaleziono wybranych ramek
)

echo.
pause