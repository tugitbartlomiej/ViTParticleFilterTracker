@echo off
rem ============================================================
rem  KROK 2 - klatki, do ktorych odnosza sie ramki z kroku 1
rem
rem  ROZMIAR: okolo 448 MB (archiwum) + tyle samo po rozpakowaniu
rem  CO DAJE: 6 094 obrazy; czesc w 720x480 (szpital Farabi),
rem           czesc w 1920x1080 (szpital Noor)
rem ============================================================
setlocal
cd /d "%~dp0"
call _config.bat

set "SEG=%DEST%\2_Instance_Segmentation\annotations\coco_json"

echo.
echo === Pobieram klatki (okolo 448 MB) ===
huggingface-cli download %REPO% --repo-type dataset --local-dir "%DEST%" --include "2_Instance_Segmentation/annotations/coco_json/images/images.zip"
if errorlevel 1 goto blad

echo.
echo === Rozpakowuje ===
if not exist "%SEG%\images\extracted" mkdir "%SEG%\images\extracted"
tar -xf "%SEG%\images\images.zip" -C "%SEG%\images\extracted"
if errorlevel 1 goto bladrozpak

echo.
echo === Gotowe ===
echo Obrazy: %SEG%\images\extracted
echo.
echo Archiwum images.zip mozesz skasowac po sprawdzeniu, zeby odzyskac 448 MB.
goto koniec

:bladrozpak
echo.
echo !!! Pobralo sie, ale rozpakowanie nie wyszlo.
echo Rozpakuj recznie: %SEG%\images\images.zip
goto koniec

:blad
echo.
echo !!! Blad pobierania. Uruchom najpierw 0_instalacja_i_login.bat

:koniec
echo.
pause
endlocal
