@echo off
rem ============================================================
rem  KROK 6 - CALY katalog 2_Instance_Segmentation
rem
rem  ROZMIAR: okolo 9 GB
rem     annotations/coco_json    10 MB ramki + 448 MB klatki
rem     annotations/yolo_txt     te same dane jako pliki tekstowe
rem     annotations/labels.zip   maski
rem     videos/                  4 archiwa po okolo 1,9 GB = 7,6 GB
rem
rem  CO DAJE: 6 094 klatki, 20 833 adnotacje, 12 klas
rem           (rogowka i zrenica na kazdym obrazie, plus dziesiec narzedzi)
rem           3 972 obrazy w 720x480, 2 122 w 1920x1080
rem ============================================================
setlocal
cd /d "%~dp0"
call _config.bat

echo.
echo === Pobieram caly katalog 2_Instance_Segmentation (okolo 9 GB) ===
echo To potrwa. Przerwanie nie szkodzi - kolejne uruchomienie dociagnie reszte.
echo.
hf download %REPO% --repo-type dataset --local-dir "%DEST%" --include "2_Instance_Segmentation/*"
if errorlevel 1 goto blad

echo.
echo === Gotowe ===
echo Katalog: %DEST%\2_Instance_Segmentation
goto koniec

:blad
echo.
echo !!! Blad pobierania. Sprawdz logowanie: hf auth whoami

:koniec
echo.
pause
endlocal
