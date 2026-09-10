@echo off
rem ============================================================
rem  KROK 3 - te same adnotacje w formacie YOLO (opcjonalne)
rem
rem  ROZMIAR: maly, to pliki tekstowe
rem  CO DAJE: ten sam zbior co w kroku 1, ale jako pliki .txt,
rem           po jednym na obraz
rem  KIEDY:   gdy trenujesz modelem rodziny YOLO i nie chcesz
rem           konwertowac z COCO samodzielnie
rem ============================================================
setlocal
cd /d "%~dp0"
call _config.bat

echo.
echo === Pobieram adnotacje w formacie YOLO ===
huggingface-cli download %REPO% --repo-type dataset --local-dir "%DEST%" --include "2_Instance_Segmentation/annotations/yolo_txt/*"
if errorlevel 1 goto blad

echo.
echo === Gotowe ===
echo Katalog: %DEST%\2_Instance_Segmentation\annotations\yolo_txt
goto koniec

:blad
echo.
echo !!! Blad pobierania. Uruchom najpierw 0_instalacja_i_login.bat

:koniec
echo.
pause
endlocal
