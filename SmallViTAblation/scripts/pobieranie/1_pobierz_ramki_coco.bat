@echo off
rem ============================================================
rem  KROK 1 - same ramki narzedzi w formacie COCO
rem
rem  ROZMIAR: okolo 10 MB
rem  CO DAJE: wspolrzedne ramek dla 6 094 klatek, 12 klas
rem           (zrenica, rogowka i dziesiec narzedzi)
rem  PO CO:   zobaczyc konwencje adnotacji, zanim pobierzesz obrazy
rem ============================================================
setlocal
cd /d "%~dp0"
call _config.bat

echo.
echo === Pobieram plik z ramkami (okolo 10 MB) ===
huggingface-cli download %REPO% --repo-type dataset --local-dir "%DEST%" --include "2_Instance_Segmentation/annotations/coco_json/_annotations.coco.json"
if errorlevel 1 goto blad

echo.
echo === Gotowe ===
echo Plik: %DEST%\2_Instance_Segmentation\annotations\coco_json\_annotations.coco.json
echo Nastepny krok: 5_sprawdz_pobrane.bat  (podglad zawartosci, bez pobierania)
goto koniec

:blad
echo.
echo !!! Blad pobierania.
echo Najczestsza przyczyna: brak logowania albo niezaakceptowane warunki dostepu.
echo Uruchom najpierw 0_instalacja_i_login.bat

:koniec
echo.
pause
endlocal
