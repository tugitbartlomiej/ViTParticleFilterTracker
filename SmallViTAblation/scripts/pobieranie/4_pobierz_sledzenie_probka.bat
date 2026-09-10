@echo off
rem ============================================================
rem  KROK 4 - PROBKA sledzenia narzedzi, dwa najmniejsze klipy
rem
rem  ROZMIAR: okolo 100 MB (dwa archiwa)
rem  CO DAJE: ramki klatka po klatce, trwale numery obiektow
rem           i WSPOLRZEDNE KONCOWKI NARZEDZIA na kazdej klatce
rem  PO CO:   to jest wzorzec do sprawdzenia wlasnej propagacji ramek
rem
rem  UWAGA: caly katalog sledzenia to 170 archiwow i 45 GB.
rem         Ten skrypt bierze DWA najmniejsze: 42 MB i 56 MB.
rem         Wszystkie pliki maja w nazwie S1, czyli szpital Farabi,
rem         rozdzielczosc 720x480.
rem ============================================================
setlocal
cd /d "%~dp0"
call _config.bat

echo.
echo === Pobieram dwa najmniejsze klipy (okolo 100 MB) ===
hf download %REPO% --repo-type dataset --local-dir "%DEST%" --include "3_Object_Tracking/annotations/TR_0129_S1_P03.zip" --include "3_Object_Tracking/annotations/TR_0128_S1_P03.zip"
if errorlevel 1 goto blad

echo.
echo === Gotowe ===
echo Katalog: %DEST%\3_Object_Tracking\annotations
echo.
echo Zeby dobrac wiecej klipow, dopisz kolejne --include z nazwa pliku.
echo NIE uzywaj --include "3_Object_Tracking/*" - to 45 GB.
goto koniec

:blad
echo.
echo !!! Blad pobierania. Uruchom najpierw 0_instalacja_i_login.bat

:koniec
echo.
pause
endlocal
