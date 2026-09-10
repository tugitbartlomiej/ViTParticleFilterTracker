@echo off
rem ============================================================
rem  KROK 5 - podglad tego, co sie pobralo
rem
rem  ROZMIAR: nic nie pobiera, czyta tylko plik z dysku
rem  CO POKAZUJE: liczbe obrazow i adnotacji, nazwy klas,
rem               rozdzielczosci, rozmiar ramek, obecnosc masek
rem  PO CO:   sprawdzic, czy ich konwencja pasuje do naszej,
rem           ZANIM pobierzesz cokolwiek wiekszego
rem ============================================================
setlocal
cd /d "%~dp0"
call _config.bat

echo.
py -3.11 sprawdz_coco.py "%DEST%"

echo.
pause
endlocal
