@echo off
REM ============================================================
REM  Przenosi swieze PDF-y z katalogu Pobrane do pulapki_optyczne tego zbioru
REM  i nadaje im nazwy w konwencji downloadera, zeby nie pobral ich
REM  drugi raz. Uzywaj po recznym sciaganiu przez Primo w przegladarce.
REM
REM  Uzycie:
REM    dwuklik                        - tylko PLAN, nic nie rusza
REM    primo_przenies_pobrane.bat tak - faktycznie przenies
REM
REM  Kod lezy w: F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\primo
REM ============================================================
setlocal
set PYTHONUTF8=1
set "ZBIOR=%~dp0..\.."

set "TRYB="
if /i "%~1"=="tak" set "TRYB=--przenies"

py -3.11 "F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\primo\przenies_pobrane.py" --zbior "%ZBIOR%" %TRYB%

echo.
if not defined TRYB echo To byl tylko plan. Aby przeniesc: primo_przenies_pobrane.bat tak
echo.
pause
endlocal
