@echo off
REM ============================================================
REM  Pobiera PDF-y z adresow rozwiazanych przez primo_otworz_partie.bat
REM  (plik primo\rozwiazane.json) prosto do PDF_all.
REM
REM  Adresy zza proxy PW wymagaja ZALOGOWANEJ sesji. Podaj ja plikiem
REM  primo\ciasteczka.txt w tym zbiorze - wzor i instrukcja sa w
REM  F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\primo\ciasteczka.przyklad.txt
REM
REM  Uzycie:
REM    dwuklik                    - pobierz wszystko, co da rade
REM    primo_pobierz.bat sucho    - tylko pokaz plan
REM    primo_pobierz.bat ponow    - tylko to, co ostatnio padlo
REM    primo_pobierz.bat otwarte  - pomin adresy zza proxy
REM
REM  Kod lezy w: F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\primo
REM ============================================================
setlocal
set PYTHONUTF8=1
set "ZBIOR=%~dp0..\.."

set "TRYB="
if /i "%~1"=="sucho"   set "TRYB=--sucho"
if /i "%~1"=="ponow"   set "TRYB=--ponow"
if /i "%~1"=="otwarte" set "TRYB=--otwarte"

if not exist "%ZBIOR%\primo\rozwiazane.json" (
  echo Brak primo\rozwiazane.json - uruchom najpierw primo_otworz_partie.bat,
  echo zeby zamienic wyszukiwania Primo na adresy pelnego tekstu.
  echo.
  pause
  exit /b 1
)

if not exist "%ZBIOR%\primo\ciasteczka.txt" (
  echo UWAGA: brak primo\ciasteczka.txt - pobiora sie tylko adresy otwarte
  echo        ^(arXiv, Unpaywall, IOP^). Instrukcja:
  echo        F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\primo\ciasteczka.przyklad.txt
  echo.
)

py -3.11 "F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\primo\pobierz_pdfy_primo.py" --zbior "%ZBIOR%" %TRYB%

echo.
echo === Gotowe. Odswiez kolejke, zeby odsiac pobrane: ===
echo   odswiez_kolejke.bat
echo.
pause
endlocal
