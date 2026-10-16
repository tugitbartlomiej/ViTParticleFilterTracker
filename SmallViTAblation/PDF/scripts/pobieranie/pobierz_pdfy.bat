@echo off
REM ============================================================
REM  Pobieranie PDF-ow z list w linki\ tego zbioru.
REM
REM  Ustawienia: config\config.json (obok tego pliku).
REM    PDF-y ida do:      F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\PDF\pulapki_optyczne
REM    nieudane linki do: nieudane\nieudane_linki.html (klikalny raport)
REM
REM  Uruchomienie: dwuklik albo w konsoli: pobierz_pdfy.bat
REM  Argumenty mozna dopisac:
REM    pobierz_pdfy.bat --watki 12
REM    pobierz_pdfy.bat --sucho
REM    pobierz_pdfy.bat linki\moja-lista.txt
REM
REM  Kod lezy w: F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader
REM ============================================================
setlocal
set PYTHONUTF8=1
py -3.11 "F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\pobierz_pdfy.py" --zbior "%~dp0..\.." %*
set KOD=%ERRORLEVEL%

echo.
if not "%KOD%"=="0" (
  echo !!! Skrypt zakonczyl sie kodem %KOD% !!!
) else (
  echo === Zakonczono. PDF-y: %~dp0..\..\pulapki_optyczne ===
)
echo     Raport nieudanych: "%~dp0..\..\nieudane\nieudane_linki.html"
echo     Pelny log:         "%~dp0..\..\pobieranie.log"
echo.
echo     Aby ponowic same nieudane: pobierz_nieudane.bat
pause
endlocal
exit /b %KOD%
