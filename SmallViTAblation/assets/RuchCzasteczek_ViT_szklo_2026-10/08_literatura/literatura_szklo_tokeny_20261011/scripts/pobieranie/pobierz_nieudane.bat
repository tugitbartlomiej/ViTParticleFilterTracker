@echo off
REM ============================================================
REM  Ponowienie tego, co wczesniej padlo (nieudane\nieudane_stan.json).
REM  Kod lezy w: F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader
REM ============================================================
setlocal
set PYTHONUTF8=1
py -3.11 "F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\pobierz_pdfy.py" --zbior "%~dp0..\.." --nieudane %*
set KOD=%ERRORLEVEL%
echo.
if not "%KOD%"=="0" echo !!! Skrypt zakonczyl sie kodem %KOD% !!!
pause
endlocal
exit /b %KOD%
