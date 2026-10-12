@echo off
REM ============================================================
REM  Zbieranie linkow do kolejki tego zbioru.
REM
REM  Nowa lista tematyczna:
REM    zbierz_linki.bat --nowa "Augmentacja obrazow" --po-co "Po co ten plik."
REM  Dopisanie linkow do istniejacej listy:
REM    zbierz_linki.bat --do augmentacja-obrazow-2026-09-11 --sekcja "MIXUP" ^
REM        --link "https://doi.org/10.48550/arXiv.1710.09412 # Zhang 2018, mixup"
REM  Co juz jest:
REM    zbierz_linki.bat --pokaz
REM
REM  Kod lezy w: F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader
REM ============================================================
setlocal
set PYTHONUTF8=1
py -3.11 "F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\nowa_lista.py" --zbior "%~dp0..\.." %*
set KOD=%ERRORLEVEL%
echo.
if not "%KOD%"=="0" echo !!! Skrypt zakonczyl sie kodem %KOD% !!!
pause
endlocal
exit /b %KOD%
