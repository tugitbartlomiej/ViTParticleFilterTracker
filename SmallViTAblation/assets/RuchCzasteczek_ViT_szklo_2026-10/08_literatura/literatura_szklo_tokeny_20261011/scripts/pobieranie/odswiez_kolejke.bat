@echo off
REM ============================================================
REM  Pelny obieg biblioteki tego zbioru - jeden przebieg.
REM
REM  1) pobierz_pdfy.py          - bierze automatem, co sie da
REM  2) odswiez_nieudane.py      - wykresla z raportu to, co juz lezy w PDF_all
REM  3) sprawdz_nieudane_pdfy.py - odsiewa to, co juz jest w PDF_all
REM  4) primo_z_nieudanych.py    - buduje kolejke Primo i listy DOI
REM  5) sprawdz_kolejke.py       - znaczy w kolejce to, co juz mam
REM
REM  Krok recznego pobierania jest MIEDZY 5 a kolejnym uruchomieniem tego
REM  pliku: pobierasz w przegladarce przez Primo, wrzucasz PDF-y do
REM  F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\literatura_szklo_tokeny_20261011\PDF_all, odpalasz to ponownie. Kolejka skurczy sie o to, co juz masz.
REM
REM  Kod lezy w: F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader
REM ============================================================
setlocal
set PYTHONUTF8=1
set ZBIOR=%~dp0..\..
set NARZ=F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader

echo.
echo === KROK 1/5: pobieranie automatem ===
py -3.11 "%NARZ%\pobierz_pdfy.py" --zbior "%ZBIOR%"
if errorlevel 1 goto blad

echo.
echo === KROK 2/5: odsiew raportu nieudanych wobec PDF_all ===
py -3.11 "%NARZ%\odswiez_nieudane.py" --zbior "%ZBIOR%"
if errorlevel 1 goto blad

echo.
echo === KROK 3/5: weryfikacja wobec PDF_all ===
py -3.11 "%NARZ%\sprawdz_nieudane_pdfy.py" --zbior "%ZBIOR%"
if errorlevel 1 goto blad

echo.
echo === KROK 4/5: kolejka Primo + listy DOI ===
py -3.11 "%NARZ%\primo_z_nieudanych.py" --zbior "%ZBIOR%"
if errorlevel 1 goto blad

echo.
echo === KROK 5/5: odsiew kolejki + powtorki ===
py -3.11 "%NARZ%\sprawdz_kolejke.py" --zbior "%ZBIOR%"
if errorlevel 1 goto blad

echo.
echo === Gotowe. Dalej recznie: ===
echo   nieudane:  nieudane\nieudane_linki.html
echo   przeglad:  primo\primo_wyszukiwania.html
echo   listy DOI: doi\doi_nieudane.txt , doi\doi_nieudane.csv
echo   pobrane wrzucaj do: %~dp0..\..\PDF_all
goto koniec

:blad
echo.
echo !!! Krok zakonczyl sie bledem - przerwano. !!!

:koniec
echo.
pause
endlocal
