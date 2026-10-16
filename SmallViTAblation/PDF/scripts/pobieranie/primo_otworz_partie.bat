@echo off
REM ============================================================
REM  Primo (dostep instytucjonalny PW) - przechodzenie kolejki partiami.
REM
REM    1. otwiera partie kart z pelnym tekstem w przegladarce,
REM    2. czeka, az sie zaladuja, i pyta, czy pobrac je automatycznie,
REM    3. pobiera co sie da i mowi, co zostalo do recznego pobrania,
REM    4. proponuje kolejna partie - Enter i leci dalej.
REM
REM  Kazda karta to od razu PELNY TEKST pierwszego wyniku: PDF, a gdy go
REM  nie ma, HTML. Adresy bierze publiczne API Primo.
REM
REM  NAJPIERW uruchom odswiez_kolejke.bat - to on buduje kolejke
REM  (primo\otworz_partie.ps1) z tego, czego nie dalo sie wziac automatem.
REM
REM  Uzycie:
REM    dwuklik                       - zapyta o pozycje startowa i wielkosc partii
REM    primo_otworz_partie.bat 11 10 - od pozycji 11, dziesiec kart
REM    primo_otworz_partie.bat html  - tylko przeglad HTML
REM    primo_otworz_partie.bat bez 1 10        - bez ponownego skanu pulapki_optyczne
REM    primo_otworz_partie.bat wszystkie 1 10  - bez odsiewania, cala lista
REM    primo_otworz_partie.bat wyniki 1 10     - strony wynikow zamiast pelnych tekstow
REM    primo_otworz_partie.bat rozwiaz         - rozwiaz CALA kolejke, bez kart
REM    primo_otworz_partie.bat pdfy 1 25       - karty tylko z PDF-ami
REM
REM  Kod lezy w: F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader
REM ============================================================
setlocal EnableDelayedExpansion
set PYTHONUTF8=1
set "ZBIOR=%~dp0..\.."
set "KOLEJKA=%ZBIOR%\primo\otworz_partie.ps1"

if /i "%~1"=="html" goto html

if not exist "%KOLEJKA%" (
  echo Brak kolejki Primo: %KOLEJKA%
  echo Uruchom najpierw odswiez_kolejke.bat - zbuduje ja z raportu nieudanych.
  echo.
  pause
  exit /b 1
)

set "SPRAWDZAJ=1"
set "TRYB="
set "PETLA=1"
if /i "%~1"=="bez" (
  set "SPRAWDZAJ="
  shift
)
if /i "%~1"=="wszystkie" (
  set "SPRAWDZAJ="
  set "TRYB=-Wszystkie"
  shift
)
if /i "%~1"=="wyniki" (
  set "TRYB=%TRYB% -Wyszukiwarka"
  shift
)
if /i "%~1"=="rozwiaz" (
  set "TRYB=%TRYB% -TylkoRozwiaz"
  set "OD=1"
  set "ILE=999"
  set "PETLA="
  shift
  goto uruchom
)
if /i "%~1"=="pdfy" (
  set "TRYB=%TRYB% -TylkoPdf"
  shift
)

if not defined SPRAWDZAJ goto po_sprawdzeniu

echo.
echo === Sprawdzam, co juz jest w pulapki_optyczne ===
py -3.11 "F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\sprawdz_kolejke.py" --zbior "%ZBIOR%" --cicho
if errorlevel 1 echo UWAGA: sprawdzanie sie nie powiodlo - kolejka nie zostala odsiana.

:po_sprawdzeniu
set "OD=%~1"
set "ILE=%~2"
if not "%OD%"=="" goto uruchom

echo.
echo === Primo: otwieranie partii wyszukiwan ===
echo   (Enter = wartosc domyslna, wpisz H aby otworzyc przeglad HTML)
echo.
set /p "OD=Pozycja startowa [1]: "
if /i "%OD%"=="H" goto html
if "%OD%"=="" set "OD=1"
set /p "ILE=Ile kart naraz [10]: "

:uruchom
if "%ILE%"=="" set "ILE=10"

:partia
echo.
echo Uruchamiam: otworz_partie.ps1 -Od %OD% -Ile %ILE% %TRYB%
powershell -NoProfile -ExecutionPolicy Bypass -File "%KOLEJKA%" -Od %OD% -Ile %ILE% %TRYB%
if errorlevel 2 goto koniec
if errorlevel 1 goto blad
if not defined PETLA goto koniec

echo.
echo ------------------------------------------------------------
echo  Karty sie otwieraja. Poczekaj, az sie zaladuja.
echo ------------------------------------------------------------
set "ODP="
set /p "ODP=Pobrac te pozycje automatycznie? [Enter = tak, n = nie]: "
if /i "%ODP%"=="n" goto po_pobraniu

echo.
py -3.11 "F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\primo\pobierz_pdfy_primo.py" --zbior "%ZBIOR%" --partia

:po_pobraniu
set /a NASTEPNA=OD+ILE
echo.
echo ------------------------------------------------------------
set "ODP2="
set /p "ODP2=Otworzyc kolejna partie od pozycji %NASTEPNA%? [Enter = tak, n = koniec]: "
if /i "%ODP2%"=="n" goto koniec
set "OD=%NASTEPNA%"
goto partia

:html
echo Otwieram przeglad: %ZBIOR%\primo\primo_wyszukiwania.html
start "" "%ZBIOR%\primo\primo_wyszukiwania.html"
goto koniec

:blad
echo.
echo !!! PowerShell zwrocil blad - przerwano. !!!

:koniec
echo.
echo === Pobrane recznie PDF-y wrzucaj do: ===
echo   %ZBIOR%\pulapki_optyczne
echo Potem odswiez kolejke:  odswiez_kolejke.bat
echo.
pause
endlocal
