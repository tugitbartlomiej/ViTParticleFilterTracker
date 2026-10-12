@echo off
REM ============================================================
REM  Prace zza paywalla PRZEZ EDGE - caly obieg w jednym wywolaniu.
REM
REM  Proxy biblioteki PW loguje przez konto Microsoft, a Edge od wersji 127
REM  szyfruje ciasteczka kluczem wymagajacym uprawnien administratora. Sesji
REM  nie da sie wiec oddac skryptowi - artykul musi otworzyc sama
REM  przegladarka, w ktorej jestes zalogowany.
REM
REM  Ten plik po kolei:
REM    1. sprawdza, czy Edge ZAPISUJE pliki PDF (a nie wyswietla w karcie),
REM    2. odczytuje z ustawien Edge, dokad pobiera,
REM    3. otwiera partie kart z gotowymi PDF-ami,
REM    4. czeka, az pobieranie ucichnie,
REM    5. nadaje plikom nazwy w konwencji downloadera i odsiewa duplikaty.
REM
REM  NAJPIERW uruchom odswiez_kolejke.bat - to on buduje kolejke Primo.
REM
REM  Uzycie:
REM    dwuklik                        - caly obieg, pierwsze 25 pozycji
REM    przez_edge.bat sprawdz         - sama diagnoza, niczego nie otwiera
REM    przez_edge.bat ustaw           - wlacz zapisywanie PDF (Edge ZAMKNIETY)
REM    przez_edge.bat 6 10            - od pozycji 6, dziesiec kart
REM    przez_edge.bat posprzataj      - bez otwierania, sam porzadek nazw
REM
REM  Kod lezy w: F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\primo
REM ============================================================
setlocal
set PYTHONUTF8=1
set "ZBIOR=%~dp0..\.."
set "SKRYPT=F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\primo\przez_edge.py"

if /i "%~1"=="sprawdz"    goto sprawdz
if /i "%~1"=="ustaw"      goto ustaw
if /i "%~1"=="posprzataj" goto posprzataj

set "OD=%~1"
set "ILE=%~2"
if "%OD%"==""  set "OD=1"
if "%ILE%"=="" set "ILE=25"
py -3.11 "%SKRYPT%" --zbior "%ZBIOR%" --od %OD% --ile %ILE%
goto koniec

:sprawdz
py -3.11 "%SKRYPT%" --zbior "%ZBIOR%" --sprawdz
goto koniec

:ustaw
py -3.11 "%SKRYPT%" --zbior "%ZBIOR%" --ustaw-pobieranie
goto koniec

:posprzataj
py -3.11 "%SKRYPT%" --zbior "%ZBIOR%" --bez-otwierania
goto koniec

:koniec
echo.
pause
endlocal
