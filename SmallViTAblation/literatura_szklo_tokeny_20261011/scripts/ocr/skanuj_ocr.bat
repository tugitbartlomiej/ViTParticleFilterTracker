@echo off
REM ============================================================
REM  MinerU - OCR wszystkich PDF-ow tego zbioru, jeden po drugim.
REM
REM  Wejscie:  folder_pdf z config\config.json (rekurencyjnie)
REM  Wyjscie:  Ocr\<nazwa PDF>\auto\<nazwa PDF>.md
REM  Jezyk:    klucz "ocr_jezyk" w config\config.json (domyslnie ch)
REM
REM  KONTYNUACJA: pomija pliki, ktore juz maja wynik .md, wiec przerwany
REM  przebieg wznawia sie bez strat.
REM  NIE generuje: _layout.pdf, _span.pdf, _origin.pdf, _model.json, _middle.json.
REM
REM  Uruchomienie: dwuklik albo w konsoli: skanuj_ocr.bat
REM  Gadatliwie (pasek z etapami MinerU): skanuj_ocr.bat -v
REM
REM  Kod lezy w: F:\Studia\Articles\Zbiory\_Narzedzia\OcrMinerU
REM  Srodowisko i model: F:\OCR\MinerU
REM ============================================================
setlocal
set MINERU_ENV=F:\OCR\MinerU
set HF_HOME=F:\HuggingFace
set PYTHONUTF8=1
set MINERU_MODEL_SOURCE=huggingface
set MINERU_DEVICE_MODE=cuda

set "GADATLIWIE="
set "SKAN_ARGS=--krotko"
if /i "%~1"=="-v" goto :gadatliwie
if /i "%~1"=="--verbose" goto :gadatliwie
goto :start
:gadatliwie
set "GADATLIWIE=--verbose"
set "SKAN_ARGS="
:start

echo === KROK 1/2: skan postepu (odswiezenie manifestu) ===
"%MINERU_ENV%\.venv\Scripts\python.exe" "F:\Studia\Articles\Zbiory\_Narzedzia\OcrMinerU\postep_ocr.py" --zbior "%~dp0..\.." %SKAN_ARGS%
if errorlevel 1 (
    echo.
    echo BLAD: skan postepu nie powiodl sie - przerywam przed OCR.
    pause
    exit /b 1
)

echo.
echo === KROK 2/2: OCR MinerU ===
"%MINERU_ENV%\.venv\Scripts\python.exe" "F:\Studia\Articles\Zbiory\_Narzedzia\OcrMinerU\skanuj_ocr.py" --zbior "%~dp0..\.." %GADATLIWIE%

echo.
echo === Zakonczono. Wyniki: %~dp0..\..\Ocr\raw ===
pause
endlocal
