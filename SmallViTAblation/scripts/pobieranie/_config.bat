@echo off
rem ============================================================
rem  Wspolna konfiguracja. Nie uruchamiaj tego pliku bezposrednio -
rem  jest wolany przez pozostale skrypty.
rem ============================================================

set "REPO=mjahmadi/Cataract-LMM"
set "DEST=F:\Cataract-LMM"

rem Katalog docelowy, gdyby go jeszcze nie bylo
if not exist "%DEST%" mkdir "%DEST%"
