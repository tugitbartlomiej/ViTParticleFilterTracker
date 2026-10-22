@echo off
rem Uruchamia aplikacje "Lasery i koloidy" (GUI).
rem Szczegoly w README.md w tym samym katalogu. Python z pliku python.txt (pythonw.exe), inaczej python z PATH.
cd /d "%~dp0"
set PY=python
if exist python.txt (
    for /f "usebackq delims=" %%p in ("python.txt") do set PY=%%~dpppython.exe
)
"%PY%" start.py
if errorlevel 1 (
    echo.
    echo Wystapil blad uruchomienia. Zobacz komunikat powyzej.
    pause
)
