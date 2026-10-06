@echo off
rem Uruchamia edytor latentu SD (GUI).
rem Szczegoly w README.md w tym samym katalogu.
cd /d "%~dp0"
python app.py
if errorlevel 1 (
    echo.
    echo Wystapil blad uruchomienia. Zobacz komunikat powyzej.
    pause
)
