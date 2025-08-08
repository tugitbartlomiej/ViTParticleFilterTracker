@echo off
echo.
echo ================================================
echo   SYNCHRONIZACJA KODU PYTHON NA KLASTER EDEN
echo ================================================
echo.

echo KROK 1/2: Przygotowanie kodu (filtrowanie plikow)...
echo.
python copy_code_only.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Blad podczas kopiowania lokalnego!
    pause
    exit /b 1
)

echo.
echo ================================================
echo KROK 2/2: Przesylanie na klaster Eden...
echo.

scp -r "../ViTParticleFilterTracker_ForEden/*" eden-cluster:/home2/faculty/bpiotrowski/ViT/

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================
    echo [SUCCESS] SYNCHRONIZACJA ZAKONCZONA POMYSLNIE!
    echo ================================================
    echo.
    echo Co zostalo zrobione:
    echo   1. Skopiowano tylko pliki kodu Python (*.py, *.slurm, *.yaml, etc.)
    echo   2. Pominieto duze pliki (modele, obrazy, logi)
    echo   3. Przeslano na klaster Eden: /home2/faculty/bpiotrowski/ViT/
    echo.
    echo Nastepne kroki:
    echo   ssh eden-cluster
    echo   cd ~/ViT
    echo   ls -la
    echo.
) else (
    echo.
    echo [ERROR] Wystapil blad podczas przesylania na klaster!
    echo Sprawdz:
    echo   - Polaczenie SSH z Eden (ssh eden-cluster)
    echo   - Dostepnosc katalogu ~/ViT na klastrze
)

echo.
pause