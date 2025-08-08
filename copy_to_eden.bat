@echo off
echo.
echo ====================================
echo   KOPIOWANIE KODU NA KLASTER EDEN
echo ====================================
echo.

echo Przesylanie plikow na klaster Eden...
echo Zrodlo: ../ViTParticleFilterTracker_ForEden/*
echo Cel: eden-cluster:/home2/faculty/bpiotrowski/ViT/
echo.

scp -r "../ViTParticleFilterTracker_ForEden/*" eden-cluster:/home2/faculty/bpiotrowski/ViT/

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Pliki zostaly przeslane pomyslnie!
    echo.
    echo Mozesz teraz polaczyc sie z klastrem:
    echo   ssh eden-cluster
    echo   cd ~/ViT
    echo   ls -la
) else (
    echo.
    echo [ERROR] Wystapil blad podczas przesylania plikow.
    echo Sprawdz polaczenie SSH z klastrem Eden.
)

echo.
pause