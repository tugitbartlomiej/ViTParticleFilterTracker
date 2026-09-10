@echo off
rem ============================================================
rem  KROK 0 - instalacja narzedzia i logowanie
rem  Uruchom RAZ, przed pozostalymi skryptami.
rem  Zbior jest za bramka: bez tokenu pobieranie zwroci blad 401.
rem ============================================================
setlocal
cd /d "%~dp0"

echo.
echo === Instalacja huggingface_hub ===
py -3.11 -m pip install -U "huggingface_hub[cli]"
if errorlevel 1 goto blad

echo.
echo === Logowanie ===
echo Token wygenerujesz tutaj:  https://huggingface.co/settings/tokens
echo Wystarczy token typu READ.
echo Wczesniej wejdz na strone zbioru i zaakceptuj warunki dostepu:
echo   https://huggingface.co/datasets/mjahmadi/Cataract-LMM
echo.
hf auth login
if errorlevel 1 goto blad

echo.
echo === Gotowe. Mozesz uruchomic 1_pobierz_ramki_coco.bat ===
goto koniec

:blad
echo.
echo !!! Cos poszlo nie tak. Przeczytaj komunikat powyzej.

:koniec
echo.
pause
endlocal
