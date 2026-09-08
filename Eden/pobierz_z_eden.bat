@echo off
setlocal EnableDelayedExpansion
rem ============================================================================
rem  pobierz_z_eden.bat
rem  Pobiera z klastra Eden checkpointy, logi i pliki bledow, zachowujac
rem  strukture katalogow. Pomija datasety, katalog bin i archiwa tar/tar.gz.
rem
rem  Domyslny cel: ...\ViTParticleFilterTracker\Eden\EdenBackup
rem
rem  Uzycie:
rem    pobierz_z_eden.bat                 - wszystko (checkpointy + logi)
rem    pobierz_z_eden.bat checkpoints     - tylko checkpointy
rem    pobierz_z_eden.bat logs            - tylko logi i pliki bledow
rem    pobierz_z_eden.bat all /dryrun     - tylko lista, bez pobierania
rem    pobierz_z_eden.bat all F:\Kopia    - wszystko do innego katalogu
rem    pobierz_z_eden.bat checkpoints /root:/mnt/evafs/faculty/home/bpiotrowski/DETR
rem                                       - ograniczenie do jednego podkatalogu
rem
rem  Wznawianie: plik obecny lokalnie o zgodnym rozmiarze jest pomijany,
rem  wiec przerwane pobieranie wznawia sie przez ponowne uruchomienie.
rem ============================================================================

rem --- konfiguracja -----------------------------------------------------------
set "SSH_HOST=eden-cluster"
set "BASE_ROOT=/mnt/evafs/faculty/home/bpiotrowski"
set "REMOTE_ROOT=%BASE_ROOT%"
set "LOCAL_ROOT=F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\EdenBackup"
set "DEPTH=8"

rem --- argumenty --------------------------------------------------------------
set "WHAT=all"
set "DRYRUN="
set "POSITIONAL=0"

:parse_args
if "%~1"=="" goto :args_done
set "ARG=%~1"
if /i "!ARG!"=="/dryrun" (
    set "DRYRUN=1"
) else if /i "!ARG:~0,6!"=="/root:" (
    set "REMOTE_ROOT=!ARG:~6!"
) else (
    set /a POSITIONAL+=1
    if !POSITIONAL!==1 set "WHAT=!ARG!"
    if !POSITIONAL!==2 set "LOCAL_ROOT=!ARG!"
)
shift
goto :parse_args
:args_done
rem usun ewentualny koncowy ukosnik zdalnego roota
if "%REMOTE_ROOT:~-1%"=="/" set "REMOTE_ROOT=%REMOTE_ROOT:~0,-1%"

rem przy zawezeniu /root: zachowaj strukture wzgledem katalogu domowego
set "SUBPREFIX="
call set "SUBPREFIX=%%REMOTE_ROOT:%BASE_ROOT%=%%"
set "SUBPREFIX=%SUBPREFIX:/=\%"

if /i "%WHAT%"=="checkpoints" goto :ok_what
if /i "%WHAT%"=="logs"        goto :ok_what
if /i "%WHAT%"=="all"         goto :ok_what
echo [BLAD] Nieznany zakres "%WHAT%". Dozwolone: checkpoints ^| logs ^| all
exit /b 1
:ok_what

rem --- wzorce plikow ----------------------------------------------------------
set "PAT_CKPT=-name '*.pth' -o -name '*.pt' -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.bin'"
set "PAT_LOG=-name '*.out' -o -name '*.err' -o -name '*.log' -o -name 'slurm-*' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml'"

if /i "%WHAT%"=="checkpoints" set "PATTERNS=%PAT_CKPT%"
if /i "%WHAT%"=="logs"        set "PATTERNS=%PAT_LOG%"
if /i "%WHAT%"=="all"         set "PATTERNS=%PAT_CKPT% -o %PAT_LOG%"

rem --- wykluczenia: datasety, bin, archiwa, smieci ----------------------------
set "EXCL_DIRS=-name '.*' -o -name 'datasets' -o -name 'datasets_*' -o -name 'dataset' -o -name 'bin' -o -name 'archives' -o -name 'site-packages' -o -name 'node_modules' -o -name '__pycache__' -o -name 'wandb'"
set "EXCL_FILES=-name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.zip' -o -name '*.7z'"

rem --- pliki robocze ----------------------------------------------------------
for /f "tokens=1-6 delims=/:. " %%a in ("%DATE% %TIME%") do set "STAMP=%%c%%b%%a_%%d%%e"
set "STAMP=%STAMP: =0%"
set "LIST=%TEMP%\eden_manifest_%STAMP%.txt"

echo == Pobieranie artefaktow z Eden ==
echo Host        : %SSH_HOST%
echo Zdalny root : %REMOTE_ROOT%
echo Lokalny root: %LOCAL_ROOT%
echo Zakres      : %WHAT%
echo Pomijane    : datasety, bin, archiwa tar/tar.gz/zip
echo.
echo Buduje liste plikow na klastrze (moze potrwac)...

ssh -o BatchMode=yes %SSH_HOST% "cd '%REMOTE_ROOT%' && find . -maxdepth %DEPTH% -mindepth 1 \( %EXCL_DIRS% \) -type d -prune -o -type f \( %PATTERNS% \) -not \( %EXCL_FILES% \) -printf '%%s|%%p\n' 2>/dev/null" > "%LIST%"
if errorlevel 1 (
    echo [BLAD] Polaczenie SSH lub find zakonczone niepowodzeniem.
    exit /b 1
)

rem --- podsumowanie listy -----------------------------------------------------
set /a COUNT=0
set "TOTAL_MB=0"
for /f "usebackq tokens=1,* delims=|" %%A in ("%LIST%") do (
    set /a COUNT+=1
    set /a "TOTAL_MB+=%%A/1048576"
)
if %COUNT%==0 (
    echo Nie znaleziono zadnych plikow pasujacych do kryteriow.
    del "%LIST%" 2>nul
    exit /b 0
)
set /a TOTAL_GB=TOTAL_MB/1024
echo Znaleziono %COUNT% plikow, lacznie ok. %TOTAL_GB% GB (%TOTAL_MB% MB).
echo Lista: %LIST%

if defined DRYRUN (
    echo.
    echo [DryRun] Zakonczono bez pobierania.
    exit /b 0
)

if not exist "%LOCAL_ROOT%" mkdir "%LOCAL_ROOT%"
if not exist "%LOCAL_ROOT%\_manifests" mkdir "%LOCAL_ROOT%\_manifests"
rem katalog kopii nie trafia do repozytorium git
if not exist "%LOCAL_ROOT%\.gitignore" echo *>"%LOCAL_ROOT%\.gitignore"
copy /y "%LIST%" "%LOCAL_ROOT%\_manifests\manifest_%WHAT%_%STAMP%.txt" >nul
set "FAILLOG=%LOCAL_ROOT%\_manifests\failed_%WHAT%_%STAMP%.txt"

echo.
echo Pobieranie (pliki o zgodnym rozmiarze sa pomijane)...
echo.

set /a DONE=0, SKIPPED=0, FAILED=0, IDX=0

for /f "usebackq tokens=1,* delims=|" %%A in ("%LIST%") do (
    set /a IDX+=1
    set "SIZE=%%A"
    set "REL=%%B"
    set "REL=!REL:./=!"
    set "WINREL=!REL:/=\!"
    set "DEST=%LOCAL_ROOT%%SUBPREFIX%\!WINREL!"

    set "SKIP="
    if exist "!DEST!" (
        for %%F in ("!DEST!") do if "%%~zF"=="!SIZE!" set "SKIP=1"
    )

    if defined SKIP (
        set /a SKIPPED+=1
    ) else (
        for %%D in ("!DEST!") do if not exist "%%~dpD" mkdir "%%~dpD"
        echo [!IDX!/%COUNT%] !REL!
        scp -O -q -o BatchMode=yes "%SSH_HOST%:%REMOTE_ROOT%/!REL!" "!DEST!" >nul 2>&1
        if errorlevel 1 (
            scp -q -o BatchMode=yes "%SSH_HOST%:%REMOTE_ROOT%/!REL!" "!DEST!" >nul 2>&1
        )
        if exist "!DEST!" (
            set /a DONE+=1
        ) else (
            set /a FAILED+=1
            echo   BLAD: !REL!
            echo !REL!>>"%FAILLOG%"
        )
    )
)

echo.
echo == Podsumowanie ==
echo Pobrane   : %DONE%
echo Pominiete : %SKIPPED% (juz obecne lokalnie)
echo Bledy     : %FAILED%
if %FAILED% GTR 0 echo Lista bledow: %FAILLOG% (uruchom skrypt ponownie, by dociagnac)
echo Dane w: %LOCAL_ROOT%

del "%LIST%" 2>nul
endlocal
exit /b 0
