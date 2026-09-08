<#
.SYNOPSIS
    Pobiera z klastra Eden checkpointy, logi i pliki bledow, zachowujac strukture katalogow.

.DESCRIPTION
    Skrypt laczy sie przez SSH (host 'eden-cluster' z ~/.ssh/config), buduje manifest
    plikow pasujacych do wybranych kategorii, a nastepnie pobiera je lokalnie odtwarzajac
    pelna strukture folderow zdalnych (relatywnie do -RemoteRoot).

    Pobieranie jest wznawialne: plik, ktory lokalnie istnieje i ma identyczny rozmiar,
    jest pomijany. Manifest zapisywany jest do pliku JSON w katalogu docelowym.

.PARAMETER What
    checkpoints | logs | all   (domyslnie: all)

.PARAMETER RemoteRoot
    Katalog zdalny do przeszukania (domyslnie: katalog domowy).

.PARAMETER LocalRoot
    Katalog docelowy na dysku lokalnym.

.PARAMETER MaxFileSizeGB
    Pomija pojedyncze pliki wieksze niz podany rozmiar (0 = bez limitu).

.PARAMETER NewerThanDays
    Bierze pod uwage tylko pliki zmodyfikowane w ciagu ostatnich N dni (0 = wszystkie).

.PARAMETER Depth
    Maksymalna glebokosc przeszukiwania find (domyslnie 8).

.PARAMETER DryRun
    Tylko manifest + podsumowanie, bez pobierania.

.PARAMETER UseRsync
    Uzywa rsync (jesli dostepny) zamiast scp - szybsze, wznawia przerwane pliki.

.EXAMPLE
    .\fetch_eden_artifacts.ps1 -DryRun
    .\fetch_eden_artifacts.ps1 -What checkpoints -LocalRoot F:\Eden_Backup
    .\fetch_eden_artifacts.ps1 -What logs -NewerThanDays 30
#>

[CmdletBinding()]
param(
    [ValidateSet('checkpoints', 'logs', 'all')]
    [string]$What = 'all',

    [string]$SshHost = 'eden-cluster',

    [string]$RemoteRoot = '/mnt/evafs/faculty/home/bpiotrowski',

    [string]$LocalRoot = 'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\EdenBackup',

    [double]$MaxFileSizeGB = 0,

    [int]$NewerThanDays = 0,

    [int]$Depth = 8,

    # katalogi pomijane (dopasowanie po nazwie katalogu, wildcardy find)
    [string[]]$ExcludeDirs = @('.*', 'site-packages', 'node_modules', '__pycache__', 'wandb',
                               'archives', 'datasets', 'datasets_*', 'dataset', 'bin'),

    # pliki pomijane niezaleznie od wzorca (archiwa itp.)
    [string[]]$ExcludeFiles = @('*.tar', '*.tar.gz', '*.tgz', '*.zip', '*.7z'),

    [switch]$DryRun,

    [switch]$UseRsync
)

$ErrorActionPreference = 'Stop'

# --- wzorce plikow -----------------------------------------------------------
$checkpointPatterns = @('*.pth', '*.pt', '*.ckpt', '*.safetensors', '*.bin', 'last.pt', 'best.pt')
$logPatterns        = @('*.out', '*.err', '*.log', 'slurm-*', '*.csv', 'results.json',
                        'results.csv', '*.yaml', '*.json', 'args.yaml', 'opt.yaml')

switch ($What) {
    'checkpoints' { $patterns = $checkpointPatterns }
    'logs'        { $patterns = $logPatterns }
    default       { $patterns = $checkpointPatterns + $logPatterns }
}

# --- budowa polecenia find ---------------------------------------------------
$findExpr = ($patterns | ForEach-Object { "-name '$_'" }) -join ' -o '
# normalizacja: tylda musi zostac rozwinieta przez zdalna powloke, wiec -> $HOME
$remoteBase = $RemoteRoot.Trim().TrimEnd('/')
if ([string]::IsNullOrWhiteSpace($remoteBase)) { $remoteBase = '$HOME' }
elseif ($remoteBase -eq '~')                   { $remoteBase = '$HOME' }
elseif ($remoteBase.StartsWith('~/'))          { $remoteBase = '$HOME/' + $remoteBase.Substring(2) }

$rootExpr = "`"$remoteBase`""
$timeExpr = if ($NewerThanDays -gt 0) { "-mtime -$NewerThanDays" } else { '' }

# wykluczenia katalogow (prune) - '.' jako root nie moze zostac przyciety
$pruneExpr = ''
if ($ExcludeDirs.Count -gt 0) {
    $names = ($ExcludeDirs | ForEach-Object { "-name '$_'" }) -join ' -o '
    $pruneExpr = "-mindepth 1 \( $names \) -type d -prune -o"
}

# separator |; find zwraca: rozmiar|mtime_epoch|sciezka_wzgledna
$notExpr = ''
if ($ExcludeFiles.Count -gt 0) {
    $notExpr = '! \( ' + (($ExcludeFiles | ForEach-Object { "-name '$_'" }) -join ' -o ') + ' \)'
}

$remoteCmd = "echo HOME=`$HOME; cd $rootExpr && find . -maxdepth $Depth $pruneExpr -type f $timeExpr \( $findExpr \) $notExpr -printf '%s|%T@|%p\n' 2>/dev/null"

Write-Host "== Eden artifact fetch ==" -ForegroundColor Cyan
Write-Host "Host        : $SshHost"
Write-Host "Zdalny root : $(if ($RemoteRoot) { $RemoteRoot } else { '~ (katalog domowy)' })"
Write-Host "Lokalny root: $LocalRoot"
Write-Host "Zakres      : $What"
Write-Host ""
Write-Host "Buduje manifest zdalnych plikow (moze potrwac)..." -ForegroundColor Yellow

$raw = & ssh -o BatchMode=yes $SshHost $remoteCmd
if ($LASTEXITCODE -ne 0) { throw "Polaczenie SSH / find zakonczone bledem (kod $LASTEXITCODE)." }

# scp (protokol SFTP) nie rozwija ~ ani $HOME - potrzebna sciezka absolutna
$homeLine = $raw | Where-Object { $_ -match '^HOME=' } | Select-Object -First 1
if (-not $homeLine) { throw "Nie udalo sie odczytac katalogu domowego na klastrze." }
$remoteHome = ($homeLine -replace '^HOME=', '').Trim()
$remoteBaseAbs = $remoteBase.Replace('$HOME', $remoteHome)

# --- parsowanie manifestu ----------------------------------------------------
$epoch = [datetime]::new(1970, 1, 1, 0, 0, 0, [System.DateTimeKind]::Utc)
$maxBytes = if ($MaxFileSizeGB -gt 0) { [long]($MaxFileSizeGB * 1GB) } else { [long]::MaxValue }

$files = @()
$skippedTooBig = @()

foreach ($line in $raw) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $parts = $line -split '\|', 3
    if ($parts.Count -lt 3) { continue }

    $size = [long]$parts[0]
    $rel  = $parts[2] -replace '^\./', ''
    $item = [pscustomobject]@{
        RelPath  = $rel
        Size     = $size
        Modified = $epoch.AddSeconds([double]$parts[1]).ToLocalTime()
    }
    if ($size -gt $maxBytes) { $skippedTooBig += $item } else { $files += $item }
}

if ($files.Count -eq 0) {
    Write-Host "Nie znaleziono zadnych plikow pasujacych do kryteriow." -ForegroundColor Yellow
    return
}

$totalGB = [math]::Round((($files | Measure-Object Size -Sum).Sum) / 1GB, 2)
Write-Host "Znaleziono $($files.Count) plikow, lacznie $totalGB GB." -ForegroundColor Green
if ($skippedTooBig.Count -gt 0) {
    Write-Host "Pominieto $($skippedTooBig.Count) plikow ponad limit $MaxFileSizeGB GB." -ForegroundColor Yellow
}

# podsumowanie wg katalogu najwyzszego poziomu
Write-Host ""
Write-Host "Rozklad wg katalogu glownego:" -ForegroundColor Cyan
$files | Group-Object { ($_.RelPath -split '/')[0] } | Sort-Object { -($_.Group | Measure-Object Size -Sum).Sum } |
    ForEach-Object {
        $gb = [math]::Round((($_.Group | Measure-Object Size -Sum).Sum) / 1GB, 2)
        "{0,-30} {1,6} plikow  {2,8} GB" -f $_.Name, $_.Count, $gb
    }

# --- zapis manifestu ---------------------------------------------------------
if (-not (Test-Path $LocalRoot)) { New-Item -ItemType Directory -Path $LocalRoot -Force | Out-Null }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$manifestPath = Join-Path $LocalRoot "manifest_${What}_$stamp.json"
$files | ConvertTo-Json -Depth 3 | Set-Content -Path $manifestPath -Encoding UTF8
Write-Host ""
Write-Host "Manifest: $manifestPath" -ForegroundColor DarkGray

if ($DryRun) {
    Write-Host "`n[DryRun] Zakonczono bez pobierania." -ForegroundColor Yellow
    return
}

# --- sciezka rsync (opcjonalna) ---------------------------------------------
if ($UseRsync -and (Get-Command rsync -ErrorAction SilentlyContinue)) {
    $listFile = Join-Path $env:TEMP "eden_files_$stamp.txt"
    ($files | ForEach-Object { $_.RelPath }) -join "`n" | Set-Content -Path $listFile -Encoding ASCII -NoNewline

    # konwersja sciezek Windows -> format cygwin/msys wymagany przez rsync
    $cygLocal = '/cygdrive/' + $LocalRoot.Substring(0, 1).ToLower() + ($LocalRoot.Substring(2) -replace '\\', '/')
    $cygList  = '/cygdrive/' + $listFile.Substring(0, 1).ToLower() + ($listFile.Substring(2) -replace '\\', '/')
    $srcRoot  = $remoteBase

    Write-Host "`nPobieranie przez rsync..." -ForegroundColor Cyan
    & rsync -avh --partial --progress --files-from="$cygList" "${SshHost}:${srcRoot}/" "$cygLocal/"
    Remove-Item $listFile -ErrorAction SilentlyContinue
    Write-Host "`nGotowe. Dane w: $LocalRoot" -ForegroundColor Green
    return
}

# --- sciezka scp (domyslna, wznawialna po rozmiarze) -------------------------
Write-Host "`nPobieranie przez scp (pliki o zgodnym rozmiarze sa pomijane)..." -ForegroundColor Cyan

$done = 0; $skipped = 0; $failed = @(); $bytesDone = 0L
$i = 0

foreach ($f in $files) {
    $i++
    $localPath = Join-Path $LocalRoot ($f.RelPath -replace '/', '\')
    $localDir  = Split-Path $localPath -Parent

    if (Test-Path $localPath) {
        if ((Get-Item $localPath).Length -eq $f.Size) {
            $skipped++
            continue
        }
    }
    if (-not (Test-Path $localDir)) { New-Item -ItemType Directory -Path $localDir -Force | Out-Null }

    $pct = [int](100 * $i / $files.Count)
    Write-Progress -Activity "Pobieranie z Eden ($What)" `
                   -Status "$i/$($files.Count)  $($f.RelPath)" -PercentComplete $pct

    $remotePath = "$remoteBaseAbs/$($f.RelPath)"
    # -O = tryb legacy SCP; niektore buildy Windows OpenSSH zawodza w trybie SFTP przez ProxyJump
    & scp -O -q -o BatchMode=yes "${SshHost}:$remotePath" "$localPath" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        & scp -q -o BatchMode=yes "${SshHost}:$remotePath" "$localPath" 2>&1 | Out-Null
    }

    if ($LASTEXITCODE -eq 0 -and (Test-Path $localPath)) {
        $done++; $bytesDone += $f.Size
    } else {
        $failed += $f.RelPath
        Write-Host "  BLAD: $($f.RelPath)" -ForegroundColor Red
    }
}
Write-Progress -Activity "Pobieranie z Eden" -Completed

# --- podsumowanie ------------------------------------------------------------
Write-Host ""
Write-Host "== Podsumowanie ==" -ForegroundColor Cyan
Write-Host "Pobrane   : $done plikow ($([math]::Round($bytesDone / 1GB, 2)) GB)"
Write-Host "Pominiete : $skipped (juz obecne lokalnie)"
Write-Host "Bledy     : $($failed.Count)"
if ($failed.Count -gt 0) {
    $failPath = Join-Path $LocalRoot "failed_${What}_$stamp.txt"
    $failed | Set-Content -Path $failPath -Encoding UTF8
    Write-Host "Lista bledow: $failPath (uruchom skrypt ponownie, by dociagnac)" -ForegroundColor Yellow
}
Write-Host "Dane w: $LocalRoot" -ForegroundColor Green
