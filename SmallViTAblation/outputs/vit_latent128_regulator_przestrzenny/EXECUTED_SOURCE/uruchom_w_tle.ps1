$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$outputDir = Join-Path $projectRoot 'outputs/vit_latent128_regulator_przestrzenny'
$processRecord = Join-Path $outputDir 'PROCESS.json'
if (Test-Path -LiteralPath (Join-Path $outputDir 'RESULT.json')) {
    throw 'Istnieje wynik tej kampanii; nie nadpisuję go.'
}
if (Test-Path -LiteralPath $processRecord) {
    $previous = Get-Content -LiteralPath $processRecord -Raw | ConvertFrom-Json
    $live = Get-Process -Id $previous.pid -ErrorAction SilentlyContinue
    if ($live -and ($live.StartTime.ToUniversalTime().ToString('o') -eq $previous.started_utc)) {
        throw 'Eksperyment już działa. Sprawdź PROGRESS.json.'
    }
    throw 'Istnieje zapis poprzedniego uruchomienia. Sprawdź jego logi przed ponownym startem.'
}
$pythonExe = 'F:\Python\python.exe'
$script = Join-Path $PSScriptRoot 'szukaj.py'
$stdout = Join-Path $outputDir 'RUN.log'
$stderr = Join-Path $outputDir 'RUN.err'
$worker = Start-Process -FilePath $pythonExe `
    -ArgumentList @('-u', ('"' + $script + '"'), '--phase', 'sweep') `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -WindowStyle Hidden -PassThru
[ordered]@{
    pid = $worker.Id
    started_utc = $worker.StartTime.ToUniversalTime().ToString('o')
    executable = $pythonExe
    script = $script
    phase = 'sweep'
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json | Set-Content -LiteralPath $processRecord -Encoding UTF8
Get-Content -LiteralPath $processRecord
