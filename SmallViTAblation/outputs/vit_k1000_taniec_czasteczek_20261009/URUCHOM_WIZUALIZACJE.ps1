$ErrorActionPreference = 'Stop'
$vizOutput = $PSScriptRoot
$vizProject = (Resolve-Path -LiteralPath (Join-Path $vizOutput '../..')).Path
$vizScript = Join-Path $vizProject 'experiments/vit_k1000_taniec_czasteczek_20261009/serve.py'
$vizListener = Get-NetTCPConnection -LocalPort 8774 -State Listen -ErrorAction SilentlyContinue
if ($vizListener) {
    $vizProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($vizListener.OwningProcess)"
    if (-not $vizProcess.CommandLine.Contains('vit_k1000_taniec_czasteczek_20261009')) {
        throw 'Port 8774 jest zajęty przez inny program. Nie zatrzymujemy go.'
    }
} else {
    Start-Process -FilePath 'F:/Python/python.exe' -ArgumentList @($vizScript,'--port','8774') `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $vizOutput 'SERVER.stdout.log') `
        -RedirectStandardError (Join-Path $vizOutput 'SERVER.stderr.log') | Out-Null
}
Start-Process 'http://127.0.0.1:8774/'
