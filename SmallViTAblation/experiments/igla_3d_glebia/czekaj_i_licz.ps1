# Czeka, aż zwolni się GPU (brak procesu horyzont.py i obciążenie karty < 20 % przez 3 kolejne minuty),
# potem uruchamia glebia_proba.py na GPU. Polecenie użytkownika 2026-09-30: „Poczekaj, aż się zwolni GPU, i potem puść na GPU”.
$ErrorActionPreference = 'Continue'
$tu = Split-Path -Parent $MyInvocation.MyCommand.Path
$log = Join-Path $tu 'wyniki\glebia_proba.log'
New-Item -ItemType Directory -Force (Join-Path $tu 'wyniki') | Out-Null
"czekam na GPU od $(Get-Date -Format s)" | Out-File $log -Encoding utf8
$wolne = 0
while ($wolne -lt 3) {
    $h = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'horyzont\.py' }
    $util = [int]((nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) | Select-Object -First 1)
    if (-not $h -and $util -lt 20) { $wolne += 1 } else { $wolne = 0 }
    Start-Sleep -Seconds 60
}
"GPU wolne od $(Get-Date -Format s), start" | Out-File $log -Append -Encoding utf8
$env:HF_HOME = 'F:\HuggingFace'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONUTF8 = '1'; $env:KMP_DUPLICATE_LIB_OK = 'TRUE'
& 'F:\Python\3.13\python.exe' -u (Join-Path $tu 'glebia_proba.py') 2>&1 | Out-File $log -Append -Encoding utf8
"KONIEC $(Get-Date -Format s) kod $LASTEXITCODE" | Out-File $log -Append -Encoding utf8
