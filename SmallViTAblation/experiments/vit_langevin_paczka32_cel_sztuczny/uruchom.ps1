$ErrorActionPreference='Stop'
$taskRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$taskOutput=Join-Path $taskRoot 'outputs/vit_langevin_paczka32_cel_sztuczny'
New-Item -ItemType Directory -Path $taskOutput -Force | Out-Null
$priorPath=$env:PYTHONPATH
$priorUtf=$env:PYTHONUTF8
$priorKmp=$env:KMP_DUPLICATE_LIB_OK
try {
    $env:PYTHONPATH='F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele/deps'
    $env:PYTHONUTF8='1'
    $env:KMP_DUPLICATE_LIB_OK='TRUE'
    $taskPython='F:/Studia/PhD_projekt/VIT/venvs/rfdetr_1_10_1/Scripts/python.exe'
    $taskProcess=Start-Process -FilePath $taskPython -ArgumentList @('-u',(Join-Path $PSScriptRoot 'worker.py')) -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskOutput 'stdout.log') -RedirectStandardError (Join-Path $taskOutput 'stderr.log') -PassThru
    $taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskOutput 'PID.txt')
    @{pid=$taskProcess.Id;status='launched';launched_utc=[DateTime]::UtcNow.ToString('o');script=(Join-Path $PSScriptRoot 'worker.py');python=$taskPython;gpu='queued behind vit_sd_cztery_szumy complete and >=6500MiB free';family='vit_langevin_paczka32_cel_sztuczny'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOutput 'LAUNCH.json')
    $taskProcess.Id
} finally {
    $env:PYTHONPATH=$priorPath
    $env:PYTHONUTF8=$priorUtf
    $env:KMP_DUPLICATE_LIB_OK=$priorKmp
}
