$ErrorActionPreference = 'Stop'
$taskRoot = 'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation'
$taskOutput = Join-Path $taskRoot 'outputs/vit_sd_cztery_szumy'
$taskPython = 'F:/Studia/PhD_projekt/VIT/venvs/rfdetr_1_10_1/Scripts/python.exe'
$taskScript = Join-Path $taskRoot 'experiments/vit_sd_cztery_szumy/generuj.py'
if (Test-Path -LiteralPath (Join-Path $taskOutput 'LAUNCH.json')) { throw 'Launch already registered; do not start a duplicate.' }
$env:PYTHONPATH = 'F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele/deps'
$env:USE_TF = '0'
$env:USE_FLAX = '0'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-u', $taskScript) -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskOutput 'stdout.log') -RedirectStandardError (Join-Path $taskOutput 'stderr.log') -PassThru
$taskLaunch = @{ pid=$taskProcess.Id; launched_utc=[DateTime]::UtcNow.ToString('o'); python=$taskPython; script=$taskScript; seeds=@(2026100511,2026100523,2026100537,2026100549); environment=@{PYTHONPATH=$env:PYTHONPATH; USE_TF='0'; HF_HUB_OFFLINE='1'; TRANSFORMERS_OFFLINE='1'} }
$taskLaunch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskOutput 'LAUNCH.json') -Encoding UTF8
$taskLaunch | ConvertTo-Json -Depth 5
