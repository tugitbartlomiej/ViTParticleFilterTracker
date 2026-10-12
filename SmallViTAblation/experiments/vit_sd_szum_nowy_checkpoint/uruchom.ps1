$ErrorActionPreference = 'Stop'
$taskRoot = 'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation'
$taskOutput = Join-Path $taskRoot 'outputs/vit_sd_szum_nowy_checkpoint'
$taskPython = 'F:/Studia/PhD_projekt/VIT/venvs/rfdetr_1_10_1/Scripts/python.exe'
$taskScript = Join-Path $taskRoot 'experiments/vit_sd_szum_nowy_checkpoint/run.py'
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
if (Test-Path -LiteralPath (Join-Path $taskOutput 'LAUNCH.json')) { throw 'Launch already exists; no duplicate.' }
$env:PYTHONPATH = 'F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele/deps'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:USE_TF = '0'
$env:USE_FLAX = '0'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-u', $taskScript) -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskOutput 'stdout.log') -RedirectStandardError (Join-Path $taskOutput 'stderr.log') -PassThru
$taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskOutput 'PID.txt')
@{pid=$taskProcess.Id;launched_utc=[DateTime]::UtcNow.ToString('o');python=$taskPython;script=$taskScript;seed=202610057719;PYTHONPATH=$env:PYTHONPATH} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOutput 'LAUNCH.json') -Encoding UTF8
$taskProcess.Id
