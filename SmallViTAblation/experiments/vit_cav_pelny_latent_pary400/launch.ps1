$ErrorActionPreference = 'Stop'
$taskRoot = 'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation'
$taskOut = Join-Path $taskRoot 'outputs/vit_cav_pelny_latent_pary400'
$taskProcess = Start-Process -FilePath 'F:/Python/python.exe' -ArgumentList @('-B','-u','experiments/vit_cav_pelny_latent_pary400/run.py') -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskOut 'stdout.log') -RedirectStandardError (Join-Path $taskOut 'stderr.log') -PassThru
@{pid=$taskProcess.Id;started=(Get-Date).ToString('o');root=$taskRoot} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOut 'LAUNCH.json') -Encoding utf8
Write-Output $taskProcess.Id
