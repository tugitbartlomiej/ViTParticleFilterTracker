param(
    [string]$HostName = "eden-cluster"
)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Hold = Join-Path $Here "eden\DEPLOY_HOLD.md"
if (Test-Path -LiteralPath $Hold) {
    throw "Deploy blocked by $Hold."
}
$Remote = "/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008"
$OutputRoot = Join-Path (Split-Path -Parent (Split-Path -Parent $Here)) "outputs\vit_k1000_1500_pelne_latenty_90_20261008"
$Provenance = Join-Path $OutputRoot "provenance"

ssh -o BatchMode=yes -o StrictHostKeyChecking=yes $HostName "test ! -e '$Remote/submission/PRODUCTION_SUBMITTED.json' && mkdir -p '$Remote/code' '$Remote/eden' '$Remote/provenance' '$Remote/logs' '$Remote/results'"
if ($LASTEXITCODE -ne 0) {
    throw "Remote directory preflight failed with exit code $LASTEXITCODE."
}
$CodeFiles = Get-ChildItem -LiteralPath $Here -File | Where-Object { $_.Extension -in @('.py', '.md', '.json') }
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes @($CodeFiles.FullName) "${HostName}:$Remote/code/"
if ($LASTEXITCODE -ne 0) {
    throw "Code upload failed with exit code $LASTEXITCODE."
}
$EdenFiles = Get-ChildItem -LiteralPath (Join-Path $Here "eden") -File | Where-Object { $_.Name -ne 'DEPLOY_HOLD.md' }
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes @($EdenFiles.FullName) "${HostName}:$Remote/eden/"
if ($LASTEXITCODE -ne 0) {
    throw "SLURM upload failed with exit code $LASTEXITCODE."
}
$ProvenanceFiles = Get-ChildItem -LiteralPath $Provenance -File | Select-Object -ExpandProperty FullName
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes $ProvenanceFiles "${HostName}:$Remote/provenance/"
if ($LASTEXITCODE -ne 0) {
    throw "Provenance upload failed with exit code $LASTEXITCODE."
}
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes $HostName "rm -f '$Remote/code/__init__.py' && find '$Remote/code' '$Remote/eden' -type f \( -name '*.py' -o -name '*.sh' -o -name '*.slurm' \) -exec sed -i 's/\r$//' {} + && chmod u+x '$Remote/eden/'*.sh && cd '$Remote/code' && /home2/faculty/bpiotrowski/venvs/nndmd_expanded_20261002/bin/python -m pytest -q test_pelne_latenty.py"
if ($LASTEXITCODE -ne 0) {
    throw "Remote tests failed with exit code $LASTEXITCODE."
}
Write-Output "Code deployed without submitting jobs: $Remote"
