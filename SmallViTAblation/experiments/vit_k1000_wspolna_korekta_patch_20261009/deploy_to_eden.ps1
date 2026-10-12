param([string]$HostName = "eden-cluster")
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent (Split-Path -Parent $Here)
$Output = Join-Path $Repo "outputs\vit_k1000_wspolna_korekta_patch_20261009"
$Remote = "/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wspolna_korekta_patch_20261009"

ssh -o BatchMode=yes -o StrictHostKeyChecking=yes $HostName "test ! -e '$Remote/submission/SMOKE_SUBMITTED.json' && mkdir -p '$Remote/code' '$Remote/eden' '$Remote/provenance' '$Remote/logs' '$Remote/results' '$Remote/submission' '$Remote/preflight'"
if ($LASTEXITCODE -ne 0) { throw "Remote root ma już zgłoszony smoke albo nie można go utworzyć." }

$CodeFiles = Get-ChildItem -LiteralPath $Here -File | Where-Object { $_.Extension -in @('.py','.md','.json') }
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes @($CodeFiles.FullName) "${HostName}:$Remote/code/"
if ($LASTEXITCODE -ne 0) { throw "Nie udało się skopiować kodu." }
$EdenFiles = Get-ChildItem -LiteralPath (Join-Path $Here 'eden') -File
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes @($EdenFiles.FullName) "${HostName}:$Remote/eden/"
if ($LASTEXITCODE -ne 0) { throw "Nie udało się skopiować SLURM." }
$ProvenanceFiles = Get-ChildItem -LiteralPath (Join-Path $Output 'provenance') -File
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes @($ProvenanceFiles.FullName) "${HostName}:$Remote/provenance/"
if ($LASTEXITCODE -ne 0) { throw "Nie udało się skopiować provenance." }

$RemoteCommand = @"
set -euo pipefail
find '$Remote/code' '$Remote/eden' -type f \( -name '*.py' -o -name '*.sh' -o -name '*.slurm' \) -exec sed -i 's/\r$//' {} +
rm -f '$Remote/code/__init__.py'
chmod u+x '$Remote/eden/'*.sh
source '$Remote/eden/common.sh'
cd '$Remote/code'
"`$TEST_PY" -m pytest -q test_wspolna_korekta.py
"`$PY" -m compileall -q .
"`$PY" '$Remote/eden/preflight.py' --mode deploy --root '$Remote' --checkpoint "`$CHECKPOINT" --adamw "`$CHECKPOINT_ADAMW" --coco "`$COCO" --archive "`$ARCHIVE" --manifest "`$MANIFEST" --old-evaluation "`$OLD_EVALUATION"
"@
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes $HostName $RemoteCommand
if ($LASTEXITCODE -ne 0) { throw "Zdalne testy lub preflight nie przeszły." }
