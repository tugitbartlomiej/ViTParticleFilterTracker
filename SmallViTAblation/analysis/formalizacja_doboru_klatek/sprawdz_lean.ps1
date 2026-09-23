param(
    [string]$MathRoot = 'F:\Studia\MCHTR\Grant\Projekt\Research\Math\lean4_spm',
    [string]$LeanExe = 'C:\Users\bartl\.elan\toolchains\leanprover--lean4---v4.28.0\bin\lean.exe'
)

$ErrorActionPreference = 'Stop'
$taskSource = Join-Path $MathRoot 'SpmProofs\ViTFrameSelection.lean'
$taskLog = Join-Path $PSScriptRoot 'LEAN_WERYFIKACJA.log'
$taskReport = Join-Path $PSScriptRoot 'LEAN_WERYFIKACJA.json'
$taskPreviousLeanPath = $env:LEAN_PATH
$taskLibraries = @((Join-Path $MathRoot '.lake\build\lib\lean'))
Get-ChildItem -LiteralPath (Join-Path $MathRoot '.lake\packages') -Directory | ForEach-Object {
    $taskLibrary = Join-Path $_.FullName '.lake\build\lib\lean'
    if (Test-Path -LiteralPath $taskLibrary) { $taskLibraries += $taskLibrary }
}

Push-Location -LiteralPath $MathRoot
try {
    $env:LEAN_PATH = $taskLibraries -join ';'
    $taskVersion = (& $LeanExe --version | Out-String).Trim()
    $taskStart = [DateTime]::UtcNow
    $taskOutput = @(& $LeanExe -j1 $taskSource 2>&1)
    $taskExitCode = $LASTEXITCODE
    $taskElapsed = ([DateTime]::UtcNow - $taskStart).TotalSeconds
    $taskText = ($taskOutput | ForEach-Object { $_.ToString() }) -join "`n"
    $taskSourceText = Get-Content -LiteralPath $taskSource -Raw -Encoding UTF8
    $taskPlaceholders = $taskSourceText -match '\b(sorry|admit)\b'
    $taskUntrustedAxiom = $taskText -match 'sorryAx|ofReduceBool|Lean\.trustCompiler'
    $taskTheorems = @([regex]::Matches($taskSourceText, '(?m)^theorem\s+(\w+)') | ForEach-Object { $_.Groups[1].Value })
    $taskAudited = @([regex]::Matches($taskText, "'SpmProofs\.ViTFrameSelection\.(\w+)' depends on axioms") | ForEach-Object { $_.Groups[1].Value })
    $taskMissingAudit = @($taskTheorems | Where-Object { $_ -notin $taskAudited })
    $taskAxiomAudit = $taskTheorems.Count -gt 0 -and $taskMissingAudit.Count -eq 0
    $taskSuccess = $taskExitCode -eq 0 -and -not $taskPlaceholders -and -not $taskUntrustedAxiom -and $taskAxiomAudit
    $taskManifest = Get-Content -LiteralPath (Join-Path $MathRoot 'lake-manifest.json') -Raw | ConvertFrom-Json
    $taskMathlib = $taskManifest.packages | Where-Object { $_.name -eq 'mathlib' }
    $taskResult = [ordered]@{
        checked_utc = [DateTime]::UtcNow.ToString('o')
        status = $(if ($taskSuccess) { 'passed' } else { 'failed' })
        exit_code = $taskExitCode
        elapsed_seconds = $taskElapsed
        lean_version = $taskVersion
        mathlib_revision = $taskMathlib.rev
        source = $taskSource
        source_sha256 = (Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash.ToLowerInvariant()
        placeholders_found = $taskPlaceholders
        untrusted_axiom_found = $taskUntrustedAxiom
        axiom_audit_present = $taskAxiomAudit
        theorems = $taskTheorems
        axioms_output = $taskText
        log = $taskLog
        scope = 'Only the named theorems and their explicit hypotheses; not empirical RF-DETR assumptions or the full document 44.'
    }
    [IO.File]::WriteAllText($taskLog, $taskText + "`n", [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText($taskReport, ($taskResult | ConvertTo-Json -Depth 5) + "`n", [Text.UTF8Encoding]::new($false))
    $taskResult | ConvertTo-Json -Compress
    if (-not $taskSuccess) { throw "Lean verification failed; inspect $taskLog" }
}
finally {
    $env:LEAN_PATH = $taskPreviousLeanPath
    Pop-Location
}
