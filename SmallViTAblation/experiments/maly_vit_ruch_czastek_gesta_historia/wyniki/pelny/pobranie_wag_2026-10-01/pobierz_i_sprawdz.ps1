param()
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$auditRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$allowedParent = 'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_ruch_czastek_gesta_historia\wyniki\pelny'
if ($auditRoot -ne (Join-Path $allowedParent 'pobranie_wag_2026-10-01')) { throw 'Unexpected audit directory.' }
$localRoot = [System.IO.Path]::GetFullPath((Join-Path $allowedParent 'wagi'))
$manifestPath = Join-Path $auditRoot 'MANIFEST_ZDALNY_2026-10-01.json'
$manifest = [System.IO.File]::ReadAllText($manifestPath) | ConvertFrom-Json
$expectedRemote = '/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi'
if ($manifest.source_host_alias -ne 'eden-cluster' -or $manifest.remote_directory -ne $expectedRemote) { throw 'Unexpected remote source.' }
$expectedNames = @(0..100 | ForEach-Object { 'k{0:D4}.pt' -f ($_ * 10) })
if ($manifest.count -ne 101 -or $manifest.files.Count -ne 101) { throw 'Expected 101 files.' }
if ((@($manifest.files.name) -join '|') -ne ($expectedNames -join '|')) { throw 'Manifest filenames differ from steps 0..1000 every 10.' }
foreach ($entry in $manifest.files) {
    if ($entry.sha256 -notmatch '^[0-9a-f]{64}$' -or [long]$entry.bytes -le 0 -or $entry.step -ne [int]$entry.name.Substring(1,4)) { throw 'Invalid manifest entry.' }
}
[System.IO.Directory]::CreateDirectory($localRoot) | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$started = (Get-Date).ToUniversalTime().ToString('o')
$freeBefore = (Get-PSDrive F).Free
$preservedCorrect = 0
$verifiedPartials = 0
$quarantined = [System.Collections.Generic.List[string]]::new()

function Test-Checkpoint([string]$path, $entry) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $path).Length -ne [long]$entry.bytes) { return $false }
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -eq $entry.sha256
}

function Move-PartialToFinal([string]$part, [string]$final) {
    foreach ($candidate in @($part, $final)) {
        if ([System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($candidate)) -ne $localRoot) { throw 'Path outside owned weight directory.' }
    }
    if (Test-Path -LiteralPath $final) { throw 'Refusing to overwrite an existing final checkpoint.' }
    Move-Item -LiteralPath $part -Destination $final
}

function Preserve-InvalidPartial([string]$part) {
    if ([System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($part)) -ne $localRoot) { throw 'Invalid partial path.' }
    $saved = $part + '.invalid-' + (Get-Date -Format 'yyyyMMdd_HHmmssfff')
    Move-Item -LiteralPath $part -Destination $saved
    $quarantined.Add([System.IO.Path]::GetFileName($saved))
}

foreach ($entry in $manifest.files) {
    $final = Join-Path $localRoot $entry.name
    $part = $final + '.part'
    if (Test-Path -LiteralPath $final) {
        if (-not (Test-Checkpoint $final $entry)) { throw ('Existing final checkpoint differs from remote; preserved: ' + $entry.name) }
        $preservedCorrect++
    } elseif (Test-Path -LiteralPath $part) {
        if (Test-Checkpoint $part $entry) { Move-PartialToFinal $part $final; $verifiedPartials++ }
        elseif ((Get-Item -LiteralPath $part).Length -ge [long]$entry.bytes) { Preserve-InvalidPartial $part }
    }
}

$attemptLogs = [System.Collections.Generic.List[string]]::new()
for ($attempt = 1; $attempt -le 3; $attempt++) {
    $pending = @($manifest.files | Where-Object { -not (Test-Path -LiteralPath (Join-Path $localRoot $_.name)) })
    if ($pending.Count -eq 0) { break }
    $remainingBytes = 0L
    foreach ($entry in $pending) {
        $part = Join-Path $localRoot ($entry.name + '.part')
        $have = if (Test-Path -LiteralPath $part) { (Get-Item -LiteralPath $part).Length } else { 0L }
        $remainingBytes += [long]$entry.bytes - $have
    }
    if ((Get-PSDrive F).Free -lt ($remainingBytes + 256MB)) { throw 'Insufficient free space for remaining transfer plus 256 MiB reserve.' }
    $batch = [System.Collections.Generic.List[string]]::new()
    $batch.Add('lcd "' + $localRoot.Replace('\','/') + '"')
    $batch.Add('cd "' + $expectedRemote + '"')
    foreach ($entry in $pending) { $batch.Add('reget "' + $entry.name + '" "' + $entry.name + '.part"') }
    $batchPath = Join-Path $auditRoot ('sftp_' + $stamp + '_attempt' + $attempt + '.txt')
    [System.IO.File]::WriteAllText($batchPath, ($batch -join "`n") + "`n", $utf8)
    $logPath = Join-Path $auditRoot ('sftp_' + $stamp + '_attempt' + $attempt + '.log')
    $attemptLogs.Add([System.IO.Path]::GetFileName($logPath))
    Write-Output ('TRANSFER_STARTED pending=' + $pending.Count + ' remaining_bytes=' + $remainingBytes + ' attempt=' + $attempt)
    & sftp -b $batchPath -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 eden-cluster *> $logPath
    $sftpExit = $LASTEXITCODE
    $promoted = 0
    foreach ($entry in $pending) {
        $final = Join-Path $localRoot $entry.name
        $part = $final + '.part'
        if (Test-Checkpoint $part $entry) { Move-PartialToFinal $part $final; $promoted++ }
        elseif ((Test-Path -LiteralPath $part) -and (Get-Item -LiteralPath $part).Length -ge [long]$entry.bytes) { Preserve-InvalidPartial $part }
    }
    Write-Output ('ATTEMPT_FINISHED sftp_exit=' + $sftpExit + ' verified_promoted=' + $promoted)
}

$localRecords = @()
foreach ($entry in $manifest.files) {
    $final = Join-Path $localRoot $entry.name
    if (-not (Test-Checkpoint $final $entry)) { throw ('Transfer incomplete or checksum mismatch: ' + $entry.name) }
    $localRecords += [pscustomobject]@{name=$entry.name; step=$entry.step; bytes=(Get-Item -LiteralPath $final).Length; local_sha256=(Get-FileHash -LiteralPath $final -Algorithm SHA256).Hash.ToLowerInvariant(); remote_sha256=$entry.sha256; match=$true}
}
$actualNames = @(Get-ChildItem -LiteralPath $localRoot -File -Filter 'k*.pt' | Sort-Object Name | ForEach-Object Name)
if (($actualNames -join '|') -ne ($expectedNames -join '|')) { throw 'Final checkpoint set differs from the exact expected 101 filenames.' }
$audit = [ordered]@{
    schema='eden-checkpoint-download-verified-v1'; status='complete_verified'; started_at_utc=$started; completed_at_utc=(Get-Date).ToUniversalTime().ToString('o');
    source_host_alias='eden-cluster'; remote_directory=$expectedRemote; local_directory=$localRoot; remote_manifest='MANIFEST_ZDALNY_2026-10-01.json';
    expected_count=101; verified_count=$localRecords.Count; total_bytes=($localRecords | Measure-Object -Property bytes -Sum).Sum;
    preserved_existing_correct=$preservedCorrect; reused_complete_partials=$verifiedPartials; quarantined_partials=@($quarantined); free_bytes_before=$freeBefore; free_bytes_after=(Get-PSDrive F).Free;
    transport='OpenSSH SFTP reget via existing eden-cluster alias'; remote_mutations=$false; latent_reconstruction=$false; latent_dense_download=$false; adamw_full_state_download=$false; attempt_logs=@($attemptLogs); files=$localRecords
}
$auditPath = Join-Path $auditRoot ('MANIFEST_LOKALNY_' + $stamp + '.json')
[System.IO.File]::WriteAllText($auditPath, ($audit | ConvertTo-Json -Depth 6) + "`n", $utf8)
$sums = @($localRecords | ForEach-Object { $_.local_sha256 + '  ' + $_.name })
[System.IO.File]::WriteAllText((Join-Path $auditRoot ('SHA256SUMS_' + $stamp + '.txt')), ($sums -join "`n") + "`n", $utf8)
[pscustomobject]@{status=$audit.status; count=$audit.verified_count; bytes=$audit.total_bytes; existing_preserved=$preservedCorrect; manifest=$auditPath; free_bytes_after=$audit.free_bytes_after} | ConvertTo-Json
