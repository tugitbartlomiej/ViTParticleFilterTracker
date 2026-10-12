$ErrorActionPreference = 'Stop'
$fastganRepo = 'deepsynthbody/deepfake_gi_fastGAN'
$fastganRevision = '5dcd21b18d67339076b41355e682ed6b5a592b45'
$fastganDestination = Join-Path $PSScriptRoot 'vendor/deepfake_gi_fastGAN'
$fastganTreeUrl = "https://huggingface.co/api/models/$fastganRepo/tree/${fastganRevision}?recursive=true"
$fastganResponse = Invoke-WebRequest -Uri $fastganTreeUrl -UseBasicParsing
if ($fastganResponse.Headers['Link'] -match 'rel="next"') {
    throw 'Lista plikow wymaga stronicowania; przerwano, aby nie pobrac niepelnego zestawu.'
}
$fastganTree = $fastganResponse.Content | ConvertFrom-Json
$fastganFiles = @($fastganTree | Where-Object {
    $_.type -eq 'file' -and $_.path -notmatch '(^|/)(__pycache__|test_out)(/|$)' -and
    $_.path -notin @('.gitattributes', '.gitignore') -and $_.path -notmatch '\.pyc$'
})
$fastganRoot = [System.IO.Path]::GetFullPath($fastganDestination)
New-Item -ItemType Directory -Path $fastganRoot -Force | Out-Null
$fastganRecords = @()
foreach ($fastganEntry in $fastganFiles) {
    $fastganTarget = [System.IO.Path]::GetFullPath((Join-Path $fastganRoot $fastganEntry.path))
    if (-not $fastganTarget.StartsWith($fastganRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Sciezka poza katalogiem: $($fastganEntry.path)"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $fastganTarget) -Force | Out-Null
    $fastganUrl = "https://huggingface.co/$fastganRepo/resolve/$fastganRevision/$($fastganEntry.path)"
    if (-not (Test-Path -LiteralPath $fastganTarget)) {
        $fastganPartial = $fastganTarget + '.download'
        if (Test-Path -LiteralPath $fastganPartial) { throw "Istnieje niepelny plik: $fastganPartial" }
        & curl.exe --fail --location --silent --show-error --retry 3 --output $fastganPartial $fastganUrl
        if ($LASTEXITCODE -ne 0) { throw "Nieudane pobranie: $($fastganEntry.path)" }
        if ((Get-Item -LiteralPath $fastganPartial).Length -ne $fastganEntry.size) {
            throw "Niezgodny rozmiar: $($fastganEntry.path)"
        }
        $fastganDigest = (Get-FileHash -LiteralPath $fastganPartial -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($fastganEntry.lfs -and $fastganDigest -ne $fastganEntry.lfs.oid) {
            throw "Niezgodna suma LFS: $($fastganEntry.path)"
        }
        Move-Item -LiteralPath $fastganPartial -Destination $fastganTarget
    }
    if ((Get-Item -LiteralPath $fastganTarget).Length -ne $fastganEntry.size) {
        throw "Istniejacy plik ma inny rozmiar: $fastganTarget"
    }
    $fastganDigest = (Get-FileHash -LiteralPath $fastganTarget -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($fastganEntry.lfs -and $fastganDigest -ne $fastganEntry.lfs.oid) {
        throw "Istniejacy plik ma inna sume LFS: $fastganTarget"
    }
    $fastganRecords += [ordered]@{ path=$fastganEntry.path; bytes=$fastganEntry.size; sha256=$fastganDigest; url=$fastganUrl }
    Write-Output "Pobrano i sprawdzono: $($fastganEntry.path)"
}
$fastganManifest = [ordered]@{
    repository=$fastganRepo; revision=$fastganRevision; downloaded_at=(Get-Date -Format o)
    destination=$fastganRoot; weights_loaded=$false; training_started=$false
    excluded=@('__pycache__', 'test_out', '.gitattributes', '.gitignore', '*.pyc')
    files=$fastganRecords
}
$fastganManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'POBRANIE.json') -Encoding UTF8
