$ErrorActionPreference = 'Stop'
$sourceRoot = 'F:/Studia/MCHTR/Grant/Projekt'
$targetRoot = 'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation'
$families = @('vit_latent128_jedno_zdjecie_90','vit_latent128_maksymalizacja','vit_latent128_obraz_sd','vit_latent128_regulator_przestrzenny','vit_latent128_wspolna_regulacja','vit_latent768_sd_od_szumu','vit_optymalizacja_latentu128')
$logDir = Join-Path $targetRoot 'outputs/przeniesienie_latent_sd_20261005'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$records = [System.Collections.Generic.List[object]]::new()
foreach ($group in @('experiments','outputs')) {
    foreach ($family in $families) {
        $from = Join-Path $sourceRoot "$group/$family"
        $to = Join-Path $targetRoot "$group/$family"
        if (Test-Path -LiteralPath $to) { throw "Folder docelowy juz istnieje: $to" }
        & robocopy $from $to /E /XJ /XD __pycache__ .git /R:1 /W:1 /COPY:DAT /DCOPY:DAT /NFL /NDL /NP /NJH /NJS "/LOG+:$logDir/robocopy.log"
        if ($LASTEXITCODE -ge 8) { throw "Blad robocopy $family : $LASTEXITCODE" }
        Write-Output "COPIED $group/$family"
        foreach ($file in Get-ChildItem -LiteralPath $from -File -Recurse) {
            $relative = [System.IO.Path]::GetRelativePath($from,$file.FullName)
            if ($relative -match '(^|[\\/])(__pycache__|\.git)([\\/]|$)') { continue }
            $destination = Join-Path $to $relative
            $originalHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
            $copiedHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
            if ($originalHash -ne $copiedHash) { throw "Rozne sumy: $destination" }
            $records.Add([pscustomobject]@{source=$file.FullName;destination=$destination;bytes=$file.Length;sha256=$originalHash})
        }
        Write-Output "VERIFIED $group/$family"
    }
}
$helperDir = Join-Path $targetRoot 'src/narzedzia'
New-Item -ItemType Directory -Path $helperDir -Force | Out-Null
$helper = Join-Path $helperDir 'deklaracja_wynikow.py'
if (Test-Path -LiteralPath $helper) { throw "Istniejacy helper: $helper" }
Copy-Item -LiteralPath (Join-Path $sourceRoot 'src/narzedzia/deklaracja_wynikow.py') -Destination $helper
[pscustomobject]@{status='complete';source_root=$sourceRoot;target_root=$targetRoot;families=$families;files=$records.Count;bytes=($records | Measure-Object bytes -Sum).Sum;verified_sha256=$true;mode='copy, sources preserved';records=$records} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $logDir 'MANIFEST.json') -Encoding utf8
Write-Output "COMPLETE verified_files=$($records.Count)"
