$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$filmRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$filmChapters = Get-Content -LiteralPath (Join-Path $filmRoot 'storyboard.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$filmAudioDir = Join-Path $filmRoot 'audio'
New-Item -ItemType Directory -Path $filmAudioDir -Force | Out-Null
$filmSpeaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$filmSpeaker.SelectVoice('Microsoft Paulina Desktop')
$filmSpeaker.Rate = 0
$filmSpeaker.Volume = 100
foreach ($filmChapter in $filmChapters) {
    $filmIndex = 0
    foreach ($filmSlide in $filmChapter.slides) {
        $filmName = ('{0:D2}_{1:D2}.wav' -f [int]$filmChapter.id,$filmIndex)
        $filmPath = Join-Path $filmAudioDir $filmName
        $filmTextPath = $filmPath + '.txt'
        $filmSameText = (Test-Path -LiteralPath $filmTextPath) -and ([System.IO.File]::ReadAllText($filmTextPath) -ceq [string]$filmSlide.narration)
        if ((-not (Test-Path -LiteralPath $filmPath)) -or (-not $filmSameText)) {
            $filmSpeaker.SetOutputToWaveFile($filmPath)
            $filmSpeaker.Speak([string]$filmSlide.narration)
            $filmSpeaker.SetOutputToNull()
            [System.IO.File]::WriteAllText($filmTextPath,[string]$filmSlide.narration,[System.Text.Encoding]::UTF8)
        }
        $filmIndex++
    }
    Write-Output ('Narracja gotowa: ' + $filmChapter.slug)
}
$filmSpeaker.Dispose()
