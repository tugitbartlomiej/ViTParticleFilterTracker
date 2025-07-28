# PowerShell script to run DINO visualization with Python 3.11
Write-Host "Running DINO Attention Visualization Test..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# Get current directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $scriptPath

Write-Host "Working directory: $scriptPath" -ForegroundColor Cyan
Write-Host ""

# Check if background frames exist
$backgroundFramesPath = Join-Path $scriptPath "background_frames"
if (Test-Path $backgroundFramesPath) {
    $frameCount = (Get-ChildItem -Path $backgroundFramesPath -Filter "*.jpg").Count
    Write-Host "Found $frameCount background frames to process" -ForegroundColor Green
} else {
    Write-Host "Error: Background frames directory not found!" -ForegroundColor Red
    Write-Host "Expected path: $backgroundFramesPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the test script
Write-Host ""
Write-Host "Starting DINO visualization..." -ForegroundColor Yellow
python test_dino_visualization.py

# Check if visualization completed
$outputPath = Join-Path $scriptPath "dino_visualization_results"
if (Test-Path $outputPath) {
    Write-Host ""
    Write-Host "Visualization completed successfully!" -ForegroundColor Green
    Write-Host "Results saved to: $outputPath" -ForegroundColor Cyan
    
    # List created files
    Write-Host ""
    Write-Host "Created files:" -ForegroundColor Yellow
    Get-ChildItem -Path $outputPath -Recurse | Where-Object { -not $_.PSIsContainer } | ForEach-Object {
        Write-Host "  - $($_.Name)" -ForegroundColor Gray
    }
}

# Keep window open
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")