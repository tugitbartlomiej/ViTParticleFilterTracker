# PowerShell script to run DINO visualization with Python 3.11
Write-Host "Running DINO Attention Visualization Test..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Change to project directory
Set-Location -Path "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks"

# Run the test script
python test_dino_visualization.py

# Keep window open
Write-Host "`nPress any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")