# Quick launcher for the web dashboard
# Double-click this file or run: .\start_dashboard.ps1

Write-Host "🚀 Starting Next-Gen IDS Dashboard..." -ForegroundColor Cyan
Write-Host ""

# Navigate to project directory
Set-Location $PSScriptRoot

# Activate virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "✓ Activating virtual environment..." -ForegroundColor Green
    & .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "✗ Virtual environment not found. Run setup first!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Set Python path
$env:PYTHONPATH = (Get-Location).Path
Write-Host "✓ PYTHONPATH set to: $env:PYTHONPATH" -ForegroundColor Green

# Check if Flask is installed
try {
    python -c "import flask" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠ Flask not found. Installing..." -ForegroundColor Yellow
        pip install flask --quiet
    }
    Write-Host "✓ Flask ready" -ForegroundColor Green
} catch {
    Write-Host "⚠ Installing Flask..." -ForegroundColor Yellow
    pip install flask
}

Write-Host ""
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║   🌐 Dashboard starting on port 5000   ║" -ForegroundColor Magenta
Write-Host "║                                        ║" -ForegroundColor Magenta
Write-Host "║   Open browser and navigate to:       ║" -ForegroundColor Magenta
Write-Host "║   http://localhost:5000                ║" -ForegroundColor Cyan
Write-Host "║                                        ║" -ForegroundColor Magenta
Write-Host "║   Press Ctrl+C to stop the server     ║" -ForegroundColor Magenta
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""

# Start dashboard
python src\dashboard.py
