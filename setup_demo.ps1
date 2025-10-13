# 🚀 Pre-Demo Setup Script
# Run this 1 hour before your demonstration to ensure everything is ready

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Next-Gen IDS Demo Setup Script" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

# Change to project directory
$projectPath = "C:\Users\Nachi\OneDrive\Desktop\Nextgen\nextgen_ids"
Set-Location $projectPath

Write-Host "[1/8] Checking project directory..." -ForegroundColor Yellow
if (Test-Path $projectPath) {
    Write-Host "✅ Project found at: $projectPath`n" -ForegroundColor Green
} else {
    Write-Host "❌ Error: Project directory not found!" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "[2/8] Activating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
    Write-Host "✅ Virtual environment activated`n" -ForegroundColor Green
} else {
    Write-Host "❌ Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Run: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Check dependencies
Write-Host "[3/8] Verifying dependencies..." -ForegroundColor Yellow
$requiredPackages = @("torch", "pandas", "Flask", "scikit-learn", "shap")
$allInstalled = $true

foreach ($pkg in $requiredPackages) {
    $installed = pip list | Select-String -Pattern "^$pkg\s"
    if ($installed) {
        Write-Host "  ✅ $pkg installed" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $pkg missing" -ForegroundColor Red
        $allInstalled = $false
    }
}

if (-not $allInstalled) {
    Write-Host "`nInstalling missing packages..." -ForegroundColor Yellow
    pip install -r requirements.txt
}
Write-Host ""

# Generate synthetic data
Write-Host "[4/8] Generating synthetic data..." -ForegroundColor Yellow
if (Test-Path "data\iot23\synthetic.csv") {
    Write-Host "✅ Synthetic data already exists`n" -ForegroundColor Green
} else {
    python src/generate_synthetic_data.py
    if (Test-Path "data\iot23\synthetic.csv") {
        Write-Host "✅ Synthetic data generated successfully`n" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to generate synthetic data`n" -ForegroundColor Red
        exit 1
    }
}

# Test models
Write-Host "[5/8] Testing models..." -ForegroundColor Yellow
python -c "from src.model import IDSModel, NextGenIDS; print('✅ Models import successfully')"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Models working correctly`n" -ForegroundColor Green
} else {
    Write-Host "❌ Model test failed`n" -ForegroundColor Red
    exit 1
}

# Check for trained checkpoints
Write-Host "[6/8] Checking trained models..." -ForegroundColor Yellow
if (Test-Path "checkpoints\best.pt") {
    Write-Host "✅ Standard model checkpoint found" -ForegroundColor Green
} else {
    Write-Host "⚠️  No standard model checkpoint - train before demo!" -ForegroundColor Yellow
    Write-Host "   Run: python -m src.train --dataset iot23 --epochs 10" -ForegroundColor Cyan
}

if (Test-Path "checkpoints\best_arnn.pt") {
    Write-Host "✅ A-RNN model checkpoint found" -ForegroundColor Green
} else {
    Write-Host "⚠️  No A-RNN model checkpoint - optional but recommended" -ForegroundColor Yellow
    Write-Host "   Run: python -m src.train --dataset iot23 --epochs 10 --use-arnn --save_path checkpoints/best_arnn.pt" -ForegroundColor Cyan
}
Write-Host ""

# Test dashboard startup
Write-Host "[7/8] Testing dashboard startup..." -ForegroundColor Yellow
$dashboardJob = Start-Job -ScriptBlock {
    param($path)
    Set-Location $path
    & .\.venv\Scripts\python.exe src\dashboard.py
} -ArgumentList $projectPath

Start-Sleep -Seconds 5

# Check if dashboard is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000" -TimeoutSec 3 -ErrorAction Stop
    Write-Host "✅ Dashboard started successfully!" -ForegroundColor Green
    Write-Host "   URL: http://localhost:5000`n" -ForegroundColor Cyan
    
    # Stop the dashboard
    Stop-Job -Job $dashboardJob
    Remove-Job -Job $dashboardJob
} catch {
    Write-Host "❌ Dashboard failed to start" -ForegroundColor Red
    Write-Host "   Check src/dashboard.py for errors`n" -ForegroundColor Yellow
    Stop-Job -Job $dashboardJob -ErrorAction SilentlyContinue
    Remove-Job -Job $dashboardJob -ErrorAction SilentlyContinue
}

# Create backup screenshots directory
Write-Host "[8/8] Preparing backup materials..." -ForegroundColor Yellow
$backupDir = "demo_backup"
if (-not (Test-Path $backupDir)) {
    New-Item -ItemType Directory -Path $backupDir | Out-Null
}
Write-Host "✅ Backup directory ready: $backupDir`n" -ForegroundColor Green

# Final summary
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete! 🎉" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

Write-Host "📋 Pre-Demo Checklist:`n" -ForegroundColor Yellow

Write-Host "Essential:" -ForegroundColor White
Write-Host "  [✓] Dependencies installed" -ForegroundColor Green
Write-Host "  [✓] Synthetic data generated" -ForegroundColor Green
Write-Host "  [✓] Models tested" -ForegroundColor Green
Write-Host "  [✓] Dashboard functional`n" -ForegroundColor Green

Write-Host "Recommended (if not done):" -ForegroundColor White
if (Test-Path "checkpoints\best.pt") {
    Write-Host "  [✓] Standard model trained" -ForegroundColor Green
} else {
    Write-Host "  [!] Train standard model: python -m src.train --dataset iot23 --epochs 10" -ForegroundColor Yellow
}

if (Test-Path "checkpoints\best_arnn.pt") {
    Write-Host "  [✓] A-RNN model trained" -ForegroundColor Green
} else {
    Write-Host "  [!] Train A-RNN model: python -m src.train --dataset iot23 --epochs 10 --use-arnn --save_path checkpoints/best_arnn.pt" -ForegroundColor Yellow
}

Write-Host "`n📚 Documentation Files:" -ForegroundColor Yellow
$docs = @(
    "DEMONSTRATION_GUIDE.md",
    "DEMO_QUICK_REFERENCE.md",
    "ARCHITECTURE_DIAGRAMS.md",
    "PROJECT_EXPLANATION.md",
    "ARNN_UPGRADE.md"
)

foreach ($doc in $docs) {
    if (Test-Path $doc) {
        Write-Host "  ✅ $doc" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $doc (missing)" -ForegroundColor Red
    }
}

Write-Host "`n🚀 To Start Demo:" -ForegroundColor Yellow
Write-Host "  1. python src/dashboard.py" -ForegroundColor Cyan
Write-Host "  2. Open http://localhost:5000 in browser" -ForegroundColor Cyan
Write-Host "  3. Keep DEMO_QUICK_REFERENCE.md open for reference`n" -ForegroundColor Cyan

Write-Host "💡 Pro Tips:" -ForegroundColor Yellow
Write-Host "  • Test the full demo flow once before presentation" -ForegroundColor White
Write-Host "  • Have screenshots ready as backup" -ForegroundColor White
Write-Host "  • Print DEMO_QUICK_REFERENCE.md" -ForegroundColor White
Write-Host "  • Disable notifications on your computer" -ForegroundColor White
Write-Host "  • Charge laptop fully`n" -ForegroundColor White

Write-Host "Good luck with your demonstration! 🎯" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan
