# Demo Environment Setup Script
# Prepares your system for a perfect screen recording

Write-Host "╔════════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              NextGen IDS - Demo Environment Setup                              ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "🎬 Preparing optimal recording environment..." -ForegroundColor Yellow
Write-Host ""

# 1. Clean terminal history
Write-Host "[1/8] Clearing terminal history..." -ForegroundColor Gray
Clear-Host
Start-Sleep -Milliseconds 500
Write-Host "      ✅ Terminal cleared" -ForegroundColor Green

# 2. Clean old results
Write-Host "[2/8] Cleaning previous results..." -ForegroundColor Gray
Remove-Item results/*.json -ErrorAction SilentlyContinue
Remove-Item results/*.png -ErrorAction SilentlyContinue
Write-Host "      ✅ Results folder cleaned" -ForegroundColor Green

# 3. Reset threat database
Write-Host "[3/8] Resetting threat database..." -ForegroundColor Gray
echo '[]' | Out-File -FilePath data/threats.json -Encoding utf8
Write-Host "      ✅ Threat database reset" -ForegroundColor Green

# 4. Verify checkpoints
Write-Host "[4/8] Verifying model checkpoints..." -ForegroundColor Gray
$checkpoints = Get-ChildItem checkpoints/*.pt
if ($checkpoints.Count -gt 0) {
    Write-Host "      ✅ Found $($checkpoints.Count) checkpoints" -ForegroundColor Green
    foreach ($ckpt in $checkpoints) {
        Write-Host "         - $($ckpt.Name)" -ForegroundColor Gray
    }
} else {
    Write-Host "      ⚠️  No checkpoints found!" -ForegroundColor Yellow
}

# 5. Verify demo data
Write-Host "[5/8] Verifying demo data files..." -ForegroundColor Gray
$demoFiles = @(
    "data/iot23/demo_attacks.csv",
    "data/iot23/multiclass_attacks.csv"
)
$allPresent = $true
foreach ($file in $demoFiles) {
    if (Test-Path $file) {
        Write-Host "      ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "      ❌ $file - MISSING!" -ForegroundColor Red
        $allPresent = $false
    }
}

# 6. Test Python environment
Write-Host "[6/8] Testing Python environment..." -ForegroundColor Gray
try {
    $pythonVersion = python --version 2>&1
    Write-Host "      ✅ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "      ❌ Python not found!" -ForegroundColor Red
}

# 7. Check dependencies
Write-Host "[7/8] Checking key Python packages..." -ForegroundColor Gray
$packages = @("torch", "flask", "pandas", "numpy", "sklearn")
foreach ($pkg in $packages) {
    try {
        python -c "import $pkg; print('$pkg OK')" 2>&1 | Out-Null
        Write-Host "      ✅ $pkg installed" -ForegroundColor Green
    } catch {
        Write-Host "      ❌ $pkg missing!" -ForegroundColor Red
    }
}

# 8. Display system info
Write-Host "[8/8] System information..." -ForegroundColor Gray
$cpu = (Get-WmiObject Win32_Processor).Name
$ram = [math]::Round((Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
Write-Host "      CPU: $cpu" -ForegroundColor Gray
Write-Host "      RAM: $ram GB" -ForegroundColor Gray

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                         ✅ Setup Complete!                                     ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

Write-Host "📋 Recording Checklist:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Before Recording:" -ForegroundColor White
Write-Host "    □ Close unnecessary applications" -ForegroundColor Gray
Write-Host "    □ Disable notifications (Windows Focus Assist)" -ForegroundColor Gray
Write-Host "    □ Set terminal font size to 14-16pt" -ForegroundColor Gray
Write-Host "    □ Set terminal colors to high contrast" -ForegroundColor Gray
Write-Host "    □ Test microphone audio" -ForegroundColor Gray
Write-Host "    □ Prepare browser window (http://localhost:8080)" -ForegroundColor Gray
Write-Host ""
Write-Host "  Terminal Settings:" -ForegroundColor White
Write-Host "    □ Font: Consolas or Cascadia Code" -ForegroundColor Gray
Write-Host "    □ Size: 14-16pt for readability" -ForegroundColor Gray
Write-Host "    □ Colors: High contrast scheme" -ForegroundColor Gray
Write-Host "    □ Window: At least 120x30 characters" -ForegroundColor Gray
Write-Host ""

Write-Host "🎬 Ready to Record!" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Start dashboard: " -NoNewline -ForegroundColor White
Write-Host "python quick_start.py" -ForegroundColor Cyan
Write-Host "  2. Run demos: " -NoNewline -ForegroundColor White
Write-Host ".\demo_scripts\RUN_ALL_DEMOS.ps1" -ForegroundColor Cyan
Write-Host "  3. Or individual: " -NoNewline -ForegroundColor White
Write-Host ".\demo_scripts\1_test_analysis.ps1" -ForegroundColor Cyan
Write-Host ""

# Optional: Set console properties for recording
Write-Host "💡 Tip: Run this command to optimize terminal for recording:" -ForegroundColor Yellow
Write-Host '   $Host.UI.RawUI.WindowSize = New-Object System.Management.Automation.Host.Size(120,30)' -ForegroundColor Cyan
Write-Host '   $Host.UI.RawUI.BufferSize = New-Object System.Management.Automation.Host.Size(120,3000)' -ForegroundColor Cyan
Write-Host ""
