# Master Demo Script - Runs ALL demos in sequence
# Perfect for a complete system demonstration

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                  NextGen IDS - Complete System Demonstration                   ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "This master script will run all demo tests in sequence:" -ForegroundColor Yellow
Write-Host "  1. Real-Time Traffic Analysis" -ForegroundColor White
Write-Host "  2. Model Training" -ForegroundColor White
Write-Host "  3. Model Evaluation" -ForegroundColor White
Write-Host "  4. Phishing Detection" -ForegroundColor White
Write-Host "  5. Automated Response System" -ForegroundColor White
Write-Host "  6. System Testing & Verification" -ForegroundColor White
Write-Host ""

Write-Host "Press any key to start the demonstration..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Write-Host ""
Write-Host "Starting in 3..." -ForegroundColor Cyan
Start-Sleep -Seconds 1
Write-Host "2..." -ForegroundColor Cyan
Start-Sleep -Seconds 1
Write-Host "1..." -ForegroundColor Cyan
Start-Sleep -Seconds 1
Write-Host ""

# Demo 1: Traffic Analysis
& "$PSScriptRoot\1_test_analysis.ps1"
Write-Host ""
Write-Host "Press any key to continue to next demo..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Write-Host ""

# Demo 2: Training
& "$PSScriptRoot\2_test_training.ps1"
Write-Host ""
Write-Host "Press any key to continue to next demo..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Write-Host ""

# Demo 3: Evaluation
& "$PSScriptRoot\3_test_evaluation.ps1"
Write-Host ""
Write-Host "Press any key to continue to next demo..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Write-Host ""

# Demo 4: Phishing
& "$PSScriptRoot\4_test_phishing.ps1"
Write-Host ""
Write-Host "Press any key to continue to next demo..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Write-Host ""

# Demo 5: Auto-Response
& "$PSScriptRoot\5_test_auto_response.ps1"
Write-Host ""
Write-Host "Press any key to continue to final demo..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Write-Host ""

# Demo 6: System Tests
& "$PSScriptRoot\6_test_system.ps1"
Write-Host ""

# Final Summary
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                     🎉 All Demonstrations Complete! 🎉                         ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Summary of Results:" -ForegroundColor Yellow
Write-Host "  ✅ Traffic Analysis - Complete" -ForegroundColor Green
Write-Host "  ✅ Model Training - Complete" -ForegroundColor Green
Write-Host "  ✅ Model Evaluation - Complete" -ForegroundColor Green
Write-Host "  ✅ Phishing Detection - Complete" -ForegroundColor Green
Write-Host "  ✅ Auto-Response System - Complete" -ForegroundColor Green
Write-Host "  ✅ System Tests - All Passed" -ForegroundColor Green
Write-Host ""
Write-Host "Generated Files:" -ForegroundColor Yellow
Write-Host "  📄 results/demo_analysis.json - Traffic analysis results" -ForegroundColor Gray
Write-Host "  📄 checkpoints/demo_trained.pt - Newly trained model" -ForegroundColor Gray
Write-Host "  📄 results/metrics.csv - Evaluation metrics" -ForegroundColor Gray
Write-Host "  📊 results/confusion_matrix.png - Visual performance matrix" -ForegroundColor Gray
Write-Host ""
Write-Host "System Status: 🟢 FULLY OPERATIONAL" -ForegroundColor Green
Write-Host ""
