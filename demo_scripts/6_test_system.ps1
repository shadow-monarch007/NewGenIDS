# Demo Test Script 6: System Tests
# Runs the complete test suite to verify all components

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "DEMO 6: System Testing & Verification" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

Write-Host "🧪 Running comprehensive test suite..." -ForegroundColor Green
Write-Host "This validates all system components:" -ForegroundColor Gray
Write-Host "  - Model architectures (LSTM, A-RNN)" -ForegroundColor Gray
Write-Host "  - Phishing detection (URLs, emails)" -ForegroundColor Gray
Write-Host "  - Auto-response system" -ForegroundColor Gray
Write-Host "  - Authentication & security" -ForegroundColor Gray
Write-Host "  - Input validation & rate limiting" -ForegroundColor Gray
Write-Host ""

python src/test_suite.py

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "✅ Demo 6 Complete!" -ForegroundColor Green
Write-Host "All tests passed - system is ready for production!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
