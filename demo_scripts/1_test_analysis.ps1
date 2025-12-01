# Demo Test Script 1: Traffic Analysis
# Analyzes CSV file and displays threat detection results

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "DEMO 1: Real-Time Traffic Analysis" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Clean previous results
Write-Host "📁 Cleaning previous results..." -ForegroundColor Gray
Remove-Item results/demo_analysis.json -ErrorAction SilentlyContinue

# Run analysis
Write-Host "🔍 Analyzing traffic file: data/iot23/demo_attacks.csv" -ForegroundColor Green
Write-Host ""

python src/predict.py `
  --input data/iot23/demo_attacks.csv `
  --checkpoint checkpoints/best_multiclass.pt `
  --output results/demo_analysis.json `
  --seq-len 50 `
  --device cpu

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "📊 Sample Results (First 3 Predictions):" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Cyan

$results = Get-Content results/demo_analysis.json | ConvertFrom-Json
$results | Select-Object -First 3 | ForEach-Object {
    Write-Host ""
    Write-Host "Sequence #$($_.sequence_idx):" -ForegroundColor White
    Write-Host "  Attack Type: $($_.attack_type)" -ForegroundColor $(if ($_.attack_type -eq 'Normal') {'Green'} else {'Red'})
    Write-Host "  Confidence: $([math]::Round($_.confidence * 100, 2))%" -ForegroundColor Cyan
    Write-Host "  Severity: $($_.severity)" -ForegroundColor Yellow
    Write-Host "  Timestamp: $($_.timestamp)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "✅ Demo 1 Complete!" -ForegroundColor Green
Write-Host "Full results saved to: results/demo_analysis.json" -ForegroundColor Gray
Write-Host "=" * 80 -ForegroundColor Cyan
