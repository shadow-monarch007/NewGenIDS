# Demo Test Script 3: Model Evaluation
# Evaluates model performance on test data

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "DEMO 3: Model Evaluation" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

Write-Host "📊 Evaluating model on test dataset..." -ForegroundColor Green
Write-Host ""

python src/evaluate.py `
  --dataset iot23 `
  --checkpoint checkpoints/best_multiclass.pt `
  --batch_size 32 `
  --seq_len 64

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "📈 Performance Metrics Summary:" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Cyan

if (Test-Path results/metrics.csv) {
    $metrics = Import-Csv results/metrics.csv | Select-Object -Last 1
    Write-Host ""
    Write-Host "  Accuracy:  $([math]::Round($metrics.test_acc * 100, 2))%" -ForegroundColor Green
    Write-Host "  Precision: $([math]::Round($metrics.test_precision * 100, 2))%" -ForegroundColor Cyan
    Write-Host "  Recall:    $([math]::Round($metrics.test_recall * 100, 2))%" -ForegroundColor Cyan
    Write-Host "  F1 Score:  $([math]::Round($metrics.test_f1 * 100, 2))%" -ForegroundColor Yellow
    Write-Host ""
}

if (Test-Path results/confusion_matrix.png) {
    Write-Host "📊 Confusion Matrix saved to: results/confusion_matrix.png" -ForegroundColor Gray
    Write-Host "   (Open this file to visualize classification performance)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "✅ Demo 3 Complete!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
