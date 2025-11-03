# Next-Gen IDS Dashboard Launcher
# Run this script to start the dashboard

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "🛡️  Next-Gen IDS Dashboard" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Check if model exists
if (!(Test-Path "checkpoints/best_iot23.pt")) {
    Write-Host "⚠️  Model not found!" -ForegroundColor Yellow
    Write-Host "   Run this first: python src/train.py --dataset iot23 --epochs 5 --batch_size 32 --seq_len 64 --save_path checkpoints/best_iot23.pt" -ForegroundColor Yellow
    Write-Host ""
    exit
}

Write-Host "✅ Model found: checkpoints/best_iot23.pt" -ForegroundColor Green
Write-Host "🚀 Starting dashboard server..." -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Dashboard will open at: http://localhost:5000" -ForegroundColor Yellow
Write-Host "🔍 Upload unlabeled CSV files to detect threats!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Start the dashboard
python src/dashboard_live.py
