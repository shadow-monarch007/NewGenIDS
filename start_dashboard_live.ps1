# Start Dashboard with Live Reload
# Run: .\start_dashboard_live.ps1

Write-Host "🚀 Starting Next-Gen IDS Dashboard..." -ForegroundColor Cyan
Write-Host "=" * 60

# Set Flask environment variables for development
$env:FLASK_APP = "src/dashboard.py"
$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = "1"

# Change to project directory
Set-Location -Path $PSScriptRoot

Write-Host "`n📋 Dashboard Configuration:" -ForegroundColor Green
Write-Host "  • Flask App: src/dashboard.py"
Write-Host "  • Environment: Development"
Write-Host "  • Debug Mode: Enabled (with auto-reload)"
Write-Host "  • Host: 0.0.0.0 (accessible from network)"
Write-Host "  • Port: 5000"

Write-Host "`n🌐 Dashboard will be available at:" -ForegroundColor Yellow
Write-Host "  • Local:   http://localhost:5000" -ForegroundColor White
Write-Host "  • Network: http://$(hostname):5000" -ForegroundColor White

Write-Host "`n⚡ Features Enabled:" -ForegroundColor Cyan
Write-Host "  ✓ Live Reload (auto-restart on code changes)"
Write-Host "  ✓ Debug Mode (detailed error pages)"
Write-Host "  ✓ Interactive Debugger"

Write-Host "`n🔄 Starting server..." -ForegroundColor Green
Write-Host "   Press Ctrl+C to stop the server`n" -ForegroundColor Yellow

# Start Flask with live reload
& .venv\Scripts\python.exe -m flask run --host=0.0.0.0 --port=5000 --reload --debugger
