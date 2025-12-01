# Start unified dashboard on port 8080 with a clean slate and open browser
param(
    [switch]$NoFreshStart,
    [switch]$NoBrowser,
    [switch]$Seed,
    [switch]$NoPersist
)

Set-Location $PSScriptRoot

if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
}

if (-not $NoFreshStart) { $env:IDS_FRESH_START = '1' } else { Remove-Item Env:\IDS_FRESH_START -ErrorAction SilentlyContinue }
if ($NoPersist) { $env:IDS_DISABLE_PERSIST = '1' } else { Remove-Item Env:\IDS_DISABLE_PERSIST -ErrorAction SilentlyContinue }
$env:DASHBOARD_PORT = '8080'

Write-Host "Starting unified dashboard on http://localhost:8080" -ForegroundColor Cyan
Start-Process -NoNewWindow -FilePath python -ArgumentList "src\\dashboard_unified.py" | Out-Null
Start-Sleep -Seconds 2

if (-not $NoBrowser) {
    Start-Process "http://localhost:8080"
}

if ($Seed) {
    Write-Host "Seeding with real demo samples..." -ForegroundColor Yellow
    python scripts\seed_from_samples.py
}
