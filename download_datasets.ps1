# Download and Setup Real IDS Datasets
# Run: .\download_datasets.ps1

Write-Host "🌐 IDS Dataset Downloader" -ForegroundColor Cyan
Write-Host "=" * 60

# Create data directory
$dataDir = "data"
if (!(Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
}

# Function to download file
function Download-Dataset {
    param (
        [string]$Url,
        [string]$OutputPath,
        [string]$DatasetName
    )
    
    Write-Host "`n📥 Downloading $DatasetName..." -ForegroundColor Green
    
    if (Test-Path $OutputPath) {
        Write-Host "✅ Already exists: $OutputPath" -ForegroundColor Yellow
        return
    }
    
    try {
        # Use WebClient for faster downloads
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($Url, $OutputPath)
        Write-Host "✅ Downloaded: $OutputPath" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to download: $_" -ForegroundColor Red
    }
}

# Show menu
Write-Host "`nAvailable Datasets:" -ForegroundColor Cyan
Write-Host "1. NSL-KDD (20 MB) - Quick testing" -ForegroundColor White
Write-Host "2. UNSW-NB15 (~2 GB) - Recommended for your project" -ForegroundColor White
Write-Host "3. CIC-IDS-2017 (~3 GB) - Most popular (manual download)" -ForegroundColor White
Write-Host "4. Download sample data (for testing)" -ForegroundColor White
Write-Host "5. All of the above" -ForegroundColor White

$choice = Read-Host "`nEnter choice (1-5)"

switch ($choice) {
    "1" {
        # NSL-KDD
        $nslDir = "$dataDir\nsl_kdd"
        New-Item -ItemType Directory -Path $nslDir -Force | Out-Null
        
        Download-Dataset `
            -Url "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt" `
            -OutputPath "$nslDir\KDDTrain+.txt" `
            -DatasetName "NSL-KDD Training"
        
        Download-Dataset `
            -Url "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt" `
            -OutputPath "$nslDir\KDDTest+.txt" `
            -DatasetName "NSL-KDD Testing"
    }
    
    "2" {
        # UNSW-NB15
        $unswDir = "$dataDir\unsw_nb15"
        New-Item -ItemType Directory -Path $unswDir -Force | Out-Null
        
        Write-Host "`n⚠️  UNSW-NB15 is large (~2GB). This may take a while..." -ForegroundColor Yellow
        
        Download-Dataset `
            -Url "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download" `
            -OutputPath "$unswDir\UNSW-NB15_1.csv" `
            -DatasetName "UNSW-NB15 Part 1"
        
        Download-Dataset `
            -Url "https://cloudstor.aarnet.edu.au/plus/s/M63LvYQFjvf9N6V/download" `
            -OutputPath "$unswDir\UNSW-NB15_2.csv" `
            -DatasetName "UNSW-NB15 Part 2"
        
        Download-Dataset `
            -Url "https://cloudstor.aarnet.edu.au/plus/s/hdAG9wlr6fRzh1O/download" `
            -OutputPath "$unswDir\UNSW_NB15_testing-set.csv" `
            -DatasetName "UNSW-NB15 Test Set"
    }
    
    "3" {
        Write-Host "`n📋 CIC-IDS-2017 Manual Download Instructions:" -ForegroundColor Yellow
        Write-Host "1. Visit: https://www.kaggle.com/datasets/cicdataset/cicids2017"
        Write-Host "2. Download the CSV files"
        Write-Host "3. Extract to: data\cicids2017\"
        Write-Host "4. Files should be named: Monday-WorkingHours.csv, Tuesday-WorkingHours.csv, etc."
        
        # Open browser
        Start-Process "https://www.kaggle.com/datasets/cicdataset/cicids2017"
    }
    
    "4" {
        Write-Host "`n✅ You already have demo data!" -ForegroundColor Green
        Write-Host "Location: data\iot23\demo_attacks.csv (4,400 samples)"
        Write-Host "Generated with 6 attack types + normal traffic"
    }
    
    "5" {
        # Download all
        & $MyInvocation.MyCommand.Path -AutoDownload
    }
    
    default {
        Write-Host "❌ Invalid choice!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n✅ Dataset setup complete!" -ForegroundColor Green
Write-Host "`n📚 Dataset Information:" -ForegroundColor Cyan
Write-Host "=" * 60

Write-Host "`nNSL-KDD:"
Write-Host "  • Location: data\nsl_kdd\"
Write-Host "  • Size: ~20 MB"
Write-Host "  • Attack Types: DoS, Probe, R2L, U2R"
Write-Host "  • Use with: --dataset nsl_kdd"

Write-Host "`nUNSW-NB15:"
Write-Host "  • Location: data\unsw_nb15\"
Write-Host "  • Size: ~2 GB"
Write-Host "  • Attack Types: 9 categories (Fuzzers, DoS, Exploits, etc.)"
Write-Host "  • Use with: --dataset unsw_nb15"

Write-Host "`nCIC-IDS-2017:"
Write-Host "  • Location: data\cicids2017\"
Write-Host "  • Size: ~3 GB"
Write-Host "  • Attack Types: DDoS, DoS, Brute Force, Web Attacks, Botnet"
Write-Host "  • Use with: --dataset cicids2017"

Write-Host "`n🚀 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Run preprocessing (if needed): python src/preprocess.py --dataset <name>"
Write-Host "2. Train your model: python src/train.py --dataset <name> --epochs 5 --use-arnn"
Write-Host "3. Evaluate: python src/evaluate.py --dataset <name>"
Write-Host "`n" -NoNewline
