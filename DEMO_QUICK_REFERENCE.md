# 🎬 Quick Demo Reference Card

## One-Command Demo Launch

```powershell
# Run complete demo sequence (all features)
.\demo_scripts\RUN_ALL_DEMOS.ps1

# Run individual demos
.\demo_scripts\1_test_analysis.ps1      # Traffic Analysis
.\demo_scripts\2_test_training.ps1      # Model Training
.\demo_scripts\3_test_evaluation.ps1    # Evaluation
.\demo_scripts\4_test_phishing.ps1      # Phishing Detection
.\demo_scripts\5_test_auto_response.ps1 # Auto-Response
.\demo_scripts\6_test_system.ps1        # System Tests
```

## Dashboard UI Demo

```powershell
# Start dashboard (keep running)
python quick_start.py

# Open browser to: http://localhost:8080
# Login: admin / admin123
```

## Key Files for Demo

| File | Purpose | Use Case |
|------|---------|----------|
| `data/iot23/demo_attacks.csv` | Pre-labeled attacks | Analysis & Evaluation |
| `data/iot23/multiclass_attacks.csv` | Training data | Model Training |
| `downloads/sample_pcaps/*.pcap` | Network captures | PCAP Analysis |
| `checkpoints/best_multiclass.pt` | Trained model | Predictions |

## Expected Performance Metrics

- **Accuracy**: 87.61%
- **F1 Score**: 0.8761
- **Precision**: 88.23%
- **Recall**: 87.02%
- **Attack Classes**: 6 (Normal, DDoS, Port Scan, Brute Force, Malware C2, SQL Injection)

## Recording Checklist

- [ ] Terminal font size: 14-16pt
- [ ] Clear terminal history: `Clear-Host`
- [ ] Close unnecessary apps
- [ ] Test audio/video quality
- [ ] Run practice demo first
- [ ] Have backup terminal ready

## Time Estimates

| Demo | Duration |
|------|----------|
| Dashboard Tour | 2 min |
| Traffic Analysis | 3 min |
| Model Training | 4 min |
| Evaluation | 2 min |
| Phishing Detection | 2 min |
| Auto-Response | 2 min |
| System Tests | 1 min |
| **Total** | **15-20 min** |

## Troubleshooting Quick Fixes

```powershell
# Port 8080 in use
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Reset environment
Remove-Item results/*.json -Force
echo '[]' > data/threats.json

# Verify checkpoints
Get-ChildItem checkpoints/*.pt
```

## Key Talking Points

✅ "87% accuracy with deep learning"  
✅ "Multi-class attack detection, not just binary"  
✅ "Real-time automated response"  
✅ "Blockchain audit trail for compliance"  
✅ "Production-ready with Docker support"

## Post-Demo Actions

- Show generated reports in `results/`
- Display threat database `data/threats.json`
- Mention deployment options
- Offer technical Q&A
