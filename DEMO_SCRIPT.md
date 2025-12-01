# NextGen IDS - Complete Demo Script for Client Presentation

## 🎥 Screen Recording Guide

**Duration**: ~15-20 minutes  
**Goal**: Demonstrate all key features of the NextGen IDS system  
**Audience**: Client stakeholders

---

## 📋 Pre-Recording Checklist

✅ Clean terminal history: `Clear-Host`  
✅ Close unnecessary applications  
✅ Set terminal to readable font size (14-16pt)  
✅ Prepare browser window at `http://localhost:8080`  
✅ Have demo files ready in `demo_files/` folder  
✅ Clear old results: `Remove-Item results/*.json -ErrorAction SilentlyContinue`  
✅ Reset threat database: `echo '[]' > data/threats.json`

---

## 🎬 DEMO SCRIPT

### **PART 1: Introduction & System Overview** (2 min)

#### Talking Points:
*"Hello, I'm demonstrating NextGen IDS - an AI-powered Intrusion Detection System that uses deep learning to detect multiple attack types in real-time. The system features:*
- *Advanced Recurrent Neural Network (A-RNN) architecture*
- *Multi-class attack detection (DDoS, Port Scan, Malware C2, Brute Force, SQL Injection)*
- *Real-time threat analysis with confidence scores*
- *Automated response and remediation capabilities*
- *Phishing detection for URLs and emails*
- *Interactive web dashboard for monitoring"*

#### Commands:
```powershell
# Show project structure
tree /F /A | Select-String -Pattern "src/|checkpoints/|data/" | Select-Object -First 20

# Show available models
Get-ChildItem checkpoints/*.pt | ForEach-Object { $_.Name }
```

#### Expected Output:
```
best_iot23.pt
best_multiclass.pt
best_uploaded.pt
```

---

### **PART 2: Dashboard Launch & Login** (1 min)

#### Talking Points:
*"Let's start the unified dashboard. This provides a single interface for all IDS operations."*

#### Commands:
```powershell
# Start the dashboard
python quick_start.py
```

#### Expected Output:
```
================================================================================
🛡️  NextGen IDS - Quick Start
================================================================================

Starting unified dashboard on http://localhost:8080

📋 Default Login Credentials:
   Username: admin
   Password: admin123
```

#### Browser Actions:
1. Open browser to `http://localhost:8080`
2. Login with `admin` / `admin123`
3. Show dashboard tabs: **Analyze**, **Train**, **Evaluate**, **Monitor**

---

### **PART 3: Real-Time Traffic Analysis** (4 min)

#### Talking Points:
*"First, let's analyze network traffic from a CSV file containing various attack patterns. The system will automatically detect and classify threats."*

#### Commands (New Terminal - keep dashboard running):
```powershell
# Analyze a pre-labeled attack dataset
python src/predict.py `
  --input data/iot23/demo_attacks.csv `
  --checkpoint checkpoints/best_multiclass.pt `
  --output results/demo_analysis.json `
  --seq-len 50 `
  --device cpu
```

#### Expected Output:
```
🔍 Analyzing traffic file: data/iot23/demo_attacks.csv
📦 Using model: checkpoints/best_multiclass.pt

📊 Analysis Results (4353 sequences analyzed):
  DDoS: 892 sequences
  Normal: 1456 sequences
  Brute_Force: 523 sequences
  Port_Scan: 412 sequences
  Malware_C2: 385 sequences
  SQL_Injection: 685 sequences
✅ Predictions saved to results/demo_analysis.json
```

#### Show Results:
```powershell
# Display first 3 predictions
Get-Content results/demo_analysis.json | ConvertFrom-Json | Select-Object -First 3 | ConvertTo-Json -Depth 3
```

#### Browser Actions:
1. In dashboard, go to **Analyze** tab
2. Select "Upload Custom Test Data"
3. Upload `data/iot23/demo_attacks.csv`
4. Click **📊 Analyze Traffic**
5. Show prediction results, confidence scores, and severity levels

---

### **PART 4: PCAP File Analysis** (3 min)

#### Talking Points:
*"The system can also process raw PCAP files captured from network interfaces. It extracts features and performs analysis automatically."*

#### Commands:
```powershell
# Download a sample PCAP (if not already present)
python download_sample_pcap.py
```

```powershell
# Analyze PCAP file
python src/predict.py `
  --input downloads/sample_pcaps/telnet_sample.pcap `
  --checkpoint checkpoints/best_multiclass.pt `
  --output results/pcap_analysis.json `
  --seq-len 30 `
  --device cpu
```

#### Expected Output:
```
🔍 Converting PCAP to features...
✅ Extracted 145 packets → 23 flow features
📊 Analysis Results (16 sequences analyzed):
  Brute_Force: 12 sequences
  Normal: 4 sequences
```

#### Browser Actions:
1. In dashboard **Analyze** tab, upload PCAP file
2. Show automatic feature extraction
3. Display threat detection results

---

### **PART 5: Model Training** (3 min)

#### Talking Points:
*"Let's train a new model on custom data. The system supports both LSTM and advanced A-RNN architectures."*

#### Browser Actions (in **Train** tab):
1. Select dataset: "IoT23"
2. Set parameters:
   - Epochs: `3` (quick demo)
   - Batch Size: `32`
   - Sequence Length: `64`
   - Use A-RNN: ✅ **Checked**
3. Click **🚀 Train Model**
4. Show training progress bar
5. Display training metrics (loss, F1, accuracy)

#### Alternative CLI Command:
```powershell
# Train via command line (faster for demo)
python src/train.py `
  --dataset iot23 `
  --epochs 3 `
  --batch_size 32 `
  --seq_len 64 `
  --use-arnn `
  --save_path checkpoints/demo_trained.pt
```

#### Expected Output:
```
Epoch 1/3: Train Loss=0.4523, Val F1=0.7845
Epoch 2/3: Train Loss=0.3102, Val F1=0.8234
Epoch 3/3: Train Loss=0.2456, Val F1=0.8512
✅ Best model saved to checkpoints/demo_trained.pt
```

---

### **PART 6: Model Evaluation** (3 min)

#### Talking Points:
*"Now let's evaluate the trained model on test data to measure its performance."*

#### Browser Actions (in **Evaluate** tab):
1. Select "Upload Custom Test Data"
2. Upload `data/iot23/demo_attacks.csv`
3. System auto-selects best matching checkpoint
4. Click **📊 Evaluate**
5. Show metrics:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
6. Display confusion matrix

#### CLI Alternative:
```powershell
python src/evaluate.py `
  --dataset iot23 `
  --checkpoint checkpoints/best_multiclass.pt `
  --batch_size 32 `
  --seq_len 64
```

#### Expected Output:
```
Test Accuracy: 0.8761 | F1: 0.8761 | Precision: 0.8823 | Recall: 0.8702
Confusion matrix saved to results/confusion_matrix.png
```

---

### **PART 7: Phishing Detection** (2 min)

#### Talking Points:
*"The system includes ML-based phishing detection for URLs and emails."*

#### Commands:
```powershell
# Test URL detection
python -c "from src.phishing_detector import classify_url; result = classify_url('http://paypal-verify-account.suspicious-site.ru/login'); print(f'URL: {result}')"
```

#### Expected Output:
```
[OK] Loaded phishing detection model from models/phishing_model.pkl
URL: {'is_phishing': True, 'confidence': 0.89, 'risk_level': 'High'}
```

```powershell
# Test email detection
python -c "from src.phishing_detector import classify_email; result = classify_email('URGENT: Your account has been suspended. Click here immediately to verify: http://verify-now.com'); print(f'Email: {result}')"
```

#### Expected Output:
```
Email: {'is_phishing': True, 'confidence': 0.92, 'risk_level': 'Critical'}
```

#### Browser Actions (if API tab exists):
1. Go to Phishing Detection section
2. Test suspicious URL
3. Show risk assessment

---

### **PART 8: Threat Monitoring & Dashboard** (2 min)

#### Talking Points:
*"The dashboard provides real-time threat monitoring with live event streams."*

#### Browser Actions:
1. Navigate to **Monitor** tab
2. Show threat statistics:
   - Total threats detected
   - Active threats
   - Threat distribution by type
3. Display recent threats table with:
   - Attack type
   - Severity
   - Confidence
   - Timestamp
4. Demonstrate filters and search
5. Click on a threat to see detailed explanation

#### Show Threat Database:
```powershell
# Display recent threats
Get-Content data/threats.json | ConvertFrom-Json | Select-Object -Last 5 | Format-Table attack_type, severity, confidence, timestamp -AutoSize
```

---

### **PART 9: Automated Response** (2 min)

#### Talking Points:
*"The system can automatically respond to detected threats by blocking IPs, isolating hosts, or alerting administrators."*

#### Commands:
```powershell
# Simulate threat detection and response
python -c "
from src.auto_response import respond_to_threat
threat = {
    'attack_type': 'DDoS',
    'severity': 'Critical',
    'confidence': 0.95,
    'src_ip': '192.168.1.100'
}
actions = respond_to_threat(threat)
print('Auto-Response Actions:')
for action in actions:
    print(f'  - {action}')
"
```

#### Expected Output:
```
Auto-Response Actions:
  - Block IP: 192.168.1.100
  - Alert administrator
  - Log to blockchain
```

#### Show Response Log:
```powershell
Get-Content results/auto_response_log.json | ConvertFrom-Json | Select-Object -Last 3
```

---

### **PART 10: Blockchain Logging** (1 min)

#### Talking Points:
*"All threat detections are logged to an immutable blockchain for audit trails and compliance."*

#### Commands:
```powershell
# Show blockchain entries
python -c "
from src.blockchain_logger import BlockchainLogger
logger = BlockchainLogger()
chain = logger.get_chain()
print(f'Blockchain Length: {len(chain)} blocks')
print(f'Last Block Hash: {chain[-1][\"hash\"][:32]}...')
print(f'Chain Valid: {logger.validate_chain()}')
"
```

---

### **PART 11: Testing Suite** (1 min)

#### Talking Points:
*"Let's run the automated test suite to verify all components are functioning correctly."*

#### Commands:
```powershell
python src/test_suite.py
```

#### Expected Output:
```
Ran 21 tests in 0.574s

OK

================================================================================
TEST SUMMARY
================================================================================
Tests run: 21
Successes: 21
Failures: 0
Errors: 0
================================================================================
```

---

### **PART 12: Performance & Scalability** (1 min)

#### Talking Points:
*"The system is designed for production deployment with:*
- *Batch processing for high-throughput analysis*
- *GPU acceleration support*
- *Horizontal scaling via multiple workers*
- *Low latency (<100ms per prediction)"*

#### Commands:
```powershell
# Show model metadata and capabilities
python -c "
import torch
ckpt = torch.load('checkpoints/best_multiclass.pt', map_location='cpu')
meta = ckpt['meta']
print('Model Specifications:')
print(f'  Input Features: {meta[\"input_dim\"]}')
print(f'  Attack Classes: {meta[\"num_classes\"]}')
print(f'  F1 Score: {meta[\"f1\"]:.4f}')
print(f'  Parameters: {sum(p.numel() for p in torch.nn.Module().parameters() if hasattr(p, \"numel\"))} (approx)')
"
```

---

## 🎯 Key Points to Emphasize

### Technical Strengths:
- ✅ **Multi-class detection**: Not just binary (attack/normal), but specific attack types
- ✅ **Deep learning architecture**: A-RNN with attention mechanism
- ✅ **Auto-selection**: Intelligent checkpoint matching based on data
- ✅ **Real-time processing**: Live threat detection and response
- ✅ **Multiple input formats**: CSV, PCAP, real-time streams
- ✅ **Explainable AI**: SHAP values and feature importance

### Business Benefits:
- 🎯 **Reduced response time**: Automated threat mitigation
- 🎯 **Compliance**: Blockchain audit trail
- 🎯 **Cost savings**: Fewer false positives with 87%+ accuracy
- 🎯 **Scalability**: Cloud-ready architecture
- 🎯 **Easy deployment**: Docker support, simple setup

---

## 📊 Demo Files Reference

### Files to Use in Demo:

1. **Traffic Analysis**: `data/iot23/demo_attacks.csv`
   - 4402 samples, 6 attack classes
   - Pre-labeled for verification

2. **PCAP Analysis**: `downloads/sample_pcaps/telnet_sample.pcap`
   - Real network capture
   - Contains brute force attempts

3. **Training Data**: `data/iot23/multiclass_attacks.csv`
   - Synthetic multi-class dataset
   - 3000 samples for quick training

4. **Evaluation**: Same as analysis files
   - System auto-selects best checkpoint

---

## 🎥 Recording Tips

### Visual Flow:
1. **Start with dashboard** - Show clean, professional UI
2. **Demo each feature tab** - Systematic walkthrough
3. **Mix UI and CLI** - Show flexibility
4. **Show real results** - Actual predictions, not mock data
5. **End with test suite** - Prove reliability

### Voice-over Script:
- Speak clearly and confidently
- Pause after each major feature (allow time to see results)
- Emphasize accuracy metrics (87%+ F1)
- Mention real-world applications (enterprise networks, cloud security)

### Screen Setup:
- **Browser**: Left side (dashboard)
- **Terminal**: Right side (commands)
- Use split-screen for simultaneous views
- Zoom in on important results

---

## 🚀 Quick Start Commands (Copy-Paste Ready)

### Terminal 1 (Dashboard):
```powershell
python quick_start.py
```

### Terminal 2 (Commands):
```powershell
# Set working directory
cd C:\Users\Nachi\OneDrive\Desktop\NewGenIDS-main\NewGenIDS-main

# Analysis
python src/predict.py --input data/iot23/demo_attacks.csv --checkpoint checkpoints/best_multiclass.pt --output results/demo_analysis.json --seq-len 50

# Training (quick demo)
python src/train.py --dataset iot23 --epochs 3 --batch_size 32 --use-arnn --save_path checkpoints/demo_trained.pt

# Evaluation
python src/evaluate.py --dataset iot23 --checkpoint checkpoints/best_multiclass.pt

# Tests
python src/test_suite.py

# Phishing test
python -c "from src.phishing_detector import classify_url; print(classify_url('http://paypal-verify.suspicious.com'))"
```

---

## ✅ Post-Demo Checklist

- [ ] Stop dashboard server (Ctrl+C)
- [ ] Show generated reports in `results/`
- [ ] Display threat database `data/threats.json`
- [ ] Mention deployment options (Docker, cloud)
- [ ] Offer Q&A / technical deep-dive

---

## 📝 Troubleshooting

### If something fails during recording:

**Dashboard won't start**:
```powershell
# Check if port is in use
netstat -ano | findstr :8080
# Kill process if needed
taskkill /PID <PID> /F
```

**Model fails to load**:
```powershell
# Verify checkpoint exists
Test-Path checkpoints/best_multiclass.pt
# Re-download if needed
python download_checkpoints.py
```

**Predictions look wrong**:
- Use `--seq-len 30` for smaller files
- Ensure CSV has numeric features
- Check auto-selection picked correct checkpoint

---

## 🎬 Final Notes

- **Total recording time**: 15-20 minutes
- **Practice run first**: Do a dry run to catch issues
- **Have backup**: Keep extra terminal open for emergencies
- **Stay calm**: If something breaks, acknowledge and skip to next feature
- **End strong**: Emphasize test results and reliability

**Good luck with your demo! 🚀**
