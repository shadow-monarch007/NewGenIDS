# NextGen IDS - User Guide

## What is This Project?

NextGen IDS is an **AI-powered Intrusion Detection System** that monitors network traffic and automatically detects cyber attacks in real-time. Think of it as a smart security guard that watches your network 24/7 and alerts you when something suspicious happens.

---

## How It Works (Simple Architecture)

### 1. **Brain of the System (AI Model)**
- Uses a **Neural Network** (LSTM + CNN) to learn patterns of normal and malicious traffic
- Trained on 305,474 parameters to recognize different attack types
- Can identify: DDoS attacks, Port Scans, SQL Injection, Malware, Brute Force, and Normal traffic

### 2. **Traffic Analysis**
```
Your Network → Capture Traffic → AI Analysis → Detect Threats → Alert You
```

### 3. **Dashboard (Web Interface)**
- Professional web interface running on http://localhost:8080
- 8 main sections:
  - **Dashboard**: Real-time threat overview
  - **Traffic Analysis**: Upload files to analyze
  - **Dataset Converter**: Convert different dataset formats
  - **Phishing Detection**: Check URLs and emails
  - **Log Analysis**: Analyze security logs
  - **Model Training**: Train AI with your own data
  - **Remediation**: Automated response actions
  - **Blockchain**: Tamper-proof audit trail

---

## Quick Start (3 Steps)

### Step 1: Start the System
```powershell
python quick_start.py
```

### Step 2: Open Dashboard
Open your browser and go to: **http://localhost:8080**

### Step 3: Login
```
Username: admin
Password: admin123
```

That's it! You're in.

---

## How to Use Each Feature

### 🎯 **Traffic Analysis** (Most Common Use)

**What it does**: Analyzes network traffic files to detect attacks

**How to use**:
1. Click **"Traffic Analysis"** tab
2. Click **"Choose File"** button
3. Select a CSV or PCAP file
4. Click **"Analyze Traffic"**
5. Wait 10-30 seconds
6. See results:
   - Attack type detected
   - Confidence level (how sure the AI is)
   - Severity (Critical, High, Medium, Low)
   - AI explanation with mitigation steps

**Example Files** (already included):
- `data/iot23/demo_samples/normal.csv` - Normal traffic
- `data/iot23/demo_samples/ddos.csv` - DDoS attack
- `data/iot23/demo_samples/port_scan.csv` - Port scanning
- `data/iot23/demo_samples/sql_injection.csv` - SQL injection

---

### 🧠 **Model Training** (Advanced Users)

**What it does**: Train the AI with your own network traffic data

**How to use**:

#### Option 1: Use Built-in Dataset
1. Go to **"Model Training"** tab
2. Keep "Built-in Dataset (IoT-23)" selected
3. Set Epochs: `5` (higher = better but slower)
4. Set Batch Size: `32`
5. Click **"Start Training"**
6. Wait 5-10 minutes
7. See training results

#### Option 2: Upload Your Own Data
1. Go to **"Model Training"** tab
2. Click dropdown: Select **"Upload Custom Dataset"**
3. Click **"Choose File"** (now visible)
4. Upload your CSV file with these columns:
   - `src_port`, `dst_port`, `protocol`, `flow_duration`
   - `tot_fwd_pkts`, `tot_bwd_pkts`, `totlen_fwd_pkts`, `totlen_bwd_pkts`
   - `fwd_pkt_len_max`, `fwd_pkt_len_min`, `fwd_pkt_len_mean`, `fwd_pkt_len_std`
   - `bwd_pkt_len_max`, `bwd_pkt_len_min`, `bwd_pkt_len_mean`, `bwd_pkt_len_std`
   - `flow_byts_s`, `flow_pkts_s`, `flow_iat_mean`, `flow_iat_std`
   - **`label`** (attack type: Normal, DDoS, Port_Scan, etc.)
5. Set Epochs and Batch Size
6. Click **"Start Training"**
7. New model saved as `checkpoints/best_uploaded.pt`

---

### 📊 **Model Evaluation** (Check Accuracy)

**What it does**: Tests how accurate the AI model is

**How to use**:

#### Test Built-in Model
1. Go to **"Model Training"** tab → **"Evaluate Model"** section
2. Keep "Built-in Dataset (IoT-23)" selected
3. Checkpoint: `best_iot23.pt` (current model)
4. Click **"Evaluate"**
5. See results:
   - Accuracy (e.g., 86%)
   - Precision, Recall, F1 Score
   - Confusion Matrix (visual accuracy chart)

#### Test with Your Data
1. Select **"Upload Custom Test Data"** from dropdown
2. Upload CSV file (same format as training)
3. Choose checkpoint to test
4. Click **"Evaluate"**

---

### 🔄 **Dataset Converter**

**What it does**: Converts different dataset formats to work with the system

**Supported Formats**:
- KDD Cup 99
- NSL-KDD
- CICIDS 2017
- UNSW-NB15

**How to use**:
1. Go to **"Dataset Converter"** tab
2. Select source format (e.g., "KDD Cup 99")
3. Upload your dataset file
4. Click **"Convert Dataset"**
5. Download converted file
6. Use converted file in Traffic Analysis or Training

---

### 🎣 **Phishing Detection**

**What it does**: Checks if URLs or emails are phishing attempts

**Check a URL**:
1. Go to **"Phishing Detection"** tab
2. Enter URL (e.g., `http://suspicious-site.com`)
3. Click **"Analyze URL"**
4. See if it's Safe or Phishing

**Check an Email**:
1. Paste email content in text box
2. Click **"Analyze Email"**
3. See phishing probability

---

### 📝 **Log Analysis**

**What it does**: Analyzes security logs for threats

**How to use**:
1. Go to **"Log Analysis"** tab
2. Select log source (Apache, Nginx, SSH, etc.)
3. Paste log entries (one per line)
4. Click **"Analyze Logs"**
5. See detected threats and severity

---

## Understanding Results

### Attack Types
- **Normal**: Safe traffic, no threat
- **DDoS**: Distributed Denial of Service attack (flooding)
- **Port_Scan**: Attacker scanning for open ports
- **SQL_Injection**: Database attack
- **Brute_Force**: Password guessing attack
- **Malware**: Malicious software communication

### Severity Levels
- 🔴 **Critical**: Immediate action required
- 🟠 **High**: Serious threat, act soon
- 🟡 **Medium**: Moderate risk, investigate
- 🟢 **Low**: Minor concern, monitor

### Confidence Score
- **90-100%**: AI is very sure
- **70-89%**: AI is confident
- **50-69%**: AI is uncertain
- **Below 50%**: Might be false positive

---

## Common Workflows

### Workflow 1: Daily Traffic Monitoring
```
1. Capture traffic → Save as CSV/PCAP
2. Open Dashboard → Traffic Analysis
3. Upload file → Click Analyze
4. Review threats → Take action if needed
```

### Workflow 2: Training with Custom Data
```
1. Collect your network traffic (labeled)
2. Format as CSV with required columns
3. Model Training → Upload Custom Dataset
4. Train for 10-20 epochs
5. Evaluate accuracy
6. Use new model for analysis
```

### Workflow 3: Converting Datasets
```
1. Have KDD/CICIDS/UNSW dataset
2. Dataset Converter → Select format
3. Upload file → Convert
4. Download IoT-23 format
5. Use in Training or Analysis
```

---

## File Locations

### Important Files
```
checkpoints/
  └── best_iot23.pt          ← Current AI model (retrained, 86% accuracy)
  └── best_iot23_retrained.pt ← Backup of retrained model

data/
  └── threats.json            ← Stored threat detections
  └── iot23/
      └── demo_samples/       ← Example attack files

uploads/
  └── uploaded/               ← Your uploaded files stored here

templates/
  └── dashboard.html          ← Web interface

src/
  └── dashboard_unified.py    ← Main server
  └── predict.py              ← AI prediction logic
  └── train.py                ← Model training
```

---

## Troubleshooting

### Dashboard Won't Start
```powershell
# Kill existing Python processes
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Start fresh
python quick_start.py
```

### Analysis Takes Too Long
- Use smaller files (< 10,000 rows for testing)
- System processes 64 samples at a time
- Large files (22K+ rows) take 15-30 seconds

### Upload Button Not Showing
1. Go to Model Training tab
2. Click the **dropdown menu** "Training Data Source"
3. Select **"Upload Custom Dataset"**
4. File input field will appear

### Model Predicts Everything as DDoS
- Old model was overtrained (fixed now)
- Current model: `best_iot23.pt` (86.38% F1 score)
- If issue persists, retrain model

---

## Performance Tips

### For Faster Analysis
- Use CSV instead of PCAP (PCAP needs conversion)
- Limit files to 10,000 rows for quick tests
- System uses batch processing (64 samples/batch)

### For Better Accuracy
- Train with diverse attack samples
- Use 20-30 epochs for training
- Balance dataset (equal Normal and Attack samples)
- Validate with separate test set

---

## System Requirements

### Minimum
- Python 3.8+
- 4 GB RAM
- Windows/Linux/Mac

### Recommended
- Python 3.10+
- 8 GB RAM
- GPU (CUDA) for faster training
- SSD for faster file processing

---

## Default Credentials

### Admin Account
```
Username: admin
Password: admin123
```

### Demo Account
```
Username: demo
Password: demo123
```

**⚠️ Security Note**: Change these passwords in production!

---

## Quick Commands

### Start Dashboard
```powershell
python quick_start.py
```

### Train Model (Command Line)
```powershell
python src/train.py --dataset iot23 --epochs 20 --batch_size 32
```

### Evaluate Model (Command Line)
```powershell
python src/evaluate.py --dataset iot23 --checkpoint best_iot23.pt
```

### Analyze File (Command Line)
```powershell
python src/predict.py --csv data/iot23/demo_samples/ddos.csv
```

---

## Support & Documentation

### Quick Reference Files
- `READY_TO_DEMO.md` - Demo instructions
- `SETUP_GUIDE.txt` - Installation guide
- `DATASETS_QUICK_REFERENCE.txt` - Dataset info
- `MODEL_RETRAINED.md` - Model retraining details

### Getting Help
1. Check terminal output for error messages
2. Verify all dependencies installed: `pip install -r requirements.txt`
3. Ensure virtual environment activated: `.\.venv\Scripts\Activate.ps1`

---

## What Makes This System Special

### ✅ Real-time Detection
- Analyzes traffic instantly
- Automatic threat alerts
- Live dashboard updates

### ✅ AI-Powered
- Deep learning neural network
- Learns from data (not just rules)
- Adapts to new attack patterns

### ✅ User-Friendly
- Web-based interface (no command line needed)
- Upload files → Get results
- Clear explanations for detected threats

### ✅ Customizable
- Train with your own data
- Upload custom datasets
- Adjust detection thresholds

### ✅ Professional Features
- Blockchain audit trail (tamper-proof logs)
- Automated remediation actions
- Phishing detection
- Multi-format dataset support

---

## Next Steps

### Beginner Path
1. ✅ Start dashboard
2. ✅ Test with demo files (`data/iot23/demo_samples/`)
3. ✅ Understand attack types
4. ⬜ Try with your own traffic

### Intermediate Path
1. ✅ Collect your network traffic
2. ✅ Convert to IoT-23 format
3. ✅ Train custom model
4. ⬜ Evaluate accuracy
5. ⬜ Deploy in test environment

### Advanced Path
1. ✅ Integrate with SIEM tools
2. ✅ Set up automated responses
3. ✅ Fine-tune model parameters
4. ⬜ Deploy in production
5. ⬜ Continuous retraining

---

**🎯 Remember**: This is a security tool. Always test in a safe environment before using with real production traffic.

**📧 Questions?** Review the included documentation files or check the code comments in `src/` folder.

---

*Last Updated: November 2025*
*Model Version: best_iot23.pt (86.38% F1 Score)*
