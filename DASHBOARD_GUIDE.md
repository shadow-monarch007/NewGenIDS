# 🌐 Web Dashboard Quick Start Guide

## What is the Dashboard?

A **user-friendly web interface** where you can:
1. ✅ Upload CSV files (network traffic data)
2. ✅ Train the AI model with one click
3. ✅ See results (accuracy, confusion matrix)
4. ✅ Get AI-powered explanations of detected threats
5. ✅ Receive mitigation recommendations

---

## 🚀 How to Run the Dashboard

### Step 1: Install Flask
```powershell
cd nextgen_ids
.\.venv\Scripts\Activate.ps1
pip install flask
```

### Step 2: Set PYTHONPATH
```powershell
$env:PYTHONPATH = (Get-Location).Path
```

### Step 3: Start the Dashboard
```powershell
python src\dashboard.py
```

### Step 4: Open in Browser
Navigate to: **http://localhost:5000**

---

## 📖 How to Use the Dashboard

### 1️⃣ Upload Your Data
- Click the **upload area**
- Select a CSV file with network traffic data
- File should have numeric features (packet sizes, ports, protocols, etc.)
- Optional: Include a `label` column with attack types (0=normal, 1=attack)

### 2️⃣ Configure Training
- **Epochs**: How many times the AI sees all data (5-20 recommended)
- **Batch Size**: How many samples processed at once (32 for 8GB RAM)
- **Sequence Length**: Time window size (64-100 recommended)

### 3️⃣ Train the Model
- Click **"Start Training"**
- Watch progress bar
- Wait for "Training complete!" message

### 4️⃣ Evaluate Performance
- Click **"Run Evaluation"**
- See accuracy, precision, recall, F1 score
- View confusion matrix

### 5️⃣ Get AI Explanations
- Click **"Generate Threat Analysis"**
- See:
  - What type of attack was detected
  - Why the AI flagged it (key indicators)
  - How to fix/mitigate the threat
  - Severity level (Critical/Medium/Low)

---

## 🎯 Example Workflow

```
1. Upload: iot23_traffic.csv (10,000 rows)
   ↓
2. Train: 10 epochs, batch 32 → 95% accuracy
   ↓
3. Evaluate: See confusion matrix, F1: 94%
   ↓
4. AI Explains: "DDoS attack detected"
   - Indicator: High packet rate (10,000 pkt/s)
   - Mitigation: Enable rate limiting, deploy DDoS protection
```

---

## 🤖 AI Explanations Feature

The dashboard provides **intelligent threat analysis** for common attack types:

### DDoS (Distributed Denial of Service)
- **What it is**: Flooding network with traffic to crash services
- **Indicators**: High packet rate, multiple source IPs
- **Mitigation**: Rate limiting, DDoS protection (Cloudflare), block IPs

### Port Scan
- **What it is**: Attacker probing for open ports/vulnerabilities
- **Indicators**: Sequential port access, rapid connections
- **Mitigation**: Close unnecessary ports, enable IDS alerts, use fail2ban

### Malware C2 (Command & Control)
- **What it is**: Infected device calling home to attacker
- **Indicators**: Unusual outbound connections, periodic beaconing
- **Mitigation**: Isolate device, run antivirus, block C2 servers

### Unknown Anomalies
- **What it is**: Suspicious behavior not matching known patterns
- **Indicators**: Based on SHAP feature importance
- **Mitigation**: Manual investigation, correlation with logs

---

## 🔧 Advanced: Integrate Real AI (OpenAI GPT)

To get **even smarter explanations**, you can integrate OpenAI's API:

### Step 1: Install OpenAI SDK
```powershell
pip install openai
```

### Step 2: Get API Key
- Sign up at https://platform.openai.com
- Create an API key
- Set environment variable:
```powershell
$env:OPENAI_API_KEY = "your-api-key-here"
```

### Step 3: Update dashboard.py
Replace the `generate_ai_explanation` function with:

```python
import openai

def generate_ai_explanation(intrusion_type, features, confidence):
    prompt = f"""
    You are a cybersecurity expert. A network intrusion was detected:
    
    Type: {intrusion_type}
    Confidence: {confidence:.1%}
    Key Features: {features}
    
    Provide:
    1. A clear explanation of this attack
    2. 3-5 key indicators
    3. 5-7 specific mitigation steps
    4. Severity assessment (Critical/Medium/Low)
    
    Format as JSON.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.choices[0].message.content)
```

Now the AI will generate **custom, context-aware explanations** for each threat!

---

## 📊 Dashboard Screenshots (What You'll See)

### Upload Section
```
┌────────────────────────────────┐
│   📤 Click to upload CSV       │
│   Supported: .csv files         │
└────────────────────────────────┘

File Statistics:
- Filename: traffic_data.csv
- Rows: 10,000
- Columns: 25
- Has Label: ✓ Yes
```

### Training Section
```
Training Progress:
[████████████████████] 100%

✓ Training complete! Best F1: 94.2%
```

### Results Section
```
┌────────────┬────────────┬────────────┬────────────┐
│ Accuracy   │ Precision  │ Recall     │ F1 Score   │
│   95.2%    │   94.8%    │   93.5%    │   94.2%    │
└────────────┴────────────┴────────────┴────────────┘

Confusion Matrix:
[Image showing 2x2 grid of predictions vs actual]
```

### AI Explanation Section
```
⚠️ DDoS - Critical Severity

Description:
Distributed Denial of Service attack detected.
Multiple sources flooding the network.

🎯 Detection Confidence: 92.0%

Key Indicators:
• Abnormally high packet rate
• Multiple source IPs targeting single destination
• Repetitive packet patterns

🛠️ Mitigation Steps:
1. Enable rate limiting on firewall
2. Deploy DDoS protection service (Cloudflare, AWS Shield)
3. Block malicious IP ranges
4. Scale infrastructure to handle traffic spike
```

---

## 🛠️ Troubleshooting

### "ModuleNotFoundError: No module named 'flask'"
```powershell
pip install flask
```

### "ModuleNotFoundError: No module named 'src'"
```powershell
$env:PYTHONPATH = (Get-Location).Path
```

### Dashboard not loading
- Check if port 5000 is already in use
- Try a different port:
```python
# In dashboard.py, change last line to:
app.run(debug=True, host='0.0.0.0', port=8080)
```
Then access: http://localhost:8080

### Training takes too long
- Reduce epochs to 3-5
- Reduce batch size to 16
- Use smaller dataset (< 10,000 rows)

---

## 🎓 For Your Project Presentation

**Demo Flow for Professors/Examiners:**

1. **Open dashboard** in browser
2. **Upload sample CSV** (IoT-23 or synthetic data)
3. **Show file stats** appearing automatically
4. **Click "Start Training"** → explain hybrid CNN+LSTM architecture
5. **Show progress bar** → mention GPU/CPU optimization
6. **Results appear** → explain metrics (accuracy, F1, confusion matrix)
7. **Generate AI explanation** → highlight explainability + mitigation
8. **Show blockchain logging** (optional: mention tamper-proof audit trail)

**Key Points to Mention:**
- ✅ End-to-end pipeline from raw data to actionable insights
- ✅ Explainable AI (SHAP + natural language explanations)
- ✅ Production-ready with web interface
- ✅ Real-world mitigation recommendations
- ✅ Scalable architecture (works on laptop, scales to server)

---

## 📝 Next Steps

1. **Test with real data**: Upload IoT-23 or BETH dataset CSVs
2. **Customize explanations**: Add more attack types to `generate_ai_explanation`
3. **Integrate OpenAI**: For smarter, context-aware threat analysis
4. **Add authentication**: Secure the dashboard with login (Flask-Login)
5. **Deploy online**: Host on Heroku, AWS, or Azure for remote access

---

**The dashboard is ready to use! Start it now and impress your evaluators! 🚀**
