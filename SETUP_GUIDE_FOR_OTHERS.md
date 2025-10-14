# 🚀 Setup Guide - For New Users

**Version:** 1.0  
**Last Updated:** October 14, 2025

This guide will help you set up and run the Next-Gen IDS project after receiving the zip file.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup (Automated)](#quick-setup-automated)
3. [Manual Setup (Step by Step)](#manual-setup-step-by-step)
4. [Running the Project](#running-the-project)
5. [Troubleshooting](#troubleshooting)
6. [First-Time Demo](#first-time-demo)

---

## 📦 Prerequisites

Before you start, make sure you have these installed on your computer:

### Required Software

1. **Python 3.8 or higher**
   - Download: https://www.python.org/downloads/
   - **IMPORTANT:** During installation, check ✅ "Add Python to PATH"
   - Verify installation:
     ```bash
     python --version
     # Should show: Python 3.8.x or higher
     ```

2. **Git** (Optional, but recommended)
   - Download: https://git-scm.com/downloads
   - Used for version control
   - Verify installation:
     ```bash
     git --version
     ```

### System Requirements

- **Operating System:** Windows 10/11, macOS, or Linux
- **RAM:** Minimum 4GB (8GB recommended)
- **Disk Space:** At least 2GB free space
- **Internet:** Required for downloading dependencies

---

## ⚡ Quick Setup (Automated)

### For Windows Users (PowerShell)

**Option A: Using the Automated Setup Script**

1. **Extract the ZIP file**
   - Right-click → Extract All
   - Choose a location (e.g., `C:\Projects\nextgen_ids`)

2. **Open PowerShell**
   - Press `Win + X` → Choose "Windows PowerShell" or "Terminal"
   - Navigate to the project folder:
     ```powershell
     cd C:\Projects\nextgen_ids
     ```

3. **Run the setup script**
   ```powershell
   .\setup_demo.ps1
   ```

4. **What the script does:**
   - ✅ Creates Python virtual environment
   - ✅ Installs all dependencies
   - ✅ Verifies installation
   - ✅ Generates demo data
   - ✅ Tests the system
   - ✅ Starts the dashboard

5. **Access the dashboard**
   - Open browser: http://localhost:5000

**Done! Skip to [First-Time Demo](#first-time-demo) section.**

---

## 🔧 Manual Setup (Step by Step)

If the automated script doesn't work or you prefer manual setup:

### Step 1: Extract the Project

1. **Unzip the file**
   ```
   nextgen_ids.zip → nextgen_ids/
   ```

2. **Navigate to the folder**
   - **Windows PowerShell:**
     ```powershell
     cd C:\Path\To\nextgen_ids
     ```
   
   - **macOS/Linux Terminal:**
     ```bash
     cd /path/to/nextgen_ids
     ```

### Step 2: Create Virtual Environment

**Why virtual environment?**  
It keeps project dependencies isolated from your system Python.

**Windows:**
```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

**How to know it's activated?**  
You'll see `(.venv)` at the beginning of your command prompt:
```
(.venv) PS C:\Projects\nextgen_ids>
```

### Step 3: Install Dependencies

With the virtual environment activated:

```bash
# Upgrade pip (package manager)
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**This will install:**
- PyTorch (deep learning framework)
- Flask (web framework)
- Pandas, NumPy (data processing)
- Scikit-learn (machine learning utilities)
- Matplotlib (visualization)
- SHAP (explainability)
- And other dependencies...

**Installation time:** 2-5 minutes (depends on internet speed)

### Step 4: Verify Installation

```bash
# Check if key packages are installed
pip list | findstr "torch pandas Flask scikit-learn"
```

**Expected output:**
```
Flask                 3.0.0
pandas                2.2.0
scikit-learn          1.4.0
torch                 2.2.0
```

### Step 5: Generate Demo Data (Optional but recommended)

```bash
python generate_demo_data.py
```

**This creates:**
- `data/iot23/demo_attacks.csv` (4,400 network traffic samples)
- 6 attack types: Normal, DDoS, Port Scan, Malware, Brute Force, SQL Injection

---

## 🎮 Running the Project

### Method 1: Web Dashboard (Recommended for Beginners)

**Start the dashboard:**

**Windows:**
```powershell
# Make sure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Start dashboard
python src/dashboard.py
```

**macOS/Linux:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Start dashboard
python src/dashboard.py
```

**Output you should see:**
```
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.x.x:5000
```

**Access the dashboard:**
- Open browser
- Go to: http://localhost:5000
- You'll see the Next-Gen IDS interface

**Dashboard Features:**
- 📤 Upload CSV datasets
- 🎓 Train models
- 📊 Evaluate performance
- 🔍 Detect intrusions
- 🤖 Get AI explanations

### Method 2: Command Line Training

**Train a model from scratch:**

**Windows:**
```powershell
# With A-RNN architecture (recommended)
python src/train.py --dataset iot23 --epochs 10 --use-arnn

# Without A-RNN (baseline)
python src/train.py --dataset iot23 --epochs 10
```

**macOS/Linux:**
```bash
python src/train.py --dataset iot23 --epochs 10 --use-arnn
```

**Training parameters:**
- `--dataset`: Dataset name (iot23, nsl_kdd, unsw_nb15, cicids2017)
- `--epochs`: Training iterations (default: 10)
- `--batch_size`: Samples per batch (default: 64)
- `--seq_len`: Sequence length for LSTM (default: 100)
- `--use-arnn`: Enable A-RNN architecture (recommended)

**Expected output:**
```
Loading dataset: iot23
Train samples: 3080, Val: 440, Test: 880
Epoch 1/10: Loss=0.8234, F1=0.7234
Epoch 2/10: Loss=0.6123, F1=0.8456
...
Training complete! Best F1: 0.9342
Model saved to: checkpoints/best_iot23.pt
```

### Method 3: Evaluate Existing Model

**Test a trained model:**

```bash
python src/evaluate.py --dataset iot23 --checkpoint checkpoints/best_iot23.pt
```

**Output:**
```
Evaluation Results:
-------------------
Accuracy:  94.23%
Precision: 93.87%
Recall:    94.56%
F1-Score:  94.21%

Confusion matrix saved to: results/confusion_matrix.png
```

---

## 🎯 First-Time Demo

Follow these steps for your first demonstration:

### Step 1: Start the Dashboard

```powershell
# Activate environment (if not already)
.\.venv\Scripts\Activate.ps1

# Start dashboard
python src/dashboard.py
```

**Wait for:**
```
* Running on http://127.0.0.1:5000
```

### Step 2: Open Browser

- Go to: http://localhost:5000
- You should see the Next-Gen IDS dashboard

### Step 3: Upload Demo Dataset

1. Click **"Choose File"** or drag & drop
2. Navigate to: `data/iot23/demo_attacks.csv`
3. Click **"Upload"**

**You'll see:**
- Dataset stats: 4,400 rows, 39 features
- Preview of first few rows

### Step 4: Train the Model

1. **Configure training:**
   - Epochs: `5` (for quick demo)
   - Batch Size: `32`
   - Sequence Length: `64`
   - ✅ **Check "Use A-RNN"** (recommended)

2. Click **"Start Training"**

3. **Watch real-time progress:**
   - Training progress bar
   - Loss decreasing
   - Metrics updating

**Expected time:** 2-3 minutes

### Step 5: Evaluate Performance

1. Click **"Run Evaluation"**

2. **View results:**
   - Accuracy: ~93-95%
   - Precision, Recall, F1-Score
   - Confusion matrix visualization

### Step 6: Test Detection

1. Click **"Generate Threat Analysis"**

2. **Select attack type:**
   - Normal Traffic
   - DDoS Attack
   - Port Scan
   - Malware C2
   - Brute Force
   - SQL Injection

3. **See AI explanation:**
   - Attack description
   - Key indicators
   - Severity level
   - Recommended mitigations

### Step 7: Explore Features

Try different attack types and observe how the AI explains each one!

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Python is not recognized"

**Problem:** Python not in system PATH

**Solution:**
1. Reinstall Python
2. ✅ Check "Add Python to PATH" during installation
3. OR manually add Python to PATH

**Quick fix (Windows):**
```powershell
# Use full path to Python
C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
```

---

#### Issue 2: "Cannot activate virtual environment"

**Windows PowerShell Error:**
```
cannot be loaded because running scripts is disabled
```

**Solution:**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Try activation again
.\.venv\Scripts\Activate.ps1
```

**Alternative (if above doesn't work):**
```powershell
# Use cmd.exe activation instead
.\.venv\Scripts\activate.bat
```

---

#### Issue 3: "pip install fails"

**Problem:** Network issues or missing dependencies

**Solutions:**

**Option A: Use cached packages**
```bash
pip install --no-cache-dir -r requirements.txt
```

**Option B: Install PyTorch separately (it's the largest)**
```bash
# CPU-only version (smaller, faster download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install rest
pip install -r requirements.txt
```

**Option C: Install one by one**
```bash
pip install torch
pip install pandas
pip install Flask
pip install scikit-learn
pip install matplotlib
pip install shap
```

---

#### Issue 4: "Port 5000 already in use"

**Problem:** Another application using port 5000

**Solution A: Kill the process**

**Windows:**
```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill it (replace PID with actual number)
taskkill /PID <PID> /F
```

**macOS/Linux:**
```bash
# Find process
lsof -i :5000

# Kill it
kill -9 <PID>
```

**Solution B: Use different port**

Edit `src/dashboard.py`, change last line:
```python
# Change from:
app.run(debug=True, host='0.0.0.0', port=5000)

# To:
app.run(debug=True, host='0.0.0.0', port=8080)
```

Then access: http://localhost:8080

---

#### Issue 5: "ModuleNotFoundError"

**Problem:** Package not installed or virtual environment not activated

**Check if virtual environment is active:**
```bash
# You should see (.venv) in prompt
(.venv) PS C:\Projects\nextgen_ids>
```

**Solution:**
```bash
# Activate environment
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # macOS/Linux

# Reinstall packages
pip install -r requirements.txt
```

---

#### Issue 6: "CUDA errors" or "GPU not found"

**Problem:** PyTorch trying to use GPU but none available

**Solution:** Don't worry! The code automatically falls back to CPU.

**Verify:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Output:**
- `CUDA available: False` → Using CPU (perfectly fine)
- `CUDA available: True` → Using GPU (faster, but optional)

**CPU is perfectly fine for this project!** Training will just take a few extra minutes.

---

#### Issue 7: "Demo data not found"

**Problem:** `data/iot23/demo_attacks.csv` missing

**Solution:**
```bash
# Generate it
python generate_demo_data.py

# Verify it was created
# Windows:
Test-Path data\iot23\demo_attacks.csv

# macOS/Linux:
ls -la data/iot23/demo_attacks.csv
```

---

#### Issue 8: "Training is very slow"

**Normal training times:**
- 5 epochs: 2-3 minutes (on modern CPU)
- 10 epochs: 5-7 minutes

**If slower:**

**Speed up training:**
1. **Reduce epochs:** `--epochs 5` instead of 10
2. **Reduce batch size:** `--batch_size 16` instead of 64
3. **Use smaller dataset:** Only upload first 1000 rows of CSV
4. **Close other applications:** Free up RAM and CPU

---

## 📚 Project Structure Explanation

```
nextgen_ids/
│
├── src/                          # Source code
│   ├── model.py                  # Neural network architectures
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── dashboard.py              # Web interface (START HERE)
│   ├── data_loader.py            # Data processing
│   └── utils.py                  # Helper functions
│
├── data/                         # Datasets
│   └── iot23/
│       └── demo_attacks.csv      # Demo dataset (4,400 samples)
│
├── templates/                    # Web interface HTML
│   └── dashboard.html
│
├── checkpoints/                  # Saved models (created during training)
├── results/                      # Training results (created automatically)
├── uploads/                      # Uploaded files (created automatically)
│
├── .venv/                        # Virtual environment (you create this)
│
├── requirements.txt              # Dependencies list
├── README.md                     # Main documentation
├── generate_demo_data.py         # Demo data generator
└── setup_demo.ps1                # Automated setup script (Windows)
```

---

## 🎓 Learning Path

### For Complete Beginners:

1. **Day 1:** Setup → Run dashboard → Upload demo data
2. **Day 2:** Train model → Evaluate → Understand metrics
3. **Day 3:** Explore different attack types → Read documentation
4. **Day 4:** Try command-line training → Modify parameters
5. **Day 5:** Read code → Understand architecture

### For Experienced Users:

1. **Quick Start:** Run `setup_demo.ps1` → Dashboard ready in 5 minutes
2. **Explore:** Train with different datasets, modify hyperparameters
3. **Extend:** Add new attack types, integrate with your systems
4. **Deploy:** Use Docker, deploy to cloud, create APIs

---

## 📖 Additional Documentation

After setup, read these files for deeper understanding:

1. **README.md** - Complete project overview
2. **DASHBOARD_GUIDE.md** - Web interface tutorial
3. **TRAINING_VS_DETECTION.md** - Understand the concepts
4. **HOW_IT_WORKS_SIMPLE.md** - Technical explanation
5. **DEMONSTRATION_GUIDE.md** - Professional demo script
6. **ARCHITECTURE_DIAGRAMS.md** - Visual architecture
7. **EXTERNAL_DATASETS_GUIDE.md** - Using real-world datasets

---

## 🆘 Getting Help

### If You're Still Stuck:

1. **Check error message carefully** - Often tells you what's wrong
2. **Search error on Google** - Usually someone faced same issue
3. **Check GitHub Issues** - https://github.com/shadow-monarch007/NewGenIDS/issues
4. **Read documentation files** - Answer might be there
5. **Ask the person who sent you this** - They know the project!

### Useful Commands for Debugging:

```bash
# Check Python version
python --version

# Check if virtual environment is active
where python    # Windows
which python    # macOS/Linux

# List installed packages
pip list

# Check if key package works
python -c "import torch; print(torch.__version__)"
python -c "import pandas; print(pandas.__version__)"
python -c "import flask; print(flask.__version__)"
```

---

## ✅ Quick Reference Card

**Print this and keep it handy!**

### Essential Commands:

```powershell
# === SETUP (do once) ===
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS/Linux
pip install -r requirements.txt

# === DAILY USE ===
# 1. Activate environment
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS/Linux

# 2. Start dashboard
python src/dashboard.py
# → Open http://localhost:5000

# 3. Train model (command line)
python src/train.py --dataset iot23 --epochs 10 --use-arnn

# 4. Evaluate model
python src/evaluate.py --dataset iot23 --checkpoint checkpoints/best_iot23.pt

# === CLEANUP ===
# Stop dashboard: Ctrl+C
# Deactivate environment: deactivate
```

---

## 🎉 Success Checklist

- [ ] Python 3.8+ installed
- [ ] Project extracted
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Demo data generated
- [ ] Dashboard running
- [ ] First model trained
- [ ] Evaluation completed
- [ ] AI explanations working

**If all checked ✅, congratulations! You're ready to explore Next-Gen IDS!**

---

## 📞 Contact

**Project:** Next-Gen IDS  
**GitHub:** https://github.com/shadow-monarch007/NewGenIDS  
**Author:** Shadow Monarch  

---

**Last Updated:** October 14, 2025  
**Version:** 1.0

🚀 **Happy Intrusion Detecting!**
