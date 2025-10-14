# 🚀 QUICK START - 5 Minutes to Running

**For someone who just received the zip file**

---

## ⚡ Super Fast Setup (Windows)

### Step 1: Extract ZIP
```
nextgen_ids.zip → Extract to: C:\nextgen_ids
```

### Step 2: Open PowerShell
```
Win + X → Windows PowerShell
```

### Step 3: Navigate to folder
```powershell
cd C:\nextgen_ids
```

### Step 4: Run Automated Setup
```powershell
.\setup_demo.ps1
```

**That's it!** The script will:
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Generate demo data
- ✅ Start dashboard

**Dashboard opens at:** http://localhost:5000

---

## 🔧 Manual Setup (if script fails)

### 1. Create Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Generate Demo Data
```powershell
python generate_demo_data.py
```

### 4. Start Dashboard
```powershell
python src/dashboard.py
```

**Open browser:** http://localhost:5000

---

## 🎮 First Demo (3 Steps)

### Step 1: Upload Data
- Click "Choose File"
- Select: `data/iot23/demo_attacks.csv`
- Click "Upload"

### Step 2: Train Model
- Epochs: `5`
- Batch Size: `32`
- ✅ Check "Use A-RNN"
- Click "Start Training"
- Wait 2-3 minutes

### Step 3: Evaluate
- Click "Run Evaluation"
- See results: ~94% accuracy
- Click "Generate Threat Analysis"
- Explore different attack types

**Done! You're running Next-Gen IDS!** 🎉

---

## 🆘 Quick Troubleshooting

**"Python not found"**
→ Install Python 3.8+ from python.org (check "Add to PATH")

**"Cannot activate .venv"**
→ Run PowerShell as Admin:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**"Port 5000 in use"**
→ Edit `src/dashboard.py`, change port to `8080`

**"Packages fail to install"**
→ Install PyTorch first:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 📚 Need More Help?

Read: **SETUP_GUIDE_FOR_OTHERS.md** (detailed 50-page guide)

---

**Time to running:** ~5 minutes  
**Time to first demo:** ~10 minutes total

🚀 **Enjoy!**
