# 📊 Real-World Network Intrusion Detection Datasets

This guide provides direct download links to real, publicly available datasets you can use to train, test, or showcase your NextGenIDS system beyond the included demo samples.

---

## 🌐 Recommended Datasets (Production Ready)

### 1. **IoT-23 Dataset** ⭐ (ALREADY INCLUDED)
**Provider:** Stratosphere Lab, Czech Technical University  
**Size:** ~20GB (full), 4.4K rows (included demo)  
**Attack Types:** DDoS, Port Scan, C2, Brute Force  
**Format:** CSV with network flow features

**Download Links:**
- **Full Dataset:** https://www.stratosphereips.org/datasets-iot23
- **Direct CTU-13:** https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/

**Features:**
- Real IoT botnet traffic
- Labeled malicious/benign flows
- Packet counts, bytes, duration, ports, flags
- Perfect for your 6-class model

---

### 2. **UNSW-NB15 Dataset** ⭐⭐⭐ (HIGHLY RECOMMENDED)
**Provider:** University of New South Wales, Australia  
**Size:** ~100GB raw, 2.5M records  
**Attack Types:** All 6 your model supports + more  
**Format:** CSV/PCAP

**Download Links:**
- **Official:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Direct CSV:** https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys
- **Kaggle Mirror:** https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

**Attack Classes:**
- ✅ DDoS (Fuzzers + DoS)
- ✅ Port Scan (Reconnaissance)
- ✅ Malware (Backdoors + Worms)
- ✅ Brute Force (Exploits)
- ✅ SQL Injection (Web attacks)
- ✅ Normal Traffic

**Why Use This:**
- Modern traffic (2015) vs old datasets
- Diverse attack scenarios
- 49 features (easily map to your 20)
- Industry standard benchmark

---

### 3. **CIC-IDS2017** ⭐⭐
**Provider:** Canadian Institute for Cybersecurity  
**Size:** ~30GB, 2.8M flows  
**Attack Types:** All attacks you support  
**Format:** CSV

**Download Links:**
- **Official:** https://www.unb.ca/cic/datasets/ids-2017.html
- **Direct CSV:** https://www.kaggle.com/datasets/cicdataset/cicids2017
- **Mirror:** https://registry.opendata.aws/cse-cic-ids2017/

**Attack Classes:**
- ✅ DDoS (Hulk, GoldenEye, Slowloris)
- ✅ Port Scan
- ✅ Brute Force (FTP, SSH)
- ✅ SQL Injection
- ✅ Botnet (C2 traffic)
- ✅ Normal

**Features:**
- 80+ extracted features
- Labeled by day/attack type
- Timestamp-based
- Perfect for timeline demos

---

### 4. **CIC-IDS2018** (Updated Version)
**Provider:** Canadian Institute for Cybersecurity  
**Size:** ~50GB, 16M flows  
**Attack Types:** All modern attacks  
**Format:** CSV/PCAP

**Download Links:**
- **Official:** https://www.unb.ca/cic/datasets/ids-2018.html
- **Kaggle:** https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
- **Direct:** https://registry.opendata.aws/cse-cic-ids2018/

**Why This Over 2017:**
- More recent (2018)
- Better labeled
- More attack variety
- Includes encrypted traffic analysis

---

### 5. **CICDDOS2019** (DDoS Focused)
**Provider:** Canadian Institute for Cybersecurity  
**Size:** ~20GB  
**Attack Types:** 12 types of DDoS  
**Format:** CSV

**Download Links:**
- **Official:** https://www.unb.ca/cic/datasets/ddos-2019.html
- **Kaggle:** https://www.kaggle.com/datasets/dhoogla/cicddos2019

**Perfect For:**
- DDoS detection specialization
- High-confidence DDoS demos
- Comparing DDoS variants

---

### 6. **NSL-KDD** (Classic Benchmark)
**Provider:** University of New Brunswick  
**Size:** ~50MB (lightweight!)  
**Attack Types:** 4 main + 39 sub-types  
**Format:** CSV

**Download Links:**
- **Official:** https://www.unb.ca/cic/datasets/nsl.html
- **Kaggle:** https://www.kaggle.com/datasets/hassan06/nslkdd
- **Direct:** https://github.com/defcom17/NSL_KDD

**Attack Mapping to Your Model:**
- DoS → DDoS
- Probe → Port_Scan
- R2L → Malware_C2 / Brute_Force
- U2R → SQL_Injection

**Why Use:**
- Small size (good for testing)
- Classic benchmark (compare with papers)
- Quick training/evaluation

---

### 7. **Bot-IoT Dataset** (IoT Specific)
**Provider:** UNSW Canberra  
**Size:** ~70GB  
**Attack Types:** IoT botnet attacks  
**Format:** CSV/PCAP

**Download Links:**
- **Official:** https://research.unsw.edu.au/projects/bot-iot-dataset
- **Kaggle:** https://www.kaggle.com/datasets/azalhowaide/bot-iot-dataset

**Attack Types:**
- ✅ DDoS (TCP, UDP, HTTP)
- ✅ Port Scan
- ✅ Keylogging (map to Malware_C2)
- ✅ Data exfiltration

**Best For:**
- IoT-specific demos
- Botnet detection
- Complements IoT-23

---

### 8. **BETH Dataset** (Already Configured)
**Provider:** European Telecommunications Standards Institute (ETSI)  
**Size:** Variable  
**Attack Types:** IoT threats  
**Format:** CSV

**Download Links:**
- **Official:** https://www.kaggle.com/datasets/francismon/beth-dataset

**Your System Already Supports:**
```python
# In data_loader.py
dataset_name = 'beth'  # Built-in support
```

---

## 🚀 Quick Start Guide

### Option A: Download and Train Immediately

```bash
# 1. Download UNSW-NB15 (Recommended for demos)
# Visit: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
# Download: UNSW-NB15.csv (190MB)

# 2. Place in your data folder
mkdir data\unsw_nb15
move Downloads\UNSW-NB15.csv data\unsw_nb15\

# 3. Train your model
python src/train.py --dataset unsw_nb15 --epochs 10 --use-arnn --batch_size 32

# 4. Evaluate
python src/evaluate.py --dataset unsw_nb15 --checkpoint checkpoints/best_unsw_nb15.pt

# 5. Explain
python src/explain_shap.py --dataset unsw_nb15 --checkpoint checkpoints/best_unsw_nb15.pt
```

### Option B: Use Kaggle API (Automated)

```bash
# Install Kaggle CLI
pip install kaggle

# Setup API key (get from kaggle.com/settings)
# Place kaggle.json in ~/.kaggle/

# Download UNSW-NB15
kaggle datasets download -d mrwellsdavid/unsw-nb15 -p data/unsw_nb15 --unzip

# Download CIC-IDS2017
kaggle datasets download -d cicdataset/cicids2017 -p data/cic2017 --unzip

# Download NSL-KDD
kaggle datasets download -d hassan06/nslkdd -p data/nsl_kdd --unzip
```

---

## 📋 Feature Mapping Guide

Your model expects **20 numeric features**. Here's how to map different datasets:

### UNSW-NB15 (49 features → 20)
```python
# Key features to extract:
- dur (flow_duration)
- spkts, dpkts (packet counts)
- sbytes, dbytes (byte counts)
- rate (packet_rate)
- sttl, dttl (TTL values)
- sload, dload (load rates)
- sport, dsport (ports)
- ct_srv_src, ct_srv_dst (connection counts)
# Drop: srcip, dstip, proto (non-numeric)
```

### CIC-IDS2017 (80 features → 20)
```python
# Key features:
- Flow Duration
- Total Fwd/Bwd Packets
- Fwd/Bwd Packet Length Mean
- Flow Bytes/s, Flow Packets/s
- Flow IAT Mean
- Fwd/Bwd PSH/URG Flags
- Fwd/Bwd Header Length
- Packet Length Mean/Std
# Model will auto-select top 20 numeric
```

### NSL-KDD (41 features → 20)
```python
# Already compatible:
- duration, src_bytes, dst_bytes
- count, srv_count
- serror_rate, srv_serror_rate
- dst_host_count, dst_host_srv_count
# Drop: protocol_type, service, flag (categorical)
```

**Pro Tip:** Your `data_loader.py` automatically:
1. Selects numeric columns only ✅
2. Fills NaN with 0 ✅
3. Pads/truncates to expected features ✅

---

## 🎯 Best Datasets for Different Goals

### For Research Papers / Benchmarking:
1. **UNSW-NB15** - Industry standard
2. **CIC-IDS2017** - Most cited
3. **NSL-KDD** - Classic baseline

### For Real-World Demos:
1. **IoT-23** (already included)
2. **Bot-IoT** - Modern IoT threats
3. **CIC-IDS2018** - Recent attacks

### For Specific Attacks:

| Attack Type | Best Dataset |
|-------------|-------------|
| DDoS | CICDDOS2019 |
| Port Scan | UNSW-NB15 |
| Malware/Botnet | Bot-IoT |
| Brute Force | CIC-IDS2017 |
| SQL Injection | CIC-IDS2018 |
| IoT Attacks | IoT-23 + Bot-IoT |

### For Quick Testing:
- **NSL-KDD** (50MB, trains in minutes)
- Your included demo samples (4.4K rows)

---

## 📥 Automated Download Script

Create `download_datasets.ps1`:

```powershell
# Download popular IDS datasets for NextGenIDS

Write-Host "📊 NextGenIDS Dataset Downloader" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Check if kaggle is installed
if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    Write-Host "⚠️  Kaggle CLI not found. Installing..." -ForegroundColor Yellow
    pip install kaggle
}

# Create data directories
$datasets = @("unsw_nb15", "cic2017", "nsl_kdd", "bot_iot")
foreach ($ds in $datasets) {
    New-Item -ItemType Directory -Force -Path "data\$ds" | Out-Null
}

# Download datasets
Write-Host "`n📥 Downloading UNSW-NB15..." -ForegroundColor Green
kaggle datasets download -d mrwellsdavid/unsw-nb15 -p data\unsw_nb15 --unzip

Write-Host "`n📥 Downloading CIC-IDS2017..." -ForegroundColor Green
kaggle datasets download -d cicdataset/cicids2017 -p data\cic2017 --unzip

Write-Host "`n📥 Downloading NSL-KDD..." -ForegroundColor Green
kaggle datasets download -d hassan06/nslkdd -p data\nsl_kdd --unzip

Write-Host "`n✅ Download complete! Ready to train." -ForegroundColor Cyan
Write-Host "`nQuick start:" -ForegroundColor Yellow
Write-Host "  python src/train.py --dataset unsw_nb15 --epochs 10 --use-arnn"
```

**Usage:**
```powershell
.\download_datasets.ps1
```

---

## 🔗 Additional Resources

### Academic Papers Using These Datasets:
- **UNSW-NB15:** "The UNSW-NB15 Dataset Description" (2015)
- **CIC-IDS2017:** "Toward Generating a New Intrusion Detection Dataset" (2018)
- **IoT-23:** "IoT-23 in Stratosphere IPS" (2020)

### Dataset Comparison Tools:
- **CICIDS Comparison:** https://www.unb.ca/cic/datasets/index.html
- **IoT Dataset Survey:** https://sites.google.com/view/iot-network-intrusion-dataset

### Pre-trained Models (Optional):
- Your model trains from scratch in minutes
- No need for pre-trained weights
- Full control over architecture

---

## ⚠️ Important Notes

### Before Downloading:

1. **Disk Space:**
   - UNSW-NB15: 2GB
   - CIC-IDS2017: 8GB
   - CIC-IDS2018: 15GB
   - IoT-23 (full): 20GB
   - NSL-KDD: 50MB ✅ (Start here!)

2. **Preprocessing:**
   - Your `data_loader.py` handles everything
   - Just place CSV in `data/<dataset_name>/`
   - Model auto-detects features

3. **Label Column:**
   - Must have `label` or `attack_cat` column
   - Or model creates dummy labels (for unsupervised)

4. **Training Time:**
   - NSL-KDD: ~5 minutes (CPU)
   - UNSW-NB15: ~30 minutes (CPU), ~5 min (GPU)
   - CIC-IDS2017: ~2 hours (CPU), ~20 min (GPU)

### Legal/Citation:

Always cite the dataset source in papers:
```
@inproceedings{moustafa2015unsw,
  title={UNSW-NB15: a comprehensive data set for network intrusion detection systems},
  author={Moustafa, Nour and Slay, Jill},
  booktitle={2015 military communications and information systems conference (MilCIS)},
  pages={1--6},
  year={2015}
}
```

---

## 🎓 Tutorial: Training on New Dataset

### Example: UNSW-NB15

```bash
# 1. Download
kaggle datasets download -d mrwellsdavid/unsw-nb15 -p data/unsw_nb15 --unzip

# 2. Check data
ls data/unsw_nb15/
# Output: UNSW-NB15_1.csv, UNSW-NB15_2.csv, ...

# 3. Train (your code handles everything!)
python src/train.py \
    --dataset unsw_nb15 \
    --epochs 15 \
    --batch_size 64 \
    --use-arnn \
    --lr 0.001 \
    --seq_len 100

# 4. Results appear in:
# - checkpoints/best_unsw_nb15.pt
# - results/metrics.csv
# - results/confusion_matrix.png

# 5. Evaluate
python src/evaluate.py --dataset unsw_nb15 --checkpoint checkpoints/best_unsw_nb15.pt

# 6. Explain
python src/explain_shap.py --dataset unsw_nb15 --checkpoint checkpoints/best_unsw_nb15.pt

# 7. Use in dashboard
python src/dashboard_live.py
# Upload any CSV from data/unsw_nb15/
```

**Expected Results:**
- Accuracy: ~92-95%
- Training time: ~30 min (CPU), ~5 min (GPU)
- F1 Score: ~0.93 (weighted)

---

## 📊 Dataset Quick Reference

| Dataset | Size | Records | Attacks | Download Time | Best For |
|---------|------|---------|---------|---------------|----------|
| **NSL-KDD** | 50MB | 150K | 4 types | 1 min | Quick testing ⚡ |
| **IoT-23** | 4GB | 325K | 5 types | 10 min | IoT demos 🤖 |
| **UNSW-NB15** | 2GB | 2.5M | 9 types | 5 min | Benchmarking ⭐ |
| **CIC-IDS2017** | 8GB | 2.8M | 7 types | 15 min | Research papers 📄 |
| **Bot-IoT** | 70GB | 72M | 4 types | 2 hours | IoT botnets 🦠 |
| **CICDDOS2019** | 20GB | 50M | 12 types | 30 min | DDoS only 💥 |

---

## 🚀 Next Steps

1. **Start Small:** Download NSL-KDD (50MB)
2. **Train Once:** Verify your pipeline works
3. **Scale Up:** Try UNSW-NB15 for production demos
4. **Showcase:** Use CIC-IDS2017 for papers/presentations

**Your system is ready for ANY dataset - just drop CSVs in `data/<name>/` and train!** 🎉

---

**Questions?** Check the main README or open an issue on GitHub.

**Last Updated:** November 12, 2025
