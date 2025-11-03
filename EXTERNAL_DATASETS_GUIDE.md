# 📊 External IDS Datasets Guide

## Quick Reference Table

| Dataset | Size | Attack Types | Best For | Difficulty |
|---------|------|--------------|----------|------------|
| **NSL-KDD** | 20 MB | 4 categories | Quick testing | ⭐ Easy |
| **UNSW-NB15** | 2 GB | 9 categories | Realistic traffic | ⭐⭐ Medium |
| **CIC-IDS-2017** | 3 GB | 7 categories | Comprehensive | ⭐⭐ Medium |
| **CIC-IDS-2018** | 4 GB | 7 categories | Modern attacks | ⭐⭐⭐ Hard |
| **IoT-23** | 325 GB | IoT malware | IoT-specific | ⭐⭐⭐ Hard |
| **Bot-IoT** | 69 GB | IoT botnet | IoT security | ⭐⭐⭐ Hard |

---

## 🎯 Recommended Downloads (Step by Step)

### Option 1: Quick Testing (5 minutes)
**Dataset:** NSL-KDD (20 MB)

```powershell
# Run the download script
.\download_datasets.ps1

# Choose option 1
# Files will be in: data\nsl_kdd\
```

**What you get:**
- 125,973 training samples
- 22,544 test samples
- 4 attack categories: DoS, Probe, R2L, U2R
- 41 features

---

### Option 2: Best for Your Project (1 hour)
**Dataset:** UNSW-NB15 (2 GB) ⭐ **RECOMMENDED**

```powershell
# Run the download script
.\download_datasets.ps1

# Choose option 2
# Wait ~30-60 minutes for download
# Files will be in: data\unsw_nb15\
```

**What you get:**
- 2,540,044 samples
- 9 attack categories:
  - Fuzzers (fuzzing attacks)
  - Analysis (port scans, spyware)
  - Backdoors (unauthorized access)
  - DoS (denial of service)
  - Exploits (vulnerability exploitation)
  - Generic (attacks against block ciphers)
  - Reconnaissance (information gathering)
  - Shellcode (code injection)
  - Worms (self-replicating malware)
- 49 features
- Highly realistic modern traffic

**Why this is best:**
- Perfect size (not too small, not too large)
- Modern attack patterns (2015 data)
- Well-documented features
- Widely used in research
- Matches your project's scope

---

### Option 3: Most Popular (2 hours + manual)
**Dataset:** CIC-IDS-2017 (3 GB)

**Download Steps:**
1. Visit Kaggle: https://www.kaggle.com/datasets/cicdataset/cicids2017
2. Create free account (if needed)
3. Click "Download" button
4. Extract ZIP to: `data\cicids2017\`

**Alternative (direct from source):**
1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Fill out request form
3. Receive download link via email
4. Download 8 CSV files
5. Place in: `data\cicids2017\`

**What you get:**
- Full week of network traffic (Monday-Friday)
- Attack scenarios:
  - **Monday:** Normal traffic (baseline)
  - **Tuesday:** Brute Force FTP, Brute Force SSH
  - **Wednesday:** DoS/DDoS attacks (GoldenEye, Slowloris, Hulk, Heartbleed)
  - **Thursday:** Web attacks (SQL Injection, XSS, Brute Force Web)
  - **Friday:** Botnet (ARES), Port Scan, DDoS (LOIT)
- 80+ features
- 2.8 million samples

**Why this is popular:**
- Most cited in research papers
- Comprehensive attack coverage
- Well-labeled data
- Matches real-world scenarios
- Your 6 attack types are all here!

---

## 📋 Dataset Details

### 1. NSL-KDD (Small, Classic)

**Download:**
```bash
# Automated (using script)
.\download_datasets.ps1  # Choose option 1

# Manual
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt
wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt
```

**Features:** 41 features including:
- Duration, protocol_type, service, flag
- src_bytes, dst_bytes, land, wrong_fragment
- urgent, hot, num_failed_logins
- logged_in, num_compromised, root_shell
- etc.

**Attack Types:**
1. **DoS** (Denial of Service): Teardrop, Smurf, Pod, Land, Neptune, Back
2. **Probe** (Surveillance): Satan, Ipsweep, Nmap, Portsweep, Mscan, Saint
3. **R2L** (Remote to Local): Guess_passwd, Ftp_write, Imap, Phf, Multihop, Warezmaster
4. **U2R** (User to Root): Buffer_overflow, Loadmodule, Rootkit, Perl

**Pros:**
- Small and fast
- Good for initial testing
- Classic benchmark

**Cons:**
- Old data (1999 KDD Cup)
- Not realistic modern traffic
- Limited attack diversity

---

### 2. UNSW-NB15 (Medium, Modern) ⭐

**Download:**
```bash
# Automated (using script)
.\download_datasets.ps1  # Choose option 2

# Manual
wget https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download -O UNSW-NB15_1.csv
wget https://cloudstor.aarnet.edu.au/plus/s/M63LvYQFjvf9N6V/download -O UNSW-NB15_2.csv
wget https://cloudstor.aarnet.edu.au/plus/s/NsYt6fTfqiKN2Ub/download -O UNSW-NB15_3.csv
wget https://cloudstor.aarnet.edu.au/plus/s/ZHQ0jl9F0LYHXfG/download -O UNSW-NB15_4.csv
wget https://cloudstor.aarnet.edu.au/plus/s/hdAG9wlr6fRzh1O/download -O UNSW_NB15_testing-set.csv
```

**Features:** 49 features including:
- srcip, sport, dstip, dsport, proto
- state, dur, sbytes, dbytes, sttl, dttl
- sloss, dloss, service, Sload, Dload
- Spkts, Dpkts, swin, dwin, stcpb, dtcpb
- smeansz, dmeansz, trans_depth, res_bdy_len
- Sjit, Djit, Stime, Ltime
- Sintpkt, Dintpkt, tcprtt, synack, ackdat
- is_sm_ips_ports, ct_state_ttl, ct_flw_http_mthd
- is_ftp_login, ct_ftp_cmd, ct_srv_src, ct_srv_dst
- ct_dst_ltm, ct_src_ltm, ct_src_dport_ltm, ct_dst_sport_ltm
- ct_dst_src_ltm

**Attack Types:** 9 categories
1. **Fuzzers:** Fuzzing attacks to cause program crashes
2. **Analysis:** Port scans, spam, HTML file penetrations
3. **Backdoors:** Unauthorized access bypassing security
4. **DoS:** Denial of service attacks
5. **Exploits:** Exploiting vulnerabilities (buffer overflow, heap overflow, etc.)
6. **Generic:** Attacks against block ciphers
7. **Reconnaissance:** Information gathering (port scan, IP scan)
8. **Shellcode:** Code injection exploits
9. **Worms:** Self-replicating malware

**Pros:**
- Modern traffic (2015)
- Realistic network behavior
- Diverse attack types
- Good size for training
- Well-documented

**Cons:**
- Requires preprocessing
- Some features complex
- Medium download time

---

### 3. CIC-IDS-2017 (Large, Popular) ⭐

**Download:**
```bash
# Kaggle (easier)
https://www.kaggle.com/datasets/cicdataset/cicids2017

# Direct (requires form)
https://www.unb.ca/cic/datasets/ids-2017.html
```

**Features:** 80+ features including:
- Flow Duration, Total Fwd/Bwd Packets
- Total/Min/Max/Mean/Std Length of Fwd/Bwd Packets
- Flow Bytes/s, Flow Packets/s
- Flow IAT Mean/Std/Max/Min
- Fwd/Bwd IAT Total/Mean/Std/Max/Min
- Fwd/Bwd PSH/URG Flags
- Fwd/Bwd Header Length
- Fwd/Bwd Packets/s
- Min/Max/Mean/Std Packet Length
- FIN/SYN/RST/PSH/ACK/URG/CWE/ECE Flag Count
- Down/Up Ratio, Average Packet Size
- Fwd/Bwd Segment Size Avg
- Fwd/Bwd Bytes/Bulk Avg
- Subflow Fwd/Bwd Packets/Bytes
- Init_Win_bytes_forward/backward
- act_data_pkt_fwd, min_seg_size_forward
- Active Mean/Std/Max/Min
- Idle Mean/Std/Max/Min

**Attack Scenarios:**
1. **Monday:** Benign (normal traffic)
2. **Tuesday:**
   - Brute Force FTP (9:20 AM - 10:20 AM)
   - Brute Force SSH (2:00 PM - 3:00 PM)
3. **Wednesday:**
   - DoS GoldenEye (9:47 AM - 10:10 AM)
   - DoS Slowloris (10:14 AM - 11:00 AM)
   - DoS Slowhttptest (11:00 AM - 11:23 AM)
   - DoS Hulk (11:10 AM - 11:23 AM)
   - Heartbleed (3:25 PM - 4:05 PM)
4. **Thursday:**
   - Web Attack - Brute Force (9:20 AM - 10:00 AM)
   - Web Attack - XSS (10:15 AM - 10:35 AM)
   - Web Attack - SQL Injection (10:40 AM - 10:42 AM)
   - Infiltration (2:30 PM - 4:35 PM)
5. **Friday:**
   - Botnet ARES (10:02 AM - 11:02 AM)
   - Port Scan (1:55 PM - 3:55 PM)
   - DDoS LOIT (3:45 PM - 4:08 PM)

**Pros:**
- Most cited in research
- Comprehensive attack coverage
- Matches your 6 attack types!
- Well-labeled and documented
- Realistic timing and behavior

**Cons:**
- Large download
- Requires preprocessing
- Many redundant features
- Class imbalance issues

---

## 🔧 How to Use Downloaded Datasets

### Step 1: Download
```powershell
# Quick way
.\download_datasets.ps1

# Or download manually from links above
```

### Step 2: Verify Files
```powershell
# Check what you have
ls data\*\*.csv

# Should see:
# data\nsl_kdd\KDDTrain+.txt
# data\unsw_nb15\UNSW-NB15_1.csv
# data\cicids2017\Monday-WorkingHours.csv
# etc.
```

### Step 3: Preprocess (if needed)
```bash
# Your code should already handle this!
# But if you need custom preprocessing:
python src/preprocess.py --dataset unsw_nb15
```

### Step 4: Train Your Model
```bash
# Train with new dataset
python src/train.py --dataset unsw_nb15 --epochs 10 --use-arnn

# Or via dashboard
python src/dashboard.py
# Then upload the CSV files
```

### Step 5: Evaluate
```bash
# Test accuracy
python src/evaluate.py --dataset unsw_nb15

# Or via dashboard's evaluation feature
```

---

## 📊 Dataset Comparison for Your Project

### Your Current Demo Data
- **Size:** 4,400 samples
- **Source:** Synthetic (generated)
- **Attack Types:** 6 (DDoS, Port Scan, Malware C2, Brute Force, SQL Injection, Normal)
- **Use Case:** Demo, testing, quick validation

### Recommended Real Data: UNSW-NB15
- **Size:** 2.5 million samples
- **Source:** Real network traffic
- **Attack Types:** 9 categories (covers all your types!)
- **Use Case:** Training, production, research paper

### Mapping Your Attack Types:

| Your Type | UNSW-NB15 Equivalent |
|-----------|---------------------|
| Normal | Normal |
| DDoS | DoS |
| Port Scan | Reconnaissance + Analysis |
| Malware C2 | Backdoors + Worms |
| Brute Force | Exploits |
| SQL Injection | Exploits |

---

## 🚀 Quick Start Recommendations

### For Demo Tomorrow:
**Keep your synthetic data!**
- Already works
- Perfectly labeled
- Fast to load
- Demonstrates all features

### For Research Paper:
**Use UNSW-NB15** ⭐
- Download today: `.\download_datasets.ps1` → Option 2
- Train overnight
- Compare with your demo results
- Include both in paper: "synthetic vs real data comparison"

### For Production System:
**Use CIC-IDS-2017**
- Most comprehensive
- Real-world scenarios
- Industry standard
- Good for baseline comparison

---

## 📝 Citation Information

### NSL-KDD
```
Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). 
A detailed analysis of the KDD CUP 99 data set. 
In IEEE Symposium on Computational Intelligence for Security and Defense Applications.
```

### UNSW-NB15
```
Moustafa, N., & Slay, J. (2015). 
UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). 
In Military Communications and Information Systems Conference (MilCIS).
```

### CIC-IDS-2017
```
Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). 
Toward generating a new intrusion detection dataset and intrusion traffic characterization. 
In ICISSP (pp. 108-116).
```

---

## ❓ FAQs

**Q: Which dataset should I use for my demo?**
A: Keep your synthetic demo_attacks.csv! It's perfect for demos. Use real datasets for validation.

**Q: Which is best for research papers?**
A: UNSW-NB15 or CIC-IDS-2017. Both are highly cited and respected.

**Q: How long does download take?**
A: NSL-KDD: 1 minute, UNSW-NB15: 30-60 minutes, CIC-IDS-2017: 1-2 hours

**Q: Do I need to preprocess?**
A: Your code might already handle it. Test by training on raw data first.

**Q: Can I mix datasets?**
A: Yes! Train on one, test on another to show generalization.

**Q: What if download fails?**
A: Try the script again, or download manually from the links provided.

---

## 🎯 My Final Recommendation

**For your external guide demo:**
1. **Keep using demo_attacks.csv** (shows all 6 attack types clearly)
2. **Download UNSW-NB15 as backup** (shows real-world validation)
3. **Mention in presentation:** "Validated on both synthetic (4,400 samples) and real-world (2.5M samples) data"

**Script to run:**
```powershell
# Download UNSW-NB15
.\download_datasets.ps1  # Choose option 2

# Wait for download (~30-60 min)

# Train on real data
python src/train.py --dataset unsw_nb15 --epochs 5 --use-arnn

# Compare results with demo data
# Show guide: "Works on both synthetic and real data!"
```

This gives you the best of both worlds! 🎉
