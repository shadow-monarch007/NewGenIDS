# 🚀 IDS File Upload Quick Reference

## Supported File Types

| Format | Extension | Description | Auto-Convert |
|--------|-----------|-------------|--------------|
| **CSV** | `.csv` | Traffic features (20 columns) | ❌ Direct |
| **PCAP** | `.pcap` | Raw packet capture | ✅ Yes |
| **PCAPNG** | `.pcapng` | Next-gen packet capture | ✅ Yes |

## Quick Commands

### Upload via Dashboard (Easiest!)
```
1. Visit: http://localhost:5000
2. Choose file (CSV or PCAP)
3. Click "Analyze Traffic"
4. ✅ Done!
```

### Convert PCAP to CSV
```bash
# Basic conversion
python src/pcap_converter.py capture.pcap

# Fast (limit packets)
python src/pcap_converter.py capture.pcap --max-packets 5000

# Custom window
python src/pcap_converter.py capture.pcap --window 10

# Specify output
python src/pcap_converter.py capture.pcap --output features.csv
```

### Get Sample Files
```bash
# Download test PCAP files
python download_sample_pcap.py

# Download test datasets (NSL-KDD, etc.)
python test_real_dataset.py
```

### Run Predictions
```bash
# Predict from CSV
python src/predict.py --input data.csv

# Predict from PCAP (convert first)
python src/pcap_converter.py capture.pcap
python src/predict.py --input capture_features.csv
```

## Required CSV Format

**20 columns** in this exact order:
1. `packet_rate` - Packets per second
2. `packet_size` - Average packet size (bytes)
3. `byte_rate` - Bytes per second  
4. `flow_duration` - Flow duration (seconds)
5. `total_packets` - Total packet count
6. `total_bytes` - Total bytes transferred
7. `entropy` - Data entropy (randomness)
8. `port_scan_score` - Port scan indicator
9. `syn_flag_count` - SYN flags count
10. `ack_flag_count` - ACK flags count
11. `fin_flag_count` - FIN flags count
12. `rst_flag_count` - RST flags count
13. `psh_flag_count` - PSH flags count
14. `urg_flag_count` - URG flags count
15. `unique_src_ports` - Unique source ports
16. `unique_dst_ports` - Unique destination ports
17. `payload_entropy` - Payload entropy
18. `dns_query_count` - DNS query count
19. `http_request_count` - HTTP request count
20. `ssl_handshake_count` - SSL handshake count

**⚠️ Important:** No label/attack_type column! System predicts unlabeled data.

## What Files Work

### ✅ Works Great
- Wireshark captures (.pcap, .pcapng)
- tcpdump output (.pcap)
- Pre-extracted CSV (20 columns)
- Downloaded datasets (NSL-KDD, CICIDS, etc. - after conversion)

### ❌ Not Supported
- Excel files (.xlsx, .xls)
- JSON files
- Different CSV formats (<20 or >20 columns)
- Encrypted traffic (can't extract features)

## Capture Your Own Traffic

### Windows (Wireshark)
```
1. Open Wireshark
2. Select network interface
3. Start capture
4. Stop after desired duration
5. File → Save As → capture.pcap
6. Upload to dashboard!
```

### Linux/Mac (tcpdump)
```bash
# Capture 1000 packets
sudo tcpdump -i eth0 -w capture.pcap -c 1000

# Capture for 60 seconds
sudo timeout 60 tcpdump -i eth0 -w capture.pcap

# Upload capture.pcap to dashboard
```

## File Size Guidelines

| PCAP Size | Conversion Time | Recommendation |
|-----------|-----------------|----------------|
| < 1 MB | < 5 seconds | ✅ Perfect |
| 1-10 MB | 5-30 seconds | ✅ Good |
| 10-50 MB | 30-120 seconds | ⚠️ Use `--max-packets 10000` |
| > 50 MB | > 2 minutes | ⚠️ Use `--max-packets 20000` |

### Speed Up Large Files
```bash
# Limit to first 10,000 packets
python src/pcap_converter.py big.pcap --max-packets 10000

# Larger time windows (fewer flows)
python src/pcap_converter.py big.pcap --window 15

# Both
python src/pcap_converter.py huge.pcap --max-packets 20000 --window 10
```

## Detection Capabilities

| Attack Type | CSV Support | PCAP Support |
|-------------|-------------|--------------|
| DDoS | ✅ | ✅ |
| Port Scan | ✅ | ✅ |
| Malware C2 | ✅ | ✅ |
| Brute Force | ✅ | ✅ |
| SQL Injection | ✅ | ✅ |
| Normal Traffic | ✅ | ✅ |

## Troubleshooting

### "Invalid file format"
→ Upload CSV (20 cols) or PCAP (.pcap/.pcapng)

### "No features extracted"
→ PCAP file empty or corrupted. Check with `tcpdump -r file.pcap -c 10`

### "Conversion too slow"
→ Use `--max-packets 5000` to limit processing

### "Model not found"
→ Train model first: `python src/train.py --dataset iot23 --epochs 5`

### "Wrong number of columns"
→ Use PCAP converter or ensure CSV has exactly 20 columns

## Example Workflow

### Real-World Network Analysis
```bash
# 1. Capture network traffic (1 hour)
sudo tcpdump -i eth0 -w network_1hr.pcap -G 3600 -W 1

# 2. Convert to features (limit for speed)
python src/pcap_converter.py network_1hr.pcap --max-packets 50000

# 3. Upload to dashboard
# Go to http://localhost:5000
# Upload network_1hr_features.csv

# 4. View threat predictions!
```

### Security Investigation
```bash
# 1. Get suspicious traffic capture from SIEM
# suspicious_activity.pcap

# 2. Convert and analyze
python src/pcap_converter.py suspicious_activity.pcap

# 3. Run prediction
python src/predict.py --input suspicious_activity_features.csv

# 4. Review results for attack types
```

## Available Demo Files

### In Repository
```
data/iot23/unlabeled_samples/
  ├── normal.csv (500 rows)
  ├── ddos.csv (200 rows)
  ├── port_scan.csv (200 rows)
  ├── malware_c2.csv (200 rows)
  ├── brute_force.csv (200 rows)
  └── sql_injection.csv (200 rows)
```

### Downloaded (After `python download_sample_pcap.py`)
```
downloads/sample_pcaps/
  ├── dns_sample.pcap (38 packets)
  ├── dns_sample_features.csv (12 flows)
  └── telnet_sample.pcap (1,166 packets)
```

### Real Dataset (After `python test_real_dataset.py`)
```
downloads/
  ├── nsl_kdd_test.csv (22,544 rows, original)
  ├── kdd_traffic.csv (22,544 rows, converted)
  └── kdd_traffic_sample.csv (100 rows, sample)
```

## Best Practices

### ✅ Do:
- Upload PCAP files directly to dashboard (easiest)
- Use `--max-packets` for files >10MB
- Capture enough traffic (at least 100 packets)
- Test with demo files first

### ❌ Don't:
- Upload Excel or JSON files (not supported)
- Include label columns in CSV (model predicts)
- Use very short captures (<10 packets)
- Upload encrypted/tunneled traffic only

## Help & Documentation

- **Full PCAP Guide**: `PCAP_SUPPORT_GUIDE.md`
- **Quick Start**: `QUICK_START.md`
- **Training Guide**: `TRAINING_VS_DETECTION.md`
- **Demo Guide**: `DEMONSTRATION_GUIDE.md`

## Support Commands

```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "scapy|torch|pandas|flask"

# Test PCAP converter
python src/pcap_converter.py --help

# Test dashboard
python src/dashboard_live.py

# Test prediction
python src/predict.py --help
```

---

**🎯 TL;DR**: Upload CSV (20 cols) or PCAP to dashboard → Get threat predictions!
