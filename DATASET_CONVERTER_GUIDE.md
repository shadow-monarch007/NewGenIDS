# 🔄 Dataset Converter - User Guide

## Overview
The Dataset Converter is now integrated into the NextGen IDS dashboard, making it super easy to convert external datasets (KDD, CICIDS, UNSW-NB15, etc.) to IoT-23 format for analysis.

## ✨ Features

### Auto-Detection
- **KDD Cup 99 / NSL-KDD**: Automatically detected by 41-42 numeric columns
- **CICIDS2017**: Detected by flow-based feature names
- **UNSW-NB15**: Detected by 'sbytes'/'dbytes' columns
- **Generic CSV**: Any unknown format (creates synthetic features)

### What It Does
1. Uploads your external dataset (CSV format)
2. Auto-detects the dataset format
3. Converts all features to IoT-23 format (20 features)
4. Downloads the converted file automatically
5. Ready to upload to Traffic Analysis tab!

## 📖 How to Use

### Step 1: Access the Dashboard
```powershell
python quick_start.py
```
- Open browser: http://localhost:8080
- Login: `admin` / `admin123` (or `demo` / `demo123`)

### Step 2: Navigate to Dataset Converter
- Click on the **"🔄 Dataset Converter"** tab (3rd tab in the navigation)

### Step 3: Upload Your Dataset
1. **Select CSV File**: Click "Choose File" and select your external dataset
   - Examples: `KDDTest+.csv`, `CICIDS_Friday.csv`, `UNSW_NB15.csv`

2. **Set Row Limit (Optional)**: 
   - For faster processing, enter a number (e.g., 5000)
   - Leave empty to convert the entire dataset
   - Recommended: 5000 rows for initial testing

3. **Click "Convert to IoT-23 Format"**
   - The converter will process your file
   - Converted file downloads automatically as `converted_<original_filename>.csv`

### Step 4: Analyze Converted Dataset
1. Go to **"🔍 Traffic Analysis"** tab
2. Upload the `converted_<filename>.csv` file
3. Click **"🔎 Analyze Traffic"**
4. Results appear in 2-5 seconds! ✅

## 🎯 Example Workflow

### Converting KDDTest+.csv
```
1. Download KDDTest+.csv from KDD repository
2. Open Dashboard → Dataset Converter tab
3. Upload KDDTest+.csv
4. Set row limit: 5000 (for speed)
5. Click "Convert to IoT-23 Format"
6. Downloaded: converted_KDDTest+.csv
7. Go to Traffic Analysis tab
8. Upload converted_KDDTest+.csv
9. Analyze → Results in 3 seconds!
```

## 📊 Supported Formats

### KDD Cup 99 / NSL-KDD
- **Columns**: 41-42 numeric features
- **Mapping**: 
  - Duration → flow_duration
  - src_bytes/dst_bytes → total_bytes
  - Service → protocol counts (DNS, HTTP, SSL)
  - Flags → TCP flag counts

### CICIDS2017
- **Columns**: 80+ features with flow-based names
- **Mapping**:
  - Flow Duration → flow_duration
  - Forward/Backward packets → total_packets
  - Byte totals → total_bytes
  - Flag counts → TCP flags

### UNSW-NB15
- **Columns**: 49 features
- **Mapping**:
  - dur → flow_duration
  - sbytes/dbytes → total_bytes
  - spkts/dpkts → total_packets
  - Protocol → protocol counts

### Generic CSV
- **Any format** that doesn't match above
- **Creates**: Synthetic features based on available numeric columns
- **Warning**: Less accurate than format-specific conversions

## 🔍 What Gets Converted

The converter transforms any dataset into **20 IoT-23 features**:

1. `packet_rate` - Packets per second
2. `packet_size` - Average packet size
3. `byte_rate` - Bytes per second
4. `flow_duration` - Flow duration in seconds
5. `total_packets` - Total packet count
6. `total_bytes` - Total byte count
7. `entropy` - Data entropy
8. `port_scan_score` - Port scanning indicator
9. `syn_flag_count` - SYN flag count
10. `ack_flag_count` - ACK flag count
11. `fin_flag_count` - FIN flag count
12. `rst_flag_count` - RST flag count
13. `psh_flag_count` - PSH flag count
14. `urg_flag_count` - URG flag count
15. `unique_src_ports` - Unique source ports
16. `unique_dst_ports` - Unique destination ports
17. `payload_entropy` - Payload data entropy
18. `dns_query_count` - DNS query count
19. `http_request_count` - HTTP request count
20. `ssl_handshake_count` - SSL handshake count

## ⚡ Performance Tips

### For Large Datasets (>100K rows)
- Use row limit: 5000-10000 for initial testing
- Full conversion may take 30-60 seconds
- Dashboard shows progress indicator

### For Multiple Files
- Convert one at a time
- Download completes automatically
- Clear inputs after each conversion

## 🐛 Troubleshooting

### "No file selected" Error
- Make sure you clicked "Choose File" and selected a CSV file

### "Conversion failed" Error
- Check if file is valid CSV format
- Try limiting rows to 5000
- Check browser console for detailed errors

### Download Not Starting
- Check browser's download permissions
- Try a different browser (Chrome/Edge recommended)

### Analysis Still Hangs After Conversion
- Verify converted file has exactly 20 columns
- Check file isn't corrupted (open in Excel/text editor)
- Try limiting to 1000 rows for testing

## 📝 Command-Line Alternative

If you prefer command-line, the standalone converter is still available:

```powershell
# Convert entire dataset
python convert_to_iot23.py --input KDDTest+.csv --output kdd_converted.csv

# Convert with row limit
python convert_to_iot23.py --input KDDTest+.csv --output kdd_converted.csv --max-rows 5000

# Short form
python convert_to_iot23.py -i dataset.csv -o converted.csv -m 5000
```

## ✅ Benefits

### Web Interface (Dashboard)
- ✅ User-friendly visual interface
- ✅ Auto-download converted files
- ✅ Immediate feedback with progress indicators
- ✅ Integrated workflow (convert → upload → analyze)
- ✅ No command-line knowledge needed

### Command-Line (Script)
- ✅ Batch processing multiple files
- ✅ Integration with automation scripts
- ✅ Detailed console output
- ✅ No browser required

## 🎉 Success!

You now have a fully integrated dataset converter in your dashboard! No more compatibility issues - just upload, convert, and analyze any network traffic dataset.

---

**Need Help?**
- Check `convert_to_iot23.py` for detailed conversion logic
- Review `PROJECT_COMPLETE_FINAL.md` for full project documentation
- Test with demo files first: `data/iot23/demo_samples/*.csv`
