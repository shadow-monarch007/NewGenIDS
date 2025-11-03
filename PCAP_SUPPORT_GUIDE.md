# PCAP File Support Guide

## Overview
The IDS dashboard now supports **PCAP (Packet Capture) files** in addition to CSV files! This allows you to upload raw network traffic captures directly from Wireshark, tcpdump, or other network monitoring tools.

## Supported File Formats

### ✅ CSV Files (.csv)
- Pre-processed traffic features (20 columns)
- Ready for immediate analysis
- No conversion needed

### ✅ PCAP Files (.pcap, .pcapng)
- Raw network packet captures
- Automatically converted to features
- Works with Wireshark captures

## How It Works

```
[PCAP File] → [Auto Converter] → [20 Features CSV] → [Model] → [Predictions]
     ↓              ↓                    ↓                ↓
  Raw packets   Extract traffic      Standard       Threat
  from network  characteristics      format        detection
```

## Features Extracted from PCAP

The converter automatically extracts 20 traffic features:

1. **Packet Statistics**: Rate, size, count, bytes
2. **TCP Flags**: SYN, ACK, FIN, RST, PSH, URG counts
3. **Port Analysis**: Unique source/destination ports
4. **Protocol Detection**: DNS, HTTP, SSL/TLS
5. **Security Indicators**: Entropy, port scan scores

## Quick Start

### Method 1: Upload via Dashboard
1. Go to http://localhost:5000
2. Click "Choose CSV or PCAP File"
3. Select your .pcap or .pcapng file
4. Click "Analyze Traffic"
5. System auto-converts and predicts!

### Method 2: Command Line Conversion
```bash
# Convert PCAP to CSV
python src/pcap_converter.py capture.pcap

# Convert with custom window size
python src/pcap_converter.py capture.pcap --window 10

# Limit packets for faster processing
python src/pcap_converter.py capture.pcap --max-packets 5000

# Specify output filename
python src/pcap_converter.py capture.pcap --output traffic.csv
```

## Download Sample PCAP Files

```bash
# Download test PCAP files from public sources
python download_sample_pcap.py
```

This downloads sample captures to `downloads/sample_pcaps/`:
- `http_sample.pcap` - HTTP traffic
- `dns_sample.pcap` - DNS queries
- `telnet_sample.pcap` - Telnet session

## Testing the PCAP Converter

### Test 1: Convert Sample PCAP
```bash
python download_sample_pcap.py
python src/pcap_converter.py downloads/sample_pcaps/http_sample.pcap
```

### Test 2: Upload to Dashboard
1. Start dashboard: `python src/dashboard_live.py`
2. Open http://localhost:5000
3. Upload `downloads/sample_pcaps/http_sample.pcap`
4. View threat predictions!

### Test 3: Convert Your Own Capture
```bash
# Capture traffic with tcpdump (Linux/Mac)
sudo tcpdump -i eth0 -w my_traffic.pcap -c 1000

# Or use Wireshark to save capture as .pcap

# Convert and analyze
python src/pcap_converter.py my_traffic.pcap
```

## Conversion Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--window` | Time window (seconds) for grouping packets | 5.0 | `--window 10` |
| `--max-packets` | Maximum packets to process | All | `--max-packets 10000` |
| `--output` | Output CSV filename | Auto-generated | `--output traffic.csv` |

## Performance Tips

### For Large PCAP Files:
1. **Limit packets**: Use `--max-packets` for faster processing
2. **Adjust window**: Larger windows = fewer flows to analyze
3. **Split file**: Use Wireshark's "Export Specified Packets"

### Recommended Settings:
```bash
# Fast analysis (testing)
python src/pcap_converter.py big.pcap --max-packets 5000 --window 10

# Detailed analysis (production)
python src/pcap_converter.py capture.pcap --window 5

# Very large files (GB+)
python src/pcap_converter.py huge.pcap --max-packets 50000 --window 15
```

## What Gets Analyzed

### PCAP Processing:
1. **Read packets** - Load PCAP with scapy
2. **Group by time** - Create 5-second windows (default)
3. **Extract features** - Calculate 20 traffic characteristics per window
4. **Save CSV** - Export features for model input

### Example Output:
```
Input:  network_capture.pcap (10 MB, 50,000 packets)
Process: Extract features from 5-second windows
Output: traffic_features.csv (200 flows × 20 features)
Result: Model analyzes 200 traffic flows
```

## Common Use Cases

### 1. Live Network Monitoring
```bash
# Capture live traffic
sudo tcpdump -i eth0 -w live.pcap -G 300 -W 1

# Convert every 5 minutes
python src/pcap_converter.py live.pcap --output live_features.csv

# Upload to dashboard for analysis
```

### 2. Security Investigation
```bash
# Analyze suspicious traffic capture
python src/pcap_converter.py suspicious.pcap

# Review predictions
python src/predict.py --input suspicious_features.csv
```

### 3. Penetration Testing
```bash
# Capture pen test traffic
# Convert and analyze
python src/pcap_converter.py pentest.pcap --window 3

# Identify attack patterns detected
```

## Troubleshooting

### Issue: "Error reading PCAP"
**Solution**: Ensure file is valid PCAP format
```bash
# Check with tshark
tshark -r your_file.pcap -c 10
```

### Issue: "No features extracted"
**Solution**: PCAP may be empty or corrupted
- Check packet count: `capinfos your_file.pcap`
- Verify packets exist: `tcpdump -r your_file.pcap -c 10`

### Issue: "Conversion too slow"
**Solution**: Limit packets or increase window
```bash
python src/pcap_converter.py big.pcap --max-packets 10000 --window 10
```

## Model Compatibility

✅ **Compatible**: PCAP files work perfectly because:
- Converter extracts same 20 features as training data
- Model sees identical input format
- No accuracy loss

❌ **Not Compatible**:
- PCAP with no TCP/IP traffic
- Encrypted tunneled traffic (can't extract features)
- Very short captures (<10 packets)

## Integration with Dashboard

The dashboard automatically:
1. Detects file type (.csv vs .pcap)
2. Converts PCAP to CSV if needed
3. Runs prediction on features
4. Displays threat analysis
5. Cleans up temporary files

No manual conversion needed when uploading via dashboard!

## Advanced: Custom Feature Extraction

Edit `src/pcap_converter.py` to modify:
- Time window size (default: 5 seconds)
- Feature calculations
- Protocol detection logic
- Packet filtering

## Next Steps

1. **Download samples**: `python download_sample_pcap.py`
2. **Test converter**: `python src/pcap_converter.py downloads/sample_pcaps/http_sample.pcap`
3. **Upload to dashboard**: Visit http://localhost:5000
4. **Capture your traffic**: Use Wireshark/tcpdump
5. **Analyze**: Upload your PCAP files!

## Resources

- **Wireshark**: https://www.wireshark.org/
- **Sample captures**: https://wiki.wireshark.org/SampleCaptures
- **tcpdump guide**: https://www.tcpdump.org/
- **Scapy docs**: https://scapy.readthedocs.io/

---

**✅ System is now ready for real-world network analysis!**
