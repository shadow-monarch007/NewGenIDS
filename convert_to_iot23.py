"""
Universal Dataset Converter to IoT-23 Format
---------------------------------------------
Converts external datasets (KDD, CICIDS, UNSW-NB15, etc.) to IoT-23 format
so they can be analyzed by the trained IDS model.

Usage:
    python convert_to_iot23.py --input KDDTest+.csv --output kdd_converted.csv
    python convert_to_iot23.py --input cicids_friday.csv --output cicids_converted.csv
    python convert_to_iot23.py --input any_dataset.csv --output converted.csv
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

# IoT-23 feature names (20 features expected by the model)
IOT23_FEATURES = [
    'packet_rate',
    'packet_size',
    'byte_rate',
    'flow_duration',
    'total_packets',
    'total_bytes',
    'entropy',
    'port_scan_score',
    'syn_flag_count',
    'ack_flag_count',
    'fin_flag_count',
    'rst_flag_count',
    'psh_flag_count',
    'urg_flag_count',
    'unique_src_ports',
    'unique_dst_ports',
    'payload_entropy',
    'dns_query_count',
    'http_request_count',
    'ssl_handshake_count'
]


def detect_dataset_format(df):
    """
    Auto-detect dataset type based on column names/structure
    """
    cols = [str(c).lower() for c in df.columns]
    
    # Check for KDD/NSL-KDD
    if len(df.columns) in [41, 42, 43] and all(isinstance(c, int) for c in df.columns[:10]):
        return 'kdd'
    
    # Check for CICIDS2017
    if any('flow' in c and 'duration' in c for c in cols):
        return 'cicids'
    
    # Check for UNSW-NB15
    if 'sbytes' in cols or 'dbytes' in cols:
        return 'unsw'
    
    # Check if already IoT-23 format
    if all(f in cols for f in ['packet_rate', 'byte_rate', 'flow_duration']):
        return 'iot23'
    
    # Generic/Unknown
    return 'generic'


def convert_kdd_to_iot23(df):
    """
    Convert KDD/NSL-KDD dataset to IoT-23 format
    
    KDD Columns (42 total):
    0: duration, 1: protocol_type, 2: service, 3: flag,
    4: src_bytes, 5: dst_bytes, 6: land, 7: wrong_fragment,
    8: urgent, 9: hot, 10: num_failed_logins, ...
    41: label (attack type)
    """
    print("📊 Converting KDD/NSL-KDD format...")
    
    iot23 = pd.DataFrame()
    
    # Column 0: duration (seconds)
    duration = df.iloc[:, 0].clip(lower=0.01)  # Avoid division by zero
    
    # Column 4: src_bytes, Column 5: dst_bytes
    src_bytes = df.iloc[:, 4]
    dst_bytes = df.iloc[:, 5]
    total_bytes_val = src_bytes + dst_bytes
    
    # Estimate packets (assuming avg packet size ~1500 bytes)
    total_packets_val = (total_bytes_val / 1500).clip(lower=1).astype(int)
    
    # Calculate IoT-23 features
    iot23['packet_rate'] = (total_packets_val / duration).clip(upper=10000)
    iot23['packet_size'] = (total_bytes_val / total_packets_val).clip(upper=65535)
    iot23['byte_rate'] = (total_bytes_val / duration).clip(upper=1e9)
    iot23['flow_duration'] = duration
    iot23['total_packets'] = total_packets_val
    iot23['total_bytes'] = total_bytes_val
    
    # Entropy (can't calculate from KDD, use reasonable estimates)
    iot23['entropy'] = np.random.uniform(2.0, 7.0, len(df))
    
    # Port scan score (based on service diversity)
    # Column 22: count (connections to same host)
    if df.shape[1] > 22:
        iot23['port_scan_score'] = df.iloc[:, 22].clip(upper=100)
    else:
        iot23['port_scan_score'] = 0
    
    # TCP Flags (approximate from 'flag' column - Column 3)
    flag_col = df.iloc[:, 3].astype(str) if df.shape[1] > 3 else pd.Series(['SF'] * len(df))
    iot23['syn_flag_count'] = (flag_col.str.contains('S0|S1|S2|S3', case=False, na=False)).astype(int) * 5
    iot23['ack_flag_count'] = (flag_col == 'SF').astype(int) * 5
    iot23['fin_flag_count'] = (flag_col.str.contains('SF|S1', case=False, na=False)).astype(int) * 3
    iot23['rst_flag_count'] = (flag_col.str.contains('REJ|RSTO|RSTR', case=False, na=False)).astype(int) * 5
    iot23['psh_flag_count'] = (flag_col == 'SF').astype(int) * 2
    iot23['urg_flag_count'] = (df.iloc[:, 8] > 0).astype(int) if df.shape[1] > 8 else 0  # urgent column
    
    # Port statistics (limited in KDD)
    iot23['unique_src_ports'] = 1
    iot23['unique_dst_ports'] = 1
    
    # Payload entropy (estimate)
    iot23['payload_entropy'] = np.random.uniform(3.0, 6.0, len(df))
    
    # Protocol-specific counts (based on 'service' column - Column 2)
    service_col = df.iloc[:, 2].astype(str) if df.shape[1] > 2 else pd.Series(['other'] * len(df))
    iot23['dns_query_count'] = (service_col.str.contains('domain', case=False, na=False)).astype(int) * 10
    iot23['http_request_count'] = (service_col.str.contains('http', case=False, na=False)).astype(int) * 5
    iot23['ssl_handshake_count'] = (service_col.str.contains('ssl|https', case=False, na=False)).astype(int) * 3
    
    # Fill any NaN values
    iot23 = iot23.fillna(0)
    
    return iot23


def convert_cicids_to_iot23(df):
    """
    Convert CICIDS2017 dataset to IoT-23 format
    
    CICIDS has many columns with different naming conventions
    """
    print("📊 Converting CICIDS2017 format...")
    
    iot23 = pd.DataFrame()
    
    # CICIDS column mappings (case-insensitive search)
    cols = {c.lower(): c for c in df.columns}
    
    def get_col(patterns):
        """Find column matching any pattern"""
        for pattern in patterns:
            for key, orig in cols.items():
                if pattern in key:
                    return df[orig]
        return pd.Series([0] * len(df))
    
    # Duration
    duration = get_col(['flow duration', 'duration']).clip(lower=0.001) / 1e6  # Convert microseconds to seconds
    
    # Bytes
    fwd_bytes = get_col(['total fwd packet', 'fwd packet length total', 'totlen_fwd'])
    bwd_bytes = get_col(['total bwd packet', 'bwd packet length total', 'totlen_bwd'])
    total_bytes_val = (fwd_bytes + bwd_bytes).clip(lower=1)
    
    # Packets
    fwd_packets = get_col(['total fwd packet', 'fwd packets', 'tot_fwd'])
    bwd_packets = get_col(['total bwd packet', 'bwd packets', 'tot_bwd'])
    total_packets_val = (fwd_packets + bwd_packets).clip(lower=1)
    
    # Calculate features
    iot23['packet_rate'] = (total_packets_val / duration).clip(upper=10000)
    iot23['packet_size'] = (total_bytes_val / total_packets_val).clip(upper=65535)
    iot23['byte_rate'] = (total_bytes_val / duration).clip(upper=1e9)
    iot23['flow_duration'] = duration
    iot23['total_packets'] = total_packets_val
    iot23['total_bytes'] = total_bytes_val
    
    # Entropy
    iot23['entropy'] = get_col(['entropy']).fillna(np.random.uniform(2.0, 7.0, len(df)))
    
    # Port scan score
    iot23['port_scan_score'] = get_col(['destination port']).clip(upper=100)
    
    # TCP Flags
    iot23['syn_flag_count'] = get_col(['syn flag count', 'syn_flag']).fillna(0)
    iot23['ack_flag_count'] = get_col(['ack flag count', 'ack_flag']).fillna(0)
    iot23['fin_flag_count'] = get_col(['fin flag count', 'fin_flag']).fillna(0)
    iot23['rst_flag_count'] = get_col(['rst flag count', 'rst_flag']).fillna(0)
    iot23['psh_flag_count'] = get_col(['psh flag count', 'psh_flag']).fillna(0)
    iot23['urg_flag_count'] = get_col(['urg flag count', 'urg_flag']).fillna(0)
    
    # Port statistics
    iot23['unique_src_ports'] = get_col(['source port']).fillna(1)
    iot23['unique_dst_ports'] = get_col(['destination port']).fillna(1)
    
    # Payload entropy
    iot23['payload_entropy'] = np.random.uniform(3.0, 6.0, len(df))
    
    # Protocol-specific
    protocol = get_col(['protocol'])
    iot23['dns_query_count'] = (protocol == 17).astype(int) * 5  # UDP (often DNS)
    iot23['http_request_count'] = get_col(['destination port']).isin([80, 8080]).astype(int) * 5
    iot23['ssl_handshake_count'] = get_col(['destination port']).isin([443, 8443]).astype(int) * 3
    
    iot23 = iot23.fillna(0)
    return iot23


def convert_unsw_to_iot23(df):
    """
    Convert UNSW-NB15 dataset to IoT-23 format
    """
    print("📊 Converting UNSW-NB15 format...")
    
    iot23 = pd.DataFrame()
    
    # Duration (in seconds)
    duration = df.get('dur', pd.Series([1.0] * len(df))).clip(lower=0.001)
    
    # Bytes
    src_bytes = df.get('sbytes', pd.Series([0] * len(df)))
    dst_bytes = df.get('dbytes', pd.Series([0] * len(df)))
    total_bytes_val = (src_bytes + dst_bytes).clip(lower=1)
    
    # Packets
    src_packets = df.get('spkts', pd.Series([1] * len(df)))
    dst_packets = df.get('dpkts', pd.Series([1] * len(df)))
    total_packets_val = (src_packets + dst_packets).clip(lower=1)
    
    # Features
    iot23['packet_rate'] = (total_packets_val / duration).clip(upper=10000)
    iot23['packet_size'] = (total_bytes_val / total_packets_val).clip(upper=65535)
    iot23['byte_rate'] = (total_bytes_val / duration).clip(upper=1e9)
    iot23['flow_duration'] = duration
    iot23['total_packets'] = total_packets_val
    iot23['total_bytes'] = total_bytes_val
    iot23['entropy'] = np.random.uniform(2.0, 7.0, len(df))
    iot23['port_scan_score'] = df.get('dport', 0).clip(upper=100)
    
    # TCP flags (if available)
    iot23['syn_flag_count'] = df.get('swin', 0).apply(lambda x: 5 if x > 0 else 0)
    iot23['ack_flag_count'] = df.get('dwin', 0).apply(lambda x: 5 if x > 0 else 0)
    iot23['fin_flag_count'] = df.get('tcprtt', 0).apply(lambda x: 3 if x > 0 else 0)
    iot23['rst_flag_count'] = 0
    iot23['psh_flag_count'] = 0
    iot23['urg_flag_count'] = 0
    
    # Ports
    iot23['unique_src_ports'] = 1
    iot23['unique_dst_ports'] = 1
    iot23['payload_entropy'] = np.random.uniform(3.0, 6.0, len(df))
    
    # Protocol-specific
    proto = df.get('proto', 'other')
    iot23['dns_query_count'] = (proto == 'udp').astype(int) * 5
    iot23['http_request_count'] = df.get('dport', 0).isin([80, 8080]).astype(int) * 5
    iot23['ssl_handshake_count'] = df.get('dport', 0).isin([443, 8443]).astype(int) * 3
    
    iot23 = iot23.fillna(0)
    return iot23


def convert_generic_to_iot23(df):
    """
    Generic converter for unknown formats
    Tries to extract basic network features
    """
    print("📊 Converting generic/unknown format...")
    print("⚠️  Warning: Unknown format. Creating synthetic features.")
    
    iot23 = pd.DataFrame()
    n = len(df)
    
    # Try to find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) >= 5:
        # Use first numeric columns as proxies
        col1 = df[numeric_cols[0]].clip(lower=0.01)
        col2 = df[numeric_cols[1]].clip(lower=1)
        col3 = df[numeric_cols[2]].clip(lower=1)
        
        iot23['packet_rate'] = (col2 / col1).clip(upper=10000)
        iot23['packet_size'] = col3.clip(upper=65535)
        iot23['byte_rate'] = (col2 * col3 / col1).clip(upper=1e9)
        iot23['flow_duration'] = col1
        iot23['total_packets'] = col2
        iot23['total_bytes'] = col2 * col3
    else:
        # Create synthetic features
        iot23['packet_rate'] = np.random.uniform(10, 1000, n)
        iot23['packet_size'] = np.random.uniform(64, 1500, n)
        iot23['byte_rate'] = iot23['packet_rate'] * iot23['packet_size']
        iot23['flow_duration'] = np.random.uniform(1, 100, n)
        iot23['total_packets'] = iot23['packet_rate'] * iot23['flow_duration']
        iot23['total_bytes'] = iot23['byte_rate'] * iot23['flow_duration']
    
    # Fill remaining features with defaults
    iot23['entropy'] = np.random.uniform(2.0, 7.0, n)
    iot23['port_scan_score'] = np.random.uniform(0, 50, n)
    iot23['syn_flag_count'] = np.random.randint(0, 10, n)
    iot23['ack_flag_count'] = np.random.randint(0, 10, n)
    iot23['fin_flag_count'] = np.random.randint(0, 5, n)
    iot23['rst_flag_count'] = np.random.randint(0, 5, n)
    iot23['psh_flag_count'] = np.random.randint(0, 5, n)
    iot23['urg_flag_count'] = np.random.randint(0, 2, n)
    iot23['unique_src_ports'] = 1
    iot23['unique_dst_ports'] = 1
    iot23['payload_entropy'] = np.random.uniform(3.0, 6.0, n)
    iot23['dns_query_count'] = np.random.randint(0, 5, n)
    iot23['http_request_count'] = np.random.randint(0, 10, n)
    iot23['ssl_handshake_count'] = np.random.randint(0, 3, n)
    
    iot23 = iot23.fillna(0)
    return iot23


def convert_dataset(input_file, output_file, max_rows=None):
    """
    Main conversion function
    """
    print(f"\n{'='*70}")
    print(f"🔄 Converting Dataset to IoT-23 Format")
    print(f"{'='*70}")
    print(f"📁 Input:  {input_file}")
    print(f"📁 Output: {output_file}")
    print(f"{'='*70}\n")
    
    # Check if input exists
    if not Path(input_file).exists():
        print(f"❌ Error: Input file not found: {input_file}")
        return False
    
    try:
        # Load dataset
        print("📖 Reading input file...")
        try:
            df = pd.read_csv(input_file, low_memory=False)
        except:
            # Try without header
            df = pd.read_csv(input_file, header=None, low_memory=False)
        
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Limit rows if specified
        if max_rows and len(df) > max_rows:
            print(f"⚠️  Limiting to first {max_rows} rows (faster processing)")
            df = df.head(max_rows)
        
        # Detect format
        dataset_type = detect_dataset_format(df)
        print(f"🔍 Detected format: {dataset_type.upper()}")
        
        # Convert based on type
        if dataset_type == 'iot23':
            print("✅ Already in IoT-23 format! Copying...")
            iot23_df = df
        elif dataset_type == 'kdd':
            iot23_df = convert_kdd_to_iot23(df)
        elif dataset_type == 'cicids':
            iot23_df = convert_cicids_to_iot23(df)
        elif dataset_type == 'unsw':
            iot23_df = convert_unsw_to_iot23(df)
        else:
            iot23_df = convert_generic_to_iot23(df)
        
        # Verify we have all 20 features
        missing_features = set(IOT23_FEATURES) - set(iot23_df.columns)
        if missing_features:
            print(f"⚠️  Warning: Missing features: {missing_features}")
            for feat in missing_features:
                iot23_df[feat] = 0
        
        # Reorder columns to match IoT-23
        iot23_df = iot23_df[IOT23_FEATURES]
        
        # Remove any infinite values
        iot23_df = iot23_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Save converted dataset
        print(f"\n💾 Saving converted dataset...")
        iot23_df.to_csv(output_file, index=False)
        
        # Show statistics
        print(f"\n{'='*70}")
        print(f"✅ Conversion Complete!")
        print(f"{'='*70}")
        print(f"📊 Output Statistics:")
        print(f"   Rows:     {len(iot23_df)}")
        print(f"   Columns:  {len(iot23_df.columns)}")
        print(f"   Size:     {Path(output_file).stat().st_size / 1024:.1f} KB")
        print(f"\n📋 Sample of converted data:")
        print(iot23_df.head(3).to_string())
        print(f"\n{'='*70}")
        print(f"✅ Ready to upload: {output_file}")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert external datasets to IoT-23 format for IDS analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_to_iot23.py --input KDDTest+.csv --output kdd_converted.csv
  python convert_to_iot23.py --input cicids_friday.csv --output cicids_converted.csv
  python convert_to_iot23.py --input unknown.csv --output converted.csv --max-rows 10000

Supported Formats:
  - KDD Cup 99 / NSL-KDD (auto-detected)
  - CICIDS2017 (auto-detected)
  - UNSW-NB15 (auto-detected)
  - Generic CSV (will create synthetic features)
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    parser.add_argument('--max-rows', '-m', type=int, help='Limit number of rows (for faster processing)')
    
    args = parser.parse_args()
    
    success = convert_dataset(args.input, args.output, args.max_rows)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
