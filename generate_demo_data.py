"""
Enhanced Synthetic Data Generator for Next-Gen IDS
-------------------------------------------------
Generates realistic network traffic data with multiple attack types
for demonstration purposes. Each attack has distinct characteristics.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_normal_traffic(n_samples=1000):
    """Generate normal/benign network traffic."""
    data = {
        'timestamp': [datetime.now() + timedelta(seconds=i) for i in range(n_samples)],
        'src_port': np.random.randint(1024, 65535, n_samples),
        'dst_port': np.random.choice([80, 443, 22, 21, 25, 3306], n_samples),
        'protocol': np.random.choice([6, 17], n_samples),  # TCP=6, UDP=17
        'packet_size': np.random.normal(500, 150, n_samples).clip(64, 1500),
        'packet_rate': np.random.normal(10, 3, n_samples).clip(1, 50),
        'byte_rate': np.random.normal(5000, 1500, n_samples).clip(500, 20000),
        'flow_duration': np.random.normal(30, 10, n_samples).clip(1, 300),
        'total_packets': np.random.normal(100, 30, n_samples).clip(10, 500),
        'total_bytes': np.random.normal(50000, 15000, n_samples).clip(5000, 200000),
        'tcp_flags_syn': np.random.binomial(1, 0.1, n_samples),
        'tcp_flags_ack': np.random.binomial(1, 0.8, n_samples),
        'tcp_flags_fin': np.random.binomial(1, 0.1, n_samples),
        'tcp_flags_rst': np.random.binomial(1, 0.05, n_samples),
        'tcp_flags_psh': np.random.binomial(1, 0.3, n_samples),
        'icmp_type': np.zeros(n_samples),
        'connection_state': np.random.choice([1, 2, 3], n_samples),  # Established, SYN_SENT, etc.
        'service': np.random.choice([0, 1, 2, 3, 4], n_samples),  # HTTP, HTTPS, SSH, FTP, SMTP
        'entropy': np.random.normal(7.0, 0.5, n_samples).clip(5, 8),
        'label': np.zeros(n_samples, dtype=int),  # 0 = Normal
        'attack_type': ['Normal'] * n_samples
    }
    return pd.DataFrame(data)

def generate_ddos_attack(n_samples=500):
    """Generate DDoS attack traffic - high packet rate from multiple sources."""
    data = {
        'timestamp': [datetime.now() + timedelta(seconds=i) for i in range(n_samples)],
        'src_port': np.random.randint(1024, 65535, n_samples),
        'dst_port': np.full(n_samples, 80),  # All targeting port 80
        'protocol': np.full(n_samples, 6),  # TCP
        'packet_size': np.random.normal(64, 10, n_samples).clip(40, 100),  # Small packets
        'packet_rate': np.random.normal(1000, 200, n_samples).clip(500, 5000),  # Very high rate!
        'byte_rate': np.random.normal(64000, 15000, n_samples).clip(30000, 300000),
        'flow_duration': np.random.normal(1, 0.5, n_samples).clip(0.1, 5),  # Short flows
        'total_packets': np.random.normal(1000, 200, n_samples).clip(500, 5000),
        'total_bytes': np.random.normal(64000, 15000, n_samples).clip(30000, 300000),
        'tcp_flags_syn': np.ones(n_samples),  # SYN flood
        'tcp_flags_ack': np.zeros(n_samples),  # No ACK (incomplete handshake)
        'tcp_flags_fin': np.zeros(n_samples),
        'tcp_flags_rst': np.zeros(n_samples),
        'tcp_flags_psh': np.zeros(n_samples),
        'icmp_type': np.zeros(n_samples),
        'connection_state': np.zeros(n_samples),  # No established connection
        'service': np.zeros(n_samples),  # HTTP
        'entropy': np.random.normal(3.0, 0.5, n_samples).clip(2, 4),  # Low entropy (repetitive)
        'label': np.ones(n_samples, dtype=int),  # 1 = Attack
        'attack_type': ['DDoS'] * n_samples
    }
    return pd.DataFrame(data)

def generate_port_scan(n_samples=300):
    """Generate port scanning attack - sequential port probing."""
    data = {
        'timestamp': [datetime.now() + timedelta(seconds=i*0.1) for i in range(n_samples)],
        'src_port': np.random.randint(40000, 50000, n_samples),
        'dst_port': np.arange(1, n_samples + 1),  # Sequential ports!
        'protocol': np.full(n_samples, 6),  # TCP
        'packet_size': np.random.normal(60, 5, n_samples).clip(50, 80),  # Tiny packets
        'packet_rate': np.random.normal(100, 20, n_samples).clip(50, 300),
        'byte_rate': np.random.normal(6000, 1000, n_samples).clip(3000, 15000),
        'flow_duration': np.random.normal(0.1, 0.05, n_samples).clip(0.01, 0.5),  # Very short
        'total_packets': np.random.normal(3, 1, n_samples).clip(1, 10),  # Few packets per flow
        'total_bytes': np.random.normal(180, 50, n_samples).clip(60, 500),
        'tcp_flags_syn': np.ones(n_samples),  # All SYN (connection attempts)
        'tcp_flags_ack': np.zeros(n_samples),
        'tcp_flags_fin': np.zeros(n_samples),
        'tcp_flags_rst': np.random.binomial(1, 0.8, n_samples),  # Many RST (refused connections)
        'tcp_flags_psh': np.zeros(n_samples),
        'icmp_type': np.zeros(n_samples),
        'connection_state': np.zeros(n_samples),  # No established connections
        'service': np.full(n_samples, 99),  # Unknown service
        'entropy': np.random.normal(4.0, 0.3, n_samples).clip(3, 5),
        'label': np.ones(n_samples, dtype=int),
        'attack_type': ['Port_Scan'] * n_samples
    }
    return pd.DataFrame(data)

def generate_malware_c2(n_samples=200):
    """Generate malware command & control traffic - periodic beaconing."""
    # Beaconing pattern: regular intervals
    timestamps = []
    for i in range(n_samples):
        # Every 60 seconds with small jitter
        timestamps.append(datetime.now() + timedelta(seconds=i*60 + np.random.normal(0, 5)))
    
    data = {
        'timestamp': timestamps,
        'src_port': np.random.randint(40000, 60000, n_samples),
        'dst_port': np.random.choice([443, 8080, 8443], n_samples),  # Common C2 ports
        'protocol': np.full(n_samples, 6),  # TCP
        'packet_size': np.random.normal(200, 30, n_samples).clip(100, 400),
        'packet_rate': np.random.normal(5, 2, n_samples).clip(1, 15),  # Low, steady rate
        'byte_rate': np.random.normal(1000, 200, n_samples).clip(500, 3000),
        'flow_duration': np.random.normal(5, 2, n_samples).clip(1, 15),
        'total_packets': np.random.normal(25, 8, n_samples).clip(5, 50),
        'total_bytes': np.random.normal(5000, 1500, n_samples).clip(1000, 15000),
        'tcp_flags_syn': np.random.binomial(1, 0.2, n_samples),
        'tcp_flags_ack': np.ones(n_samples),  # Established connections
        'tcp_flags_fin': np.random.binomial(1, 0.2, n_samples),
        'tcp_flags_rst': np.zeros(n_samples),
        'tcp_flags_psh': np.ones(n_samples),  # Pushing data
        'icmp_type': np.zeros(n_samples),
        'connection_state': np.full(n_samples, 1),  # Established
        'service': np.full(n_samples, 1),  # HTTPS
        'entropy': np.random.normal(7.5, 0.3, n_samples).clip(7, 8),  # High entropy (encrypted)
        'label': np.ones(n_samples, dtype=int),
        'attack_type': ['Malware_C2'] * n_samples
    }
    return pd.DataFrame(data)

def generate_brute_force(n_samples=250):
    """Generate brute force attack - repeated login attempts."""
    data = {
        'timestamp': [datetime.now() + timedelta(seconds=i*2) for i in range(n_samples)],
        'src_port': np.random.randint(40000, 50000, n_samples),
        'dst_port': np.random.choice([22, 3389, 21], n_samples),  # SSH, RDP, FTP
        'protocol': np.full(n_samples, 6),  # TCP
        'packet_size': np.random.normal(150, 30, n_samples).clip(80, 300),
        'packet_rate': np.random.normal(30, 10, n_samples).clip(10, 100),  # Moderate rate
        'byte_rate': np.random.normal(4500, 1000, n_samples).clip(2000, 10000),
        'flow_duration': np.random.normal(2, 1, n_samples).clip(0.5, 10),
        'total_packets': np.random.normal(60, 20, n_samples).clip(20, 200),
        'total_bytes': np.random.normal(9000, 3000, n_samples).clip(3000, 30000),
        'tcp_flags_syn': np.random.binomial(1, 0.3, n_samples),
        'tcp_flags_ack': np.random.binomial(1, 0.7, n_samples),
        'tcp_flags_fin': np.random.binomial(1, 0.9, n_samples),  # Many connection terminations
        'tcp_flags_rst': np.random.binomial(1, 0.5, n_samples),  # Failed attempts
        'tcp_flags_psh': np.random.binomial(1, 0.5, n_samples),
        'icmp_type': np.zeros(n_samples),
        'connection_state': np.random.choice([1, 2], n_samples),
        'service': np.random.choice([2, 5, 3], n_samples),  # SSH, RDP, FTP
        'entropy': np.random.normal(6.0, 0.5, n_samples).clip(5, 7),
        'label': np.ones(n_samples, dtype=int),
        'attack_type': ['Brute_Force'] * n_samples
    }
    return pd.DataFrame(data)

def generate_sql_injection(n_samples=150):
    """Generate SQL injection attack - abnormal HTTP requests."""
    data = {
        'timestamp': [datetime.now() + timedelta(seconds=i*5) for i in range(n_samples)],
        'src_port': np.random.randint(40000, 60000, n_samples),
        'dst_port': np.random.choice([80, 443, 8080], n_samples),
        'protocol': np.full(n_samples, 6),  # TCP
        'packet_size': np.random.normal(800, 200, n_samples).clip(500, 1500),  # Large requests
        'packet_rate': np.random.normal(15, 5, n_samples).clip(5, 50),
        'byte_rate': np.random.normal(12000, 3000, n_samples).clip(5000, 30000),
        'flow_duration': np.random.normal(3, 1, n_samples).clip(1, 10),
        'total_packets': np.random.normal(45, 15, n_samples).clip(15, 150),
        'total_bytes': np.random.normal(36000, 10000, n_samples).clip(15000, 100000),
        'tcp_flags_syn': np.random.binomial(1, 0.2, n_samples),
        'tcp_flags_ack': np.ones(n_samples),
        'tcp_flags_fin': np.random.binomial(1, 0.2, n_samples),
        'tcp_flags_rst': np.random.binomial(1, 0.1, n_samples),
        'tcp_flags_psh': np.ones(n_samples),  # Pushing data
        'icmp_type': np.zeros(n_samples),
        'connection_state': np.full(n_samples, 1),
        'service': np.random.choice([0, 1], n_samples),  # HTTP/HTTPS
        'entropy': np.random.normal(6.5, 0.5, n_samples).clip(5.5, 7.5),
        'label': np.ones(n_samples, dtype=int),
        'attack_type': ['SQL_Injection'] * n_samples
    }
    return pd.DataFrame(data)

def generate_mixed_dataset(output_dir='data/iot23', filename='demo_attacks.csv'):
    """Generate comprehensive dataset with all attack types."""
    print("🔧 Generating synthetic attack data for demonstration...\n")
    
    # Generate each traffic type
    print("  [1/7] Normal traffic (2000 samples)...")
    normal = generate_normal_traffic(2000)
    
    print("  [2/7] DDoS attack (800 samples)...")
    ddos = generate_ddos_attack(800)
    
    print("  [3/7] Port scan (500 samples)...")
    port_scan = generate_port_scan(500)
    
    print("  [4/7] Malware C2 (400 samples)...")
    malware = generate_malware_c2(400)
    
    print("  [5/7] Brute force (400 samples)...")
    brute_force = generate_brute_force(400)
    
    print("  [6/7] SQL injection (300 samples)...")
    sql_injection = generate_sql_injection(300)
    
    # Combine all datasets
    print("  [7/7] Combining and shuffling...")
    combined = pd.concat([normal, ddos, port_scan, malware, brute_force, sql_injection], 
                         ignore_index=True)
    
    # Shuffle to mix attack types
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add additional metadata
    combined['flow_id'] = [f'flow_{i:06d}' for i in range(len(combined))]
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Drop timestamp column for CSV (convert to seconds since start)
    start_time = combined['timestamp'].min()
    combined['time_offset'] = (combined['timestamp'] - start_time).dt.total_seconds()
    combined = combined.drop('timestamp', axis=1)
    
    # Reorder columns
    feature_cols = [col for col in combined.columns if col not in ['label', 'attack_type', 'flow_id', 'time_offset']]
    combined = combined[['flow_id', 'time_offset'] + feature_cols + ['label', 'attack_type']]
    
    combined.to_csv(filepath, index=False)
    
    # Print statistics
    print(f"\n✅ Generated {len(combined):,} samples")
    print(f"📁 Saved to: {filepath}\n")
    
    print("📊 Attack Distribution:")
    print(combined['attack_type'].value_counts().to_string())
    
    print("\n📈 Label Distribution:")
    print(f"  Normal (0): {(combined['label'] == 0).sum():,} samples")
    print(f"  Attack (1): {(combined['label'] == 1).sum():,} samples")
    
    print("\n🔍 Sample Statistics:")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Feature names: {', '.join(feature_cols[:5])}...")
    
    print("\n💡 Demo Tips:")
    print("  • Use this file in dashboard for impressive multi-attack demo")
    print("  • Shows all attack types: DDoS, Port Scan, Malware C2, Brute Force, SQL Injection")
    print("  • Each attack has realistic, distinct patterns")
    print("  • AI explanations will identify attack characteristics accurately")
    
    return combined

def generate_individual_attack_files():
    """Generate separate CSV files for each attack type (for focused demos)."""
    print("\n🎯 Generating individual attack files...\n")
    
    output_dir = 'data/iot23/demo_samples'
    os.makedirs(output_dir, exist_ok=True)
    
    attacks = {
        'normal.csv': generate_normal_traffic(500),
        'ddos.csv': generate_ddos_attack(200),
        'port_scan.csv': generate_port_scan(150),
        'malware_c2.csv': generate_malware_c2(100),
        'brute_force.csv': generate_brute_force(100),
        'sql_injection.csv': generate_sql_injection(100)
    }
    
    for filename, df in attacks.items():
        # Process timestamps
        start_time = df['timestamp'].min()
        df['time_offset'] = (df['timestamp'] - start_time).dt.total_seconds()
        df = df.drop('timestamp', axis=1)
        
        # Add flow IDs
        df['flow_id'] = [f'flow_{i:06d}' for i in range(len(df))]
        
        # Reorder columns
        feature_cols = [col for col in df.columns if col not in ['label', 'attack_type', 'flow_id', 'time_offset']]
        df = df[['flow_id', 'time_offset'] + feature_cols + ['label', 'attack_type']]
        
        # Save
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"  ✅ {filename}: {len(df)} samples")
    
    print(f"\n📁 Individual files saved to: {output_dir}")
    print("💡 Use these for focused attack demonstrations")

if __name__ == "__main__":
    print("=" * 70)
    print("  Next-Gen IDS - Enhanced Demo Data Generator")
    print("=" * 70)
    print()
    
    # Generate main mixed dataset
    combined_df = generate_mixed_dataset()
    
    # Generate individual attack files
    generate_individual_attack_files()
    
    print("\n" + "=" * 70)
    print("  Data Generation Complete! 🎉")
    print("=" * 70)
    print("\n📝 Next Steps:")
    print("  1. Upload 'demo_attacks.csv' in dashboard for full demo")
    print("  2. Or use individual files in 'demo_samples/' for focused demos")
    print("  3. Train with --use-arnn flag to see A-RNN in action")
    print("  4. Check AI explanations for each attack type")
    print("\n🚀 Ready for demonstration!")
