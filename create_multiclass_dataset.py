"""
Create Multi-Class Attack Dataset
----------------------------------
Generates synthetic network traffic data with multiple attack types for training.
This will enable the IDS to distinguish between different attack categories.

Attack Types:
0 - Normal (benign traffic)
1 - DDoS (high packet rate, small packets, many connections)
2 - Port_Scan (sequential port access, short connections)
3 - Malware_C2 (periodic beacons, specific ports)
4 - Brute_Force (repeated auth attempts, failed connections)
5 - SQL_Injection (web ports, large payloads)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(42)

# Configuration
NUM_SAMPLES_PER_CLASS = 500
OUTPUT_DIR = "data/iot23"

def generate_normal_traffic(n_samples):
    """Generate normal/benign traffic patterns."""
    return {
        'src_port': np.random.randint(49152, 65535, n_samples),
        'dst_port': np.random.choice([80, 443, 8080, 22, 3389], n_samples),
        'protocol': np.random.choice([6, 17], n_samples),  # TCP/UDP
        'packet_size': np.random.normal(500, 200, n_samples).clip(64, 1500),
        'packet_rate': np.random.normal(10, 5, n_samples).clip(1, 50),
        'byte_rate': np.random.normal(5000, 2000, n_samples).clip(500, 20000),
        'flow_duration': np.random.exponential(60, n_samples).clip(1, 300),
        'total_packets': np.random.poisson(50, n_samples).clip(5, 200),
        'total_bytes': np.random.poisson(50000, n_samples).clip(5000, 200000),
        'tcp_flags_syn': np.random.binomial(1, 0.3, n_samples),
        'tcp_flags_ack': np.random.binomial(1, 0.8, n_samples),
        'tcp_flags_fin': np.random.binomial(1, 0.3, n_samples),
        'tcp_flags_rst': np.random.binomial(1, 0.05, n_samples),
        'entropy': np.random.normal(3.5, 0.5, n_samples).clip(2, 5),
        'conn_state': np.random.choice([1, 2, 3], n_samples),  # SF, S0, REJ
        'src_bytes': np.random.poisson(25000, n_samples).clip(1000, 100000),
        'dst_bytes': np.random.poisson(25000, n_samples).clip(1000, 100000),
        'wrong_fragment': np.zeros(n_samples),
        'urgent': np.zeros(n_samples),
        'label': np.zeros(n_samples, dtype=int)
    }

def generate_ddos_traffic(n_samples):
    """Generate DDoS attack patterns - high packet rate, small packets, many sources."""
    return {
        'src_port': np.random.randint(1024, 65535, n_samples),
        'dst_port': np.random.choice([80, 443, 53], n_samples),
        'protocol': np.random.choice([6, 17], n_samples, p=[0.3, 0.7]),  # More UDP
        'packet_size': np.random.normal(64, 20, n_samples).clip(40, 200),  # Small packets
        'packet_rate': np.random.exponential(500, n_samples).clip(100, 5000),  # Very high
        'byte_rate': np.random.exponential(50000, n_samples).clip(10000, 500000),
        'flow_duration': np.random.exponential(2, n_samples).clip(0.1, 10),  # Short
        'total_packets': np.random.poisson(1000, n_samples).clip(100, 10000),
        'total_bytes': np.random.poisson(100000, n_samples).clip(10000, 1000000),
        'tcp_flags_syn': np.random.binomial(1, 0.9, n_samples),  # Many SYN floods
        'tcp_flags_ack': np.random.binomial(1, 0.2, n_samples),
        'tcp_flags_fin': np.random.binomial(1, 0.05, n_samples),
        'tcp_flags_rst': np.random.binomial(1, 0.4, n_samples),
        'entropy': np.random.normal(2.0, 0.5, n_samples).clip(0.5, 4),  # Low entropy
        'conn_state': np.random.choice([2, 3], n_samples),  # S0, REJ mostly
        'src_bytes': np.random.poisson(5000, n_samples).clip(100, 20000),
        'dst_bytes': np.random.poisson(500, n_samples).clip(0, 5000),
        'wrong_fragment': np.random.binomial(1, 0.1, n_samples),
        'urgent': np.random.binomial(1, 0.05, n_samples),
        'label': np.ones(n_samples, dtype=int) * 1
    }

def generate_port_scan_traffic(n_samples):
    """Generate port scan patterns - sequential ports, many destinations."""
    return {
        'src_port': np.random.randint(49152, 65535, n_samples),
        'dst_port': np.random.randint(1, 1024, n_samples),  # Scanning well-known ports
        'protocol': np.ones(n_samples, dtype=int) * 6,  # TCP
        'packet_size': np.random.normal(60, 10, n_samples).clip(40, 100),  # Small probes
        'packet_rate': np.random.normal(100, 30, n_samples).clip(20, 300),
        'byte_rate': np.random.normal(6000, 2000, n_samples).clip(1000, 20000),
        'flow_duration': np.random.exponential(0.5, n_samples).clip(0.01, 2),  # Very short
        'total_packets': np.random.poisson(3, n_samples).clip(1, 10),  # Few packets per probe
        'total_bytes': np.random.poisson(200, n_samples).clip(50, 1000),
        'tcp_flags_syn': np.ones(n_samples, dtype=int),  # All SYN probes
        'tcp_flags_ack': np.zeros(n_samples, dtype=int),
        'tcp_flags_fin': np.zeros(n_samples, dtype=int),
        'tcp_flags_rst': np.random.binomial(1, 0.7, n_samples),  # Many RST responses
        'entropy': np.random.normal(4.5, 0.3, n_samples).clip(3, 6),  # High entropy
        'conn_state': np.random.choice([2, 3], n_samples, p=[0.6, 0.4]),
        'src_bytes': np.random.poisson(100, n_samples).clip(40, 500),
        'dst_bytes': np.random.poisson(50, n_samples).clip(0, 200),
        'wrong_fragment': np.zeros(n_samples),
        'urgent': np.zeros(n_samples),
        'label': np.ones(n_samples, dtype=int) * 2
    }

def generate_malware_c2_traffic(n_samples):
    """Generate C2 beacon patterns - periodic connections, specific ports."""
    return {
        'src_port': np.random.randint(49152, 65535, n_samples),
        'dst_port': np.random.choice([8443, 4443, 8080, 1337, 31337], n_samples),  # Common C2 ports
        'protocol': np.ones(n_samples, dtype=int) * 6,  # TCP
        'packet_size': np.random.normal(200, 50, n_samples).clip(100, 500),
        'packet_rate': np.random.normal(2, 1, n_samples).clip(0.5, 10),  # Low, periodic
        'byte_rate': np.random.normal(500, 200, n_samples).clip(100, 2000),
        'flow_duration': np.random.normal(30, 10, n_samples).clip(5, 120),  # Regular intervals
        'total_packets': np.random.poisson(20, n_samples).clip(5, 100),
        'total_bytes': np.random.poisson(5000, n_samples).clip(1000, 50000),
        'tcp_flags_syn': np.random.binomial(1, 0.3, n_samples),
        'tcp_flags_ack': np.ones(n_samples, dtype=int),
        'tcp_flags_fin': np.random.binomial(1, 0.3, n_samples),
        'tcp_flags_rst': np.random.binomial(1, 0.1, n_samples),
        'entropy': np.random.normal(6.5, 0.5, n_samples).clip(5, 8),  # High entropy (encrypted)
        'conn_state': np.ones(n_samples, dtype=int),  # SF - established
        'src_bytes': np.random.poisson(2000, n_samples).clip(500, 10000),
        'dst_bytes': np.random.poisson(3000, n_samples).clip(500, 15000),
        'wrong_fragment': np.zeros(n_samples),
        'urgent': np.zeros(n_samples),
        'label': np.ones(n_samples, dtype=int) * 3
    }

def generate_brute_force_traffic(n_samples):
    """Generate brute force patterns - repeated auth attempts."""
    return {
        'src_port': np.random.randint(49152, 65535, n_samples),
        'dst_port': np.random.choice([22, 3389, 21, 23, 445], n_samples),  # Auth ports
        'protocol': np.ones(n_samples, dtype=int) * 6,  # TCP
        'packet_size': np.random.normal(150, 30, n_samples).clip(80, 300),
        'packet_rate': np.random.normal(50, 20, n_samples).clip(10, 200),
        'byte_rate': np.random.normal(10000, 3000, n_samples).clip(2000, 30000),
        'flow_duration': np.random.exponential(5, n_samples).clip(0.5, 30),
        'total_packets': np.random.poisson(30, n_samples).clip(10, 100),
        'total_bytes': np.random.poisson(5000, n_samples).clip(1000, 30000),
        'tcp_flags_syn': np.random.binomial(1, 0.5, n_samples),
        'tcp_flags_ack': np.random.binomial(1, 0.6, n_samples),
        'tcp_flags_fin': np.random.binomial(1, 0.4, n_samples),
        'tcp_flags_rst': np.random.binomial(1, 0.3, n_samples),  # Connection resets
        'entropy': np.random.normal(3.0, 0.5, n_samples).clip(2, 5),
        'conn_state': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.3, 0.4]),
        'src_bytes': np.random.poisson(2000, n_samples).clip(500, 10000),
        'dst_bytes': np.random.poisson(1000, n_samples).clip(200, 5000),
        'wrong_fragment': np.zeros(n_samples),
        'urgent': np.zeros(n_samples),
        'label': np.ones(n_samples, dtype=int) * 4
    }

def generate_sql_injection_traffic(n_samples):
    """Generate SQL injection patterns - web traffic with large payloads."""
    return {
        'src_port': np.random.randint(49152, 65535, n_samples),
        'dst_port': np.random.choice([80, 443, 8080, 3306, 1433], n_samples),  # Web/DB ports
        'protocol': np.ones(n_samples, dtype=int) * 6,  # TCP
        'packet_size': np.random.normal(800, 200, n_samples).clip(400, 1500),  # Larger payloads
        'packet_rate': np.random.normal(20, 10, n_samples).clip(5, 100),
        'byte_rate': np.random.normal(20000, 5000, n_samples).clip(5000, 50000),
        'flow_duration': np.random.exponential(10, n_samples).clip(1, 60),
        'total_packets': np.random.poisson(50, n_samples).clip(10, 200),
        'total_bytes': np.random.poisson(50000, n_samples).clip(10000, 200000),
        'tcp_flags_syn': np.random.binomial(1, 0.3, n_samples),
        'tcp_flags_ack': np.ones(n_samples, dtype=int),
        'tcp_flags_fin': np.random.binomial(1, 0.4, n_samples),
        'tcp_flags_rst': np.random.binomial(1, 0.1, n_samples),
        'entropy': np.random.normal(5.5, 0.5, n_samples).clip(4, 7),  # SQL has patterns
        'conn_state': np.ones(n_samples, dtype=int),  # Established connections
        'src_bytes': np.random.poisson(30000, n_samples).clip(5000, 100000),
        'dst_bytes': np.random.poisson(20000, n_samples).clip(2000, 80000),
        'wrong_fragment': np.zeros(n_samples),
        'urgent': np.zeros(n_samples),
        'label': np.ones(n_samples, dtype=int) * 5
    }


def main():
    print("=" * 60)
    print("🔧 Creating Multi-Class Attack Dataset")
    print("=" * 60)
    
    # Generate data for each class
    generators = [
        ("Normal", generate_normal_traffic),
        ("DDoS", generate_ddos_traffic),
        ("Port_Scan", generate_port_scan_traffic),
        ("Malware_C2", generate_malware_c2_traffic),
        ("Brute_Force", generate_brute_force_traffic),
        ("SQL_Injection", generate_sql_injection_traffic),
    ]
    
    all_data = []
    for name, generator in generators:
        print(f"  Generating {NUM_SAMPLES_PER_CLASS} samples for {name}...")
        data = generator(NUM_SAMPLES_PER_CLASS)
        df = pd.DataFrame(data)
        all_data.append(df)
    
    # Combine and shuffle
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save training dataset
    output_path = os.path.join(OUTPUT_DIR, "multiclass_attacks.csv")
    combined_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset saved: {output_path}")
    print(f"   Total samples: {len(combined_df)}")
    print(f"   Features: {len(combined_df.columns) - 1}")
    print(f"\n   Class distribution:")
    label_counts = combined_df['label'].value_counts().sort_index()
    class_names = {0: "Normal", 1: "DDoS", 2: "Port_Scan", 3: "Malware_C2", 4: "Brute_Force", 5: "SQL_Injection"}
    for label, count in label_counts.items():
        print(f"     {label} ({class_names[label]}): {count}")
    
    # Also create separate test files for each attack type
    test_dir = os.path.join(OUTPUT_DIR, "attack_test_samples")
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"\n📁 Creating individual attack test files...")
    for name, generator in generators:
        if name == "Normal":
            continue
        test_data = generator(100)  # 100 samples each
        test_df = pd.DataFrame(test_data)
        # Remove label for prediction testing
        test_df_unlabeled = test_df.drop(columns=['label'])
        test_path = os.path.join(test_dir, f"{name.lower()}_test.csv")
        test_df_unlabeled.to_csv(test_path, index=False)
        print(f"     Created: {test_path}")
    
    print("\n" + "=" * 60)
    print("✅ Multi-class dataset creation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train model: python src/train.py --dataset iot23 --epochs 10")
    print("  2. Test predictions on individual attack files")


if __name__ == "__main__":
    main()
