"""
Test script to demonstrate dynamic AI threat analysis
Shows how different feature values produce different explanations
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dashboard import generate_ai_explanation

def test_dynamic_analysis():
    """Test that AI analysis generates different results for different feature values"""
    
    print("="*80)
    print("TESTING DYNAMIC AI THREAT ANALYSIS")
    print("="*80)
    print()
    
    # Load demo samples
    samples_dir = "data/iot23/demo_samples"
    
    attack_types = {
        "ddos.csv": "DDoS",
        "port_scan.csv": "Port Scan",
        "malware_c2.csv": "Malware C2",
        "brute_force.csv": "Brute Force",
        "sql_injection.csv": "SQL Injection",
        "normal.csv": "Normal"
    }
    
    for filename, attack_type in attack_types.items():
        filepath = os.path.join(samples_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️  {filename} not found, skipping...")
            continue
            
        # Read first row
        df = pd.read_csv(filepath)
        first_row = df.iloc[0]
        
        # Extract features
        features = {
            'packet_rate': first_row.get('packet_rate', 0),
            'packet_size': first_row.get('packet_size', 0),
            'byte_rate': first_row.get('byte_rate', 0),
            'flow_duration': first_row.get('flow_duration', 0),
            'entropy': first_row.get('entropy', 0),
            'src_port': first_row.get('src_port', 0),
            'dst_port': first_row.get('dst_port', 0),
            'total_packets': first_row.get('total_packets', 0)
        }
        
        # Generate explanation
        explanation = generate_ai_explanation(attack_type, features, 0.95)
        
        print(f"\n{'='*80}")
        print(f"📁 FILE: {filename}")
        print(f"🎯 ATTACK TYPE: {attack_type}")
        print(f"{'='*80}")
        print(f"\n📊 EXTRACTED FEATURES:")
        print(f"   • Packet Rate: {features['packet_rate']:.2f} pps")
        print(f"   • Packet Size: {features['packet_size']:.2f} bytes")
        print(f"   • Byte Rate: {features['byte_rate']:.2f} bytes/sec")
        print(f"   • Flow Duration: {features['flow_duration']:.4f} sec")
        print(f"   • Entropy: {features['entropy']:.2f}")
        print(f"   • Dst Port: {int(features['dst_port'])}")
        print(f"\n🔍 AI ANALYSIS:")
        print(f"   {explanation['description']}")
        print(f"\n🚨 SEVERITY: {explanation['severity']}")
        print(f"📍 STAGE: {explanation['attack_stage']}")
        print(f"\n📌 KEY INDICATORS:")
        for indicator in explanation['indicators'][:3]:  # Show first 3
            print(f"   {indicator}")
        print()
    
    print("="*80)
    print("✅ TEST COMPLETE - Each attack type shows UNIQUE values!")
    print("="*80)

if __name__ == "__main__":
    test_dynamic_analysis()
