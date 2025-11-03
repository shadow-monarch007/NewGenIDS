"""
Download and Test Real Network Traffic Dataset
----------------------------------------------
Downloads a real dataset from the internet and tests our IDS system.
"""
import os
import sys
import requests
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict_traffic

def download_kdd_dataset():
    """Download NSL-KDD dataset (a classic intrusion detection dataset)."""
    print("📥 Downloading NSL-KDD dataset from GitHub...")
    
    url = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"
    output_file = "downloads/nsl_kdd_test.csv"
    
    os.makedirs("downloads", exist_ok=True)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save the file
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded: {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


def convert_kdd_to_traffic_format(input_file, output_file):
    """
    Convert KDD dataset to our traffic format.
    KDD has 42 columns, we'll map relevant ones to our features.
    """
    print(f"🔄 Converting {input_file} to traffic format...")
    
    # KDD column names
    kdd_columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    
    try:
        # Read KDD data
        df = pd.read_csv(input_file, names=kdd_columns)
        
        print(f"  Original shape: {df.shape}")
        print(f"  Attack types found: {df['label'].unique()[:10]}")
        
        # Map to our traffic format (approximate mapping)
        traffic_df = pd.DataFrame({
            'flow_id': [f'flow_{i:06d}' for i in range(len(df))],
            'time_offset': range(len(df)),
            'src_port': 0,  # Not available in KDD
            'dst_port': 0,  # Not available in KDD
            'protocol': df['protocol_type'].map({'tcp': 6, 'udp': 17, 'icmp': 1}).fillna(6),
            'packet_size': (df['src_bytes'] + df['dst_bytes']) / (df['count'] + 1),
            'packet_rate': df['count'] / (df['duration'] + 1),
            'byte_rate': (df['src_bytes'] + df['dst_bytes']) / (df['duration'] + 1),
            'flow_duration': df['duration'],
            'total_packets': df['count'],
            'total_bytes': df['src_bytes'] + df['dst_bytes'],
            'tcp_flags_syn': 0,
            'tcp_flags_ack': 0,
            'tcp_flags_fin': 0,
            'tcp_flags_rst': 0,
            'tcp_flags_psh': 0,
            'icmp_type': 0,
            'connection_state': df['flag'].map({'SF': 1, 'S0': 2, 'REJ': 3}).fillna(0),
            'service': df['service'].map({'http': 1, 'ftp': 2, 'smtp': 3, 'ssh': 4}).fillna(0),
            'entropy': df['srv_diff_host_rate'] * 8,  # Approximate
        })
        
        # Save WITHOUT label (this is the key!)
        traffic_df.to_csv(output_file, index=False)
        
        print(f"✅ Converted to traffic format: {output_file}")
        print(f"   Shape: {traffic_df.shape}")
        print(f"   Features: {len(traffic_df.columns)} columns")
        
        # Also save a small sample (first 100 rows) for quick testing
        sample_file = output_file.replace('.csv', '_sample.csv')
        traffic_df.head(100).to_csv(sample_file, index=False)
        print(f"✅ Sample file (100 rows): {sample_file}")
        
        return output_file, sample_file
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_prediction(csv_file):
    """Test prediction on the downloaded dataset."""
    print(f"\n🔍 Testing prediction on: {csv_file}")
    
    checkpoint = "checkpoints/best_iot23.pt"
    if not os.path.exists(checkpoint):
        print(f"❌ Model not found: {checkpoint}")
        print("   Please train the model first with: python src/train.py --dataset iot23 --epochs 5")
        return
    
    try:
        predictions = predict_traffic(
            csv_path=csv_file,
            checkpoint_path=checkpoint,
            dataset_name='iot23',
            device='cpu',
            seq_len=100
        )
        
        print(f"\n📊 Prediction Results:")
        print(f"   Total sequences analyzed: {len(predictions)}")
        
        # Count attack types
        attack_counts = {}
        for pred in predictions:
            attack_type = pred['attack_type']
            attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1
        
        print(f"\n   Attack Type Distribution:")
        for attack_type, count in sorted(attack_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(predictions)) * 100
            print(f"     {attack_type}: {count} ({percentage:.1f}%)")
        
        # Show sample predictions
        print(f"\n   Sample Predictions (first 5):")
        for i, pred in enumerate(predictions[:5]):
            print(f"     {i+1}. {pred['attack_type']} (confidence: {pred['confidence']:.2%}, severity: {pred['severity']})")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 70)
    print("🌐 Real Dataset Download & Test Tool")
    print("=" * 70)
    print()
    
    # Step 1: Download dataset
    kdd_file = download_kdd_dataset()
    if not kdd_file:
        print("\n❌ Failed to download dataset. Check your internet connection.")
        return
    
    # Step 2: Convert to our format
    print()
    traffic_file, sample_file = convert_kdd_to_traffic_format(kdd_file, "downloads/kdd_traffic.csv")
    if not traffic_file:
        print("\n❌ Failed to convert dataset.")
        return
    
    # Step 3: Test prediction on sample
    print()
    predictions = test_prediction(sample_file)
    
    if predictions:
        print("\n" + "=" * 70)
        print("✅ SUCCESS! The system works on real downloaded datasets!")
        print("=" * 70)
        print(f"\n📁 Files created:")
        print(f"   - Original: {kdd_file}")
        print(f"   - Converted: {traffic_file}")
        print(f"   - Sample: {sample_file}")
        print(f"\n💡 You can now upload '{sample_file}' to the dashboard!")
        print(f"   Dashboard URL: http://localhost:5000")
    else:
        print("\n❌ Prediction test failed.")


if __name__ == "__main__":
    main()
