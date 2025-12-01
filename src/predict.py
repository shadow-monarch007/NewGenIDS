"""
Real-time Prediction Module
---------------------------
Predicts attack types from unlabeled network traffic CSV files.
This is what a real IDS does - analyzes traffic patterns without needing labels.
"""
import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import IDSModel, NextGenIDS
from src.data_loader import load_scaler_from_json


# Attack type mapping (0-indexed based on training)
ATTACK_TYPES = {
    0: "Normal",
    1: "DDoS",
    2: "Port_Scan", 
    3: "Malware_C2",
    4: "Brute_Force",
    5: "SQL_Injection"
}


def load_model_and_scaler(checkpoint_path: str, dataset_name: str = "iot23", device: str = "cpu"):
    """Load trained model and scaler."""
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    meta = ckpt.get('meta', {})
    input_dim = meta.get('input_dim', 64)
    num_classes = meta.get('num_classes', 2)
    
    # Detect model architecture
    state_dict_keys = list(ckpt['state_dict'].keys())
    uses_arnn = any('arnn' in key or 'slstm_cnn' in key for key in state_dict_keys)
    
    if uses_arnn:
        model = NextGenIDS(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
    else:
        model = IDSModel(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
    
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    # Load scaler
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    scaler_path = os.path.join(data_dir, f'scaler_{dataset_name}.json')
    scaler = load_scaler_from_json(scaler_path)
    
    return model, scaler, input_dim, num_classes


def preprocess_traffic_data(df: pd.DataFrame, scaler, expected_features: int) -> np.ndarray:
    """
    Preprocess raw traffic CSV for prediction.
    Removes non-feature columns, normalizes, handles missing features.
    """
    # Remove label-related columns if present (for testing with labeled data)
    label_cols = ['label', 'Label', 'attack_type', 'class', 'Class', 'target', 'Target', 'flow_id', 'time_offset']
    df_clean = df.drop(columns=[col for col in label_cols if col in df.columns], errors='ignore')
    
    # Keep only numeric columns
    numeric_df = df_clean.select_dtypes(include=[np.number]).copy()
    
    # Handle missing values
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Convert to numpy
    X = numeric_df.values.astype(np.float32)
    
    # Handle feature dimension mismatch
    if X.shape[1] < expected_features:
        # Pad with zeros
        padding = np.zeros((X.shape[0], expected_features - X.shape[1]), dtype=np.float32)
        X = np.concatenate([X, padding], axis=1)
    elif X.shape[1] > expected_features:
        # Truncate
        X = X[:, :expected_features]
    
    # Normalize using training scaler
    if scaler is not None:
        X = scaler.transform(X)
    
    return X


def create_sequences(X: np.ndarray, seq_len: int = 100) -> np.ndarray:
    """Create sliding window sequences from traffic data."""
    if len(X) < seq_len:
        # Repeat rows to create at least one sequence
        repeats = (seq_len // len(X)) + 1
        X = np.tile(X, (repeats, 1))[:seq_len]
    
    sequences = []
    for i in range(len(X) - seq_len + 1):
        sequences.append(X[i:i + seq_len])
    
    return np.array(sequences, dtype=np.float32)


def predict_traffic(csv_path: str, checkpoint_path: str, dataset_name: str = "iot23", 
                   device: str = "cpu", seq_len: int = 100) -> List[Dict]:
    """
    Predict attack types from unlabeled traffic CSV.
    
    Returns:
        List of predictions with confidence scores and metadata
    """
    # Load model and scaler
    model, scaler, input_dim, num_classes = load_model_and_scaler(checkpoint_path, dataset_name, device)
    
    # Load and preprocess traffic data
    df = pd.read_csv(csv_path)
    X = preprocess_traffic_data(df, scaler, input_dim)
    
    # Create sequences
    X_seq = create_sequences(X, seq_len)
    
    # Predict in batches for performance
    batch_size = 64
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_seq), batch_size):
            batch = X_seq[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).to(device)
            
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_classes = probs.argmax(dim=1).cpu().numpy()
            confidences = probs.max(dim=1).values.cpu().numpy()
            
            for j in range(len(batch)):
                global_idx = i + j
                pred_class = int(pred_classes[j])
                confidence = float(confidences[j])
                
                # Get attack type name
                attack_type = ATTACK_TYPES.get(pred_class, f"Unknown_{pred_class}")
                
                # Extract features from sequence for explanation
                # Use the last row of the sequence as representative
                row_idx = global_idx + seq_len - 1
                if row_idx < len(df):
                    row = df.iloc[row_idx]
                else:
                    row = df.iloc[-1]
                
                features = extract_key_features(row)
                
                predictions.append({
                    "sequence_idx": global_idx,
                    "attack_type": attack_type,
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "severity": get_severity(attack_type, confidence),
                    "features": features,
                    "timestamp": datetime.now().isoformat()
                })
    
    return predictions


def extract_key_features(row: pd.Series) -> Dict[str, float]:
    """Extract key features for explanation from a traffic row."""
    feature_mapping = {
        'packet_rate': ['packet_rate', 'packets_per_sec', 'pkt_rate'],
        'packet_size': ['packet_size', 'avg_pkt_size', 'pkt_size', 'bytes'],
        'byte_rate': ['byte_rate', 'bytes_per_sec', 'bps'],
        'flow_duration': ['flow_duration', 'duration', 'flow_dur'],
        'entropy': ['entropy', 'ent'],
        'src_port': ['src_port', 'sport', 'source_port'],
        'dst_port': ['dst_port', 'dport', 'dest_port', 'destination_port'],
        'total_packets': ['total_packets', 'tot_pkts', 'packets']
    }
    
    features = {}
    for feature_name, possible_cols in feature_mapping.items():
        for col in possible_cols:
            if col in row.index:
                features[feature_name] = float(row[col])
                break
        if feature_name not in features:
            features[feature_name] = 0.0
    
    return features


def get_severity(attack_type: str, confidence: float) -> str:
    """Determine severity level based on attack type and confidence."""
    if attack_type == "Normal":
        return "None"
    
    critical_attacks = ["DDoS", "Malware_C2"]
    high_attacks = ["Brute_Force", "SQL_Injection"]
    
    if attack_type in critical_attacks and confidence > 0.7:
        return "Critical"
    elif attack_type in critical_attacks or (attack_type in high_attacks and confidence > 0.8):
        return "High"
    elif attack_type in high_attacks or confidence > 0.6:
        return "Medium"
    else:
        return "Low"


def save_predictions(predictions: List[Dict], output_path: str):
    """Save predictions to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"✅ Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict attack types from unlabeled traffic CSV")
    parser.add_argument('--input', type=str, required=True, help='Path to traffic CSV file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_iot23.pt', help='Model checkpoint')
    parser.add_argument('--dataset', type=str, default='iot23', help='Dataset name (for scaler)')
    parser.add_argument('--output', type=str, default='results/predictions.json', help='Output JSON path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seq-len', type=int, default=100, help='Sequence length')
    args = parser.parse_args()
    
    print(f"🔍 Analyzing traffic file: {args.input}")
    print(f"📦 Using model: {args.checkpoint}")
    
    predictions = predict_traffic(
        csv_path=args.input,
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        device=args.device,
        seq_len=args.seq_len
    )
    
    # Print summary
    print(f"\n📊 Analysis Results ({len(predictions)} sequences analyzed):")
    attack_counts = {}
    for pred in predictions:
        attack_type = pred['attack_type']
        attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1
    
    for attack_type, count in sorted(attack_counts.items()):
        print(f"  {attack_type}: {count} sequences")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_predictions(predictions, args.output)
    
    print(f"\n✅ Done! Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
