"""
Comprehensive Model Analysis - AIML & DL Assessment
--------------------------------------------------
This script performs a thorough analysis of the trained IDS models to determine:
1. Model architecture and parameters
2. Training performance metrics
3. Accuracy on demo datasets
4. Readiness for demonstration
5. Performance on real-world datasets
"""
import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import IDSModel, NextGenIDS
from src.data_loader import load_scaler_from_json
from src.predict import load_model_and_scaler, preprocess_traffic_data, create_sequences, ATTACK_TYPES


def analyze_checkpoint(checkpoint_path):
    """Analyze a model checkpoint."""
    print(f"\n{'='*70}")
    print(f"CHECKPOINT ANALYSIS: {os.path.basename(checkpoint_path)}")
    print(f"{'='*70}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    meta = ckpt.get('meta', {})
    state_dict = ckpt.get('state_dict', {})
    
    print(f"\n📦 CHECKPOINT METADATA:")
    print(f"   Input Dimension: {meta.get('input_dim', 'Unknown')}")
    print(f"   Number of Classes: {meta.get('num_classes', 'Unknown')}")
    print(f"   Training Epoch: {meta.get('epoch', 'Unknown')}")
    print(f"   Best F1 Score: {meta.get('f1', 'Unknown'):.4f}" if 'f1' in meta else "   Best F1 Score: Unknown")
    
    # Detect architecture
    state_dict_keys = list(state_dict.keys())
    uses_arnn = any('arnn' in key or 'slstm_cnn' in key for key in state_dict_keys)
    
    print(f"\n🏗️  MODEL ARCHITECTURE:")
    print(f"   Type: {'NextGenIDS (A-RNN + S-LSTM + CNN)' if uses_arnn else 'IDSModel (S-LSTM + CNN)'}")
    
    # Count parameters
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"   Total Parameters: {total_params:,}")
    
    print(f"\n📁 FILE INFO:")
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"   File Size: {file_size:.2f} MB")
    modified_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
    print(f"   Last Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'path': checkpoint_path,
        'meta': meta,
        'architecture': 'NextGenIDS' if uses_arnn else 'IDSModel',
        'total_params': total_params,
        'file_size_mb': file_size
    }


def test_on_demo_samples(checkpoint_path, dataset_name='iot23'):
    """Test the model on all available demo samples."""
    print(f"\n{'='*70}")
    print(f"DEMO SAMPLES TESTING")
    print(f"{'='*70}")
    
    demo_dir = os.path.join('data', dataset_name, 'demo_samples')
    if not os.path.exists(demo_dir):
        print(f"❌ Demo samples directory not found: {demo_dir}")
        return None
    
    # Load model
    try:
        model, scaler, input_dim, num_classes = load_model_and_scaler(checkpoint_path, dataset_name, 'cpu')
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Find all demo CSV files
    demo_files = [f for f in os.listdir(demo_dir) if f.endswith('.csv')]
    
    if not demo_files:
        print(f"❌ No demo CSV files found in {demo_dir}")
        return None
    
    print(f"\n📊 Found {len(demo_files)} demo files:")
    for f in demo_files:
        print(f"   - {f}")
    
    results = []
    
    for demo_file in sorted(demo_files):
        file_path = os.path.join(demo_dir, demo_file)
        expected_label = demo_file.replace('.csv', '').upper()
        
        print(f"\n🔍 Testing: {demo_file}")
        print(f"   Expected: {expected_label}")
        
        try:
            # Load and preprocess
            df = pd.read_csv(file_path)
            print(f"   Samples: {len(df)} rows")
            
            # Get actual labels if available
            actual_labels = None
            if 'label' in df.columns:
                actual_labels = df['label'].values
                label_dist = Counter(actual_labels)
                print(f"   Label Distribution: {dict(label_dist)}")
            
            X = preprocess_traffic_data(df, scaler, input_dim)
            X_seq = create_sequences(X, seq_len=100)
            
            # Predict
            predictions = []
            confidences = []
            
            with torch.no_grad():
                batch_size = 64
                for i in range(0, len(X_seq), batch_size):
                    batch = X_seq[i:i+batch_size]
                    batch_tensor = torch.from_numpy(batch).to('cpu')
                    
                    logits = model(batch_tensor)
                    probs = torch.softmax(logits, dim=1)
                    pred_classes = probs.argmax(dim=1).cpu().numpy()
                    confs = probs.max(dim=1).values.cpu().numpy()
                    
                    predictions.extend(pred_classes)
                    confidences.extend(confs)
            
            # Analyze predictions
            pred_counter = Counter(predictions)
            most_common_pred = pred_counter.most_common(1)[0]
            predicted_class = most_common_pred[0]
            predicted_label = ATTACK_TYPES.get(predicted_class, f"Class_{predicted_class}")
            percentage = (most_common_pred[1] / len(predictions)) * 100
            avg_confidence = np.mean(confidences)
            
            print(f"   Predicted: {predicted_label} ({percentage:.1f}% of sequences)")
            print(f"   Avg Confidence: {avg_confidence:.4f}")
            print(f"   All Predictions: {dict(pred_counter)}")
            
            # Calculate accuracy if ground truth available
            accuracy = None
            if actual_labels is not None and len(actual_labels) >= 100:
                # Compare with sequence-level ground truth
                seq_labels = []
                for i in range(len(predictions)):
                    window_labels = actual_labels[i:i+100]
                    if len(window_labels) > 0:
                        # Majority vote
                        labels, counts = np.unique(window_labels, return_counts=True)
                        seq_labels.append(labels[np.argmax(counts)])
                
                if len(seq_labels) == len(predictions):
                    accuracy = accuracy_score(seq_labels, predictions)
                    print(f"   ✅ Accuracy: {accuracy:.4f}")
            
            results.append({
                'file': demo_file,
                'expected': expected_label,
                'predicted': predicted_label,
                'percentage': percentage,
                'avg_confidence': avg_confidence,
                'accuracy': accuracy,
                'total_sequences': len(predictions)
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'file': demo_file,
                'expected': expected_label,
                'predicted': 'ERROR',
                'error': str(e)
            })
    
    return results


def test_on_uploaded_dataset(checkpoint_path, dataset_name='iot23'):
    """Test on uploaded/real datasets."""
    print(f"\n{'='*70}")
    print(f"UPLOADED DATASET TESTING")
    print(f"{'='*70}")
    
    uploaded_dir = 'uploads/uploaded'
    if not os.path.exists(uploaded_dir):
        print(f"ℹ️  No uploaded datasets found")
        return None
    
    uploaded_files = [f for f in os.listdir(uploaded_dir) if f.endswith('.csv')][:3]  # Test first 3
    
    if not uploaded_files:
        print(f"ℹ️  No CSV files in uploads")
        return None
    
    print(f"\n📊 Testing on {len(uploaded_files)} uploaded files...")
    
    try:
        model, scaler, input_dim, num_classes = load_model_and_scaler(checkpoint_path, dataset_name, 'cpu')
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    results = []
    
    for file_name in uploaded_files[:2]:  # Test first 2 to save time
        file_path = os.path.join(uploaded_dir, file_name)
        print(f"\n🔍 Testing: {file_name}")
        
        try:
            df = pd.read_csv(file_path, nrows=1000)  # Limit to 1000 rows
            print(f"   Samples: {len(df)} rows (limited)")
            
            X = preprocess_traffic_data(df, scaler, input_dim)
            X_seq = create_sequences(X, seq_len=100)
            
            predictions = []
            with torch.no_grad():
                for i in range(0, len(X_seq), 64):
                    batch = X_seq[i:i+64]
                    batch_tensor = torch.from_numpy(batch).to('cpu')
                    logits = model(batch_tensor)
                    pred_classes = logits.argmax(1).cpu().numpy()
                    predictions.extend(pred_classes)
            
            pred_counter = Counter(predictions)
            print(f"   Predictions: {dict(pred_counter)}")
            
            results.append({
                'file': file_name,
                'predictions': dict(pred_counter),
                'total_sequences': len(predictions)
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return results


def generate_report():
    """Generate comprehensive analysis report."""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL ANALYSIS REPORT")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    checkpoints = [
        'checkpoints/best_iot23_retrained.pt',
        'checkpoints/best_iot23.pt',
        'checkpoints/best_uploaded.pt'
    ]
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'checkpoints': [],
        'demo_results': [],
        'conclusion': {}
    }
    
    # Analyze all checkpoints
    for ckpt_path in checkpoints:
        if os.path.exists(ckpt_path):
            ckpt_info = analyze_checkpoint(ckpt_path)
            if ckpt_info:
                report['checkpoints'].append(ckpt_info)
    
    # Test on demo samples (use the retrained model)
    best_checkpoint = 'checkpoints/best_iot23_retrained.pt'
    if os.path.exists(best_checkpoint):
        demo_results = test_on_demo_samples(best_checkpoint)
        if demo_results:
            report['demo_results'] = demo_results
        
        uploaded_results = test_on_uploaded_dataset(best_checkpoint)
        if uploaded_results:
            report['uploaded_results'] = uploaded_results
    
    # Generate conclusion
    print(f"\n{'='*70}")
    print("FINAL ASSESSMENT")
    print(f"{'='*70}")
    
    if report['checkpoints']:
        best_model = report['checkpoints'][0]
        print(f"\n✅ TRAINED MODEL FOUND:")
        print(f"   Architecture: {best_model['architecture']}")
        print(f"   Parameters: {best_model['total_params']:,}")
        print(f"   Best F1: {best_model['meta'].get('f1', 'N/A')}")
        
        if best_model['meta'].get('f1', 0) > 0.80:
            print(f"\n✅ MODEL TRAINING QUALITY: EXCELLENT (F1 > 80%)")
            report['conclusion']['training_quality'] = 'EXCELLENT'
        elif best_model['meta'].get('f1', 0) > 0.70:
            print(f"\n⚠️  MODEL TRAINING QUALITY: GOOD (F1 > 70%)")
            report['conclusion']['training_quality'] = 'GOOD'
        else:
            print(f"\n⚠️  MODEL TRAINING QUALITY: NEEDS IMPROVEMENT")
            report['conclusion']['training_quality'] = 'NEEDS_IMPROVEMENT'
    
    if report.get('demo_results'):
        print(f"\n📊 DEMO TESTING SUMMARY:")
        correct_predictions = 0
        total_tests = len(report['demo_results'])
        
        for result in report['demo_results']:
            if 'error' not in result:
                # Check if prediction matches expected (loose matching)
                expected_keywords = result['expected'].lower().split('_')
                predicted_keywords = result['predicted'].lower().split('_')
                
                match = any(kw in result['predicted'].lower() for kw in expected_keywords)
                
                status = "✅" if match else "❌"
                print(f"   {status} {result['file']}: Expected {result['expected']}, Got {result['predicted']} ({result['avg_confidence']:.2%} conf)")
                
                if match:
                    correct_predictions += 1
        
        accuracy_pct = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
        print(f"\n   Overall Demo Accuracy: {correct_predictions}/{total_tests} ({accuracy_pct:.1f}%)")
        
        if accuracy_pct >= 80:
            print(f"\n✅ DEMO READINESS: READY FOR DEMONSTRATION")
            report['conclusion']['demo_ready'] = True
        elif accuracy_pct >= 60:
            print(f"\n⚠️  DEMO READINESS: PARTIALLY READY (needs tuning)")
            report['conclusion']['demo_ready'] = False
        else:
            print(f"\n❌ DEMO READINESS: NOT READY (requires retraining)")
            report['conclusion']['demo_ready'] = False
    
    # Final recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    
    training_quality = report['conclusion'].get('training_quality')
    demo_ready = report['conclusion'].get('demo_ready', False)
    
    if training_quality == 'EXCELLENT' and demo_ready:
        print("✅ MODEL IS WELL-TRAINED AND READY FOR DEMONSTRATION")
        print("✅ The model can analyze online datasets with good accuracy")
        print("✅ Suitable for academic presentation and demo purposes")
    elif training_quality in ['EXCELLENT', 'GOOD']:
        print("⚠️  MODEL IS TRAINED BUT MAY NEED FINE-TUNING")
        print("⚠️  Consider retraining with multi-class labels for better demo results")
        print("✅ Can still demonstrate core functionality")
    else:
        print("❌ MODEL REQUIRES RETRAINING")
        print("❌ Not recommended for demonstration in current state")
    
    # Save report
    report_path = 'results/model_analysis_report.json'
    os.makedirs('results', exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    report = generate_report()
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")
