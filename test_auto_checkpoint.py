"""
Test automatic checkpoint selection based on feature dimensions
"""
import os
import torch

CHECKPOINT_DIR = 'checkpoints'

def select_best_checkpoint(num_features: int) -> tuple:
    """
    Automatically select the best matching checkpoint based on number of features.
    Returns (checkpoint_path, dataset_name) tuple.
    """
    checkpoints = []
    
    # Scan available checkpoints and get their metadata
    for ckpt_file in os.listdir(CHECKPOINT_DIR):
        if ckpt_file.endswith('.pt'):
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                meta = ckpt.get('meta', {})
                input_dim = meta.get('input_dim', 0)
                num_classes = meta.get('num_classes', 2)
                f1 = meta.get('f1', 0.0)
                
                # Extract dataset name from filename (e.g., best_iot23.pt -> iot23)
                dataset_name = ckpt_file.replace('best_', '').replace('.pt', '')
                
                checkpoints.append({
                    'file': ckpt_file,
                    'path': ckpt_path,
                    'dataset': dataset_name,
                    'input_dim': input_dim,
                    'num_classes': num_classes,
                    'f1': f1,
                    'dim_diff': abs(input_dim - num_features)
                })
            except Exception as e:
                print(f"Could not load checkpoint {ckpt_file}: {e}")
                continue
    
    if not checkpoints:
        raise ValueError("No valid checkpoints found")
    
    # Sort by: 1) smallest dimension difference, 2) highest F1 score
    checkpoints.sort(key=lambda x: (x['dim_diff'], -x['f1']))
    
    best = checkpoints[0]
    print(f"🎯 Auto-selected checkpoint: {best['file']} (input_dim={best['input_dim']}, "
          f"data_features={num_features}, f1={best['f1']:.4f})")
    
    return best['path'], best['dataset']


if __name__ == '__main__':
    print("=" * 80)
    print("Testing Automatic Checkpoint Selection")
    print("=" * 80)
    
    # Print all available checkpoints
    print("\n📦 Available Checkpoints:")
    for ckpt_file in sorted(os.listdir(CHECKPOINT_DIR)):
        if ckpt_file.endswith('.pt'):
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                meta = ckpt.get('meta', {})
                print(f"  • {ckpt_file:30s} → input_dim={meta.get('input_dim', '?'):3d}, "
                      f"num_classes={meta.get('num_classes', '?'):2d}, f1={meta.get('f1', 0):.4f}")
            except Exception as e:
                print(f"  • {ckpt_file:30s} → ERROR: {e}")
    
    # Test various feature counts
    test_cases = [
        (20, "Small dataset (20 features)"),
        (39, "IoT23 dataset (39 features)"),
        (60, "Multiclass dataset (60 features)"),
        (74, "Uploaded dataset (74 features)"),
        (79, "Demo attacks CSV (79 features)")
    ]
    
    print("\n" + "=" * 80)
    print("Auto-Selection Test Cases:")
    print("=" * 80)
    
    for num_features, description in test_cases:
        print(f"\n🔍 Test: {description}")
        try:
            ckpt_path, dataset_name = select_best_checkpoint(num_features)
            print(f"   ✅ Selected: {os.path.basename(ckpt_path)} (dataset={dataset_name})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Auto-selection test complete!")
    print("=" * 80)
