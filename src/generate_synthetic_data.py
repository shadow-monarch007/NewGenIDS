"""
Generate Synthetic Data
-----------------------
Creates small synthetic CSV files for testing the IDS pipeline without real datasets.
Generates random features and binary/multiclass labels.

CLI:
    python src/generate_synthetic_data.py --dataset iot23 --num-samples 5000 --num-features 20 --num-classes 2
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic IDS test data")
    p.add_argument("--dataset", type=str, default="iot23", help="Dataset name (creates data/<dataset>/synthetic.csv)")
    p.add_argument("--num-samples", type=int, default=5000, help="Number of rows to generate")
    p.add_argument("--num-features", type=int, default=20, help="Number of feature columns")
    p.add_argument("--num-classes", type=int, default=2, help="Number of label classes (binary=2, multiclass>2)")
    p.add_argument("--output-dir", type=str, default=None, help="Override output directory (default: data/<dataset>)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        output_dir = os.path.join(base_data_dir, args.dataset)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate random features (mean=0, std=1 with some noise patterns)
    X = np.random.randn(args.num_samples, args.num_features).astype(np.float32)
    
    # Add some structure: make features 0-2 correlated with labels for the model to learn
    labels = np.random.randint(0, args.num_classes, size=args.num_samples)
    for i in range(min(3, args.num_features)):
        # Shift features based on class to create learnable patterns
        for cls in range(args.num_classes):
            mask = labels == cls
            X[mask, i] += cls * 2.0 + np.random.randn() * 0.5
    
    # Build DataFrame
    feature_names = [f"feature_{i}" for i in range(args.num_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = labels
    
    # Save to CSV
    output_path = os.path.join(output_dir, "synthetic.csv")
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {args.num_samples} samples × {args.num_features} features → {args.num_classes} classes")
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"✓ Feature range: [{X.min():.2f}, {X.max():.2f}]")


if __name__ == "__main__":
    main()
