"""
Create Unlabeled Demo Files
---------------------------
Creates CSV files WITHOUT label columns for client demonstration.
This proves the system works on truly unlabeled data!
"""
import os
import pandas as pd

# Read original demo samples
data_dir = "data/iot23/demo_samples"
output_dir = "data/iot23/unlabeled_samples"
os.makedirs(output_dir, exist_ok=True)

# List of demo files
demo_files = [
    "normal.csv",
    "ddos.csv",
    "port_scan.csv",
    "malware_c2.csv",
    "brute_force.csv",
    "sql_injection.csv"
]

for filename in demo_files:
    input_path = os.path.join(data_dir, filename)
    if not os.path.exists(input_path):
        print(f"⚠️  Skipping {filename} (not found)")
        continue
    
    # Read the file
    df = pd.read_csv(input_path)
    
    # Remove label and attack_type columns
    cols_to_remove = ['label', 'Label', 'attack_type', 'class', 'Class']
    for col in cols_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Save unlabeled version
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"✅ Created unlabeled {filename} ({len(df)} rows, {len(df.columns)} features)")

print(f"\n🎉 Done! Unlabeled demo files saved to {output_dir}/")
print("   These files can be uploaded to the dashboard for threat detection.")
