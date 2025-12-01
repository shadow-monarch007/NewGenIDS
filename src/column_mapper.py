"""
Flexible Column Mapper for IDS Datasets
----------------------------------------
Auto-detects and maps column names from various dataset formats to expected format.
Handles missing columns by generating synthetic features or using defaults.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Set


# Standard column mappings (variations of column names found in different datasets)
COLUMN_MAPPINGS = {
    'src_port': ['src_port', 'sport', 'source_port', 'srcport', 'src port', 'source port'],
    'dst_port': ['dst_port', 'dport', 'dest_port', 'destination_port', 'dstport', 'dst port', 'dest port'],
    'protocol': ['protocol', 'proto', 'protocoltype', 'protocol_type'],
    'flow_duration': ['flow_duration', 'duration', 'flow_dur', 'flowduration', 'flow duration'],
    'tot_fwd_pkts': ['tot_fwd_pkts', 'total_fwd_packets', 'fwd_pkts', 'forward_packets', 'tot fwd pkts'],
    'tot_bwd_pkts': ['tot_bwd_pkts', 'total_bwd_packets', 'bwd_pkts', 'backward_packets', 'tot bwd pkts'],
    'totlen_fwd_pkts': ['totlen_fwd_pkts', 'total_length_fwd_packets', 'fwd_pkt_len_tot', 'totlen fwd pkts'],
    'totlen_bwd_pkts': ['totlen_bwd_pkts', 'total_length_bwd_packets', 'bwd_pkt_len_tot', 'totlen bwd pkts'],
    'fwd_pkt_len_max': ['fwd_pkt_len_max', 'fwd_packet_length_max', 'max_fwd_pkt_len', 'fwd pkt len max'],
    'fwd_pkt_len_min': ['fwd_pkt_len_min', 'fwd_packet_length_min', 'min_fwd_pkt_len', 'fwd pkt len min'],
    'fwd_pkt_len_mean': ['fwd_pkt_len_mean', 'fwd_packet_length_mean', 'mean_fwd_pkt_len', 'avg_fwd_pkt_len', 'fwd pkt len mean'],
    'fwd_pkt_len_std': ['fwd_pkt_len_std', 'fwd_packet_length_std', 'std_fwd_pkt_len', 'fwd pkt len std'],
    'bwd_pkt_len_max': ['bwd_pkt_len_max', 'bwd_packet_length_max', 'max_bwd_pkt_len', 'bwd pkt len max'],
    'bwd_pkt_len_min': ['bwd_pkt_len_min', 'bwd_packet_length_min', 'min_bwd_pkt_len', 'bwd pkt len min'],
    'bwd_pkt_len_mean': ['bwd_pkt_len_mean', 'bwd_packet_length_mean', 'mean_bwd_pkt_len', 'avg_bwd_pkt_len', 'bwd pkt len mean'],
    'bwd_pkt_len_std': ['bwd_pkt_len_std', 'bwd_packet_length_std', 'std_bwd_pkt_len', 'bwd pkt len std'],
    'flow_byts_s': ['flow_byts_s', 'flow_bytes_s', 'flow_bytes_per_sec', 'bytes_per_sec', 'flow byts s'],
    'flow_pkts_s': ['flow_pkts_s', 'flow_packets_s', 'flow_packets_per_sec', 'packets_per_sec', 'flow pkts s'],
    'flow_iat_mean': ['flow_iat_mean', 'flow_iat_avg', 'mean_flow_iat', 'avg_flow_iat', 'flow iat mean'],
    'flow_iat_std': ['flow_iat_std', 'std_flow_iat', 'flow iat std'],
    'label': ['label', 'class', 'attack', 'category', 'attack_type', 'type', 'target', 'Label', 'Class', 'Attack']
}


def normalize_column_name(col: str) -> str:
    """Normalize column name by removing spaces, underscores, and converting to lowercase."""
    return col.lower().strip().replace(' ', '').replace('_', '').replace('-', '')


def find_column_match(df_columns: List[str], target_col: str, variations: List[str]) -> str | None:
    """Find the best matching column name from variations."""
    # Normalize all column names for comparison
    normalized_df_cols = {normalize_column_name(col): col for col in df_columns}
    
    # Try exact match first
    if target_col in df_columns:
        return target_col
    
    # Try variations
    for variation in variations:
        if variation in df_columns:
            return variation
        # Try normalized match
        norm_var = normalize_column_name(variation)
        if norm_var in normalized_df_cols:
            return normalized_df_cols[norm_var]
    
    return None


def auto_map_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Automatically map columns from various dataset formats to expected format.
    Handles missing columns by generating synthetic features or using defaults.
    
    Args:
        df: Input DataFrame with any column names
        verbose: Print mapping information
        
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    df_columns = list(df.columns)
    mapped_columns = {}
    missing_columns = []
    
    # Map existing columns
    for standard_name, variations in COLUMN_MAPPINGS.items():
        matched_col = find_column_match(df_columns, standard_name, variations)
        if matched_col:
            mapped_columns[matched_col] = standard_name
            if verbose and matched_col != standard_name:
                print(f"✓ Mapped '{matched_col}' → '{standard_name}'")
        else:
            missing_columns.append(standard_name)
    
    # Rename mapped columns
    df = df.rename(columns=mapped_columns)
    
    # Generate missing columns with reasonable defaults/synthetic data
    if missing_columns and verbose:
        print(f"\n⚠ Missing columns: {missing_columns}")
        print("Generating synthetic features for missing columns...")
    
    for col in missing_columns:
        if col == 'label':
            # Try to infer label from other columns or use default
            if 'attack' in df.columns:
                df['label'] = df['attack']
            elif 'category' in df.columns:
                df['label'] = df['category']
            else:
                # Check if there's any column that looks like a label
                label_candidates = [c for c in df.columns if any(x in c.lower() for x in ['label', 'class', 'attack', 'type', 'category'])]
                if label_candidates:
                    df['label'] = df[label_candidates[0]]
                    if verbose:
                        print(f"  Using '{label_candidates[0]}' as label column")
                else:
                    # Default to 0 (normal traffic)
                    df['label'] = 0
                    if verbose:
                        print("  No label column found, using default (0 = normal)")
        
        elif col in ['src_port', 'dst_port']:
            # Generate random ports in valid range
            df[col] = np.random.randint(1024, 65535, size=len(df))
        
        elif col == 'protocol':
            # Default to TCP (6)
            df[col] = 6
        
        elif col == 'flow_duration':
            # Calculate from other features or use reasonable default
            if 'tot_fwd_pkts' in df.columns and 'tot_bwd_pkts' in df.columns:
                total_pkts = df['tot_fwd_pkts'] + df['tot_bwd_pkts']
                df[col] = total_pkts * np.random.uniform(0.01, 0.1, size=len(df))
            else:
                df[col] = np.random.uniform(0.1, 5.0, size=len(df))
        
        elif 'pkt' in col and 'len' in col:
            # Packet length features - generate based on typical network packet sizes
            if 'max' in col:
                df[col] = np.random.uniform(1000, 1500, size=len(df))
            elif 'min' in col:
                df[col] = np.random.uniform(40, 100, size=len(df))
            elif 'mean' in col or 'avg' in col:
                df[col] = np.random.uniform(200, 800, size=len(df))
            elif 'std' in col:
                df[col] = np.random.uniform(50, 200, size=len(df))
            else:
                df[col] = np.random.uniform(100, 1000, size=len(df))
        
        elif 'tot' in col and 'fwd' in col:
            df[col] = np.random.uniform(1000, 50000, size=len(df))
        
        elif 'tot' in col and 'bwd' in col:
            df[col] = np.random.uniform(500, 30000, size=len(df))
        
        elif 'flow_byts_s' in col:
            # Flow bytes per second
            if 'totlen_fwd_pkts' in df.columns and 'totlen_bwd_pkts' in df.columns and 'flow_duration' in df.columns:
                total_bytes = df['totlen_fwd_pkts'] + df['totlen_bwd_pkts']
                df[col] = total_bytes / (df['flow_duration'] + 0.001)  # Avoid division by zero
            else:
                df[col] = np.random.uniform(1000, 100000, size=len(df))
        
        elif 'flow_pkts_s' in col:
            # Flow packets per second
            if 'tot_fwd_pkts' in df.columns and 'tot_bwd_pkts' in df.columns and 'flow_duration' in df.columns:
                total_pkts = df['tot_fwd_pkts'] + df['tot_bwd_pkts']
                df[col] = total_pkts / (df['flow_duration'] + 0.001)
            else:
                df[col] = np.random.uniform(10, 1000, size=len(df))
        
        elif 'iat' in col:
            # Inter-arrival time features
            if 'mean' in col:
                df[col] = np.random.uniform(0.001, 1.0, size=len(df))
            elif 'std' in col:
                df[col] = np.random.uniform(0.0001, 0.5, size=len(df))
            else:
                df[col] = np.random.uniform(0.001, 1.0, size=len(df))
        
        else:
            # Default: small random values
            df[col] = np.random.uniform(0, 100, size=len(df))
    
    if verbose and missing_columns:
        print("✓ Synthetic features generated\n")
    
    return df


def ensure_required_columns(df: pd.DataFrame, required_columns: List[str] = None) -> pd.DataFrame:
    """
    Ensure DataFrame has all required columns, auto-mapping and generating as needed.
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names (if None, uses all standard columns)
        
    Returns:
        DataFrame with all required columns
    """
    if required_columns is None:
        required_columns = list(COLUMN_MAPPINGS.keys())
    
    # First, auto-map existing columns
    df = auto_map_columns(df, verbose=True)
    
    # Ensure all required columns exist
    for col in required_columns:
        if col not in df.columns:
            # This should rarely happen after auto_map_columns, but just in case
            df[col] = 0
    
    return df


def validate_and_prepare_csv(csv_path: str, output_path: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Load, validate, and prepare a CSV file for training/evaluation.
    Auto-maps columns and generates missing features.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Optional path to save the prepared CSV
        verbose: Print progress information
        
    Returns:
        Prepared DataFrame
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {csv_path}")
        print(f"{'='*60}\n")
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        if verbose:
            print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")
    
    # Auto-map and generate features
    df = auto_map_columns(df, verbose=verbose)
    
    # Save prepared CSV if requested
    if output_path:
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\n✓ Saved prepared CSV to: {output_path}")
    
    return df


__all__ = ['auto_map_columns', 'ensure_required_columns', 'validate_and_prepare_csv', 'COLUMN_MAPPINGS']
