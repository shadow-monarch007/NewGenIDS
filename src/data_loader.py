"""
Data Loader Module
------------------
Loads and preprocesses CSV data for intrusion detection.
- Supports IoT-23 and BETH datasets (CSV files inside data/<dataset>/).
- Train/val/test split
- Normalization (StandardScaler)
- Returns PyTorch Dataset and DataLoader utilities

Usage:
    from src.data_loader import create_dataloaders
    train_loader, val_loader, test_loader, input_dim = create_dataloaders(
        dataset_name="iot23", batch_size=64, seq_len=100
    )
"""
from __future__ import annotations

import os
import glob
import json
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DataConfig:
    dataset_name: str  # 'iot23' or 'beth'
    data_dir: str
    batch_size: int = 64
    seq_len: int = 100
    num_workers: int = 0
    test_size: float = 0.2
    val_size: float = 0.1
    shuffle: bool = True
    target_column: str = "label"
    cache_scaler_path: Optional[str] = None


class TimeSeriesCSVDataset(Dataset):
    """Converts tabular rows into sliding time windows for sequence models.

    Expects a fully numeric feature matrix X and a binary/ multiclass label vector y.
    The dataset creates sequences of length seq_len via a simple sliding window.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        assert len(X) == len(y), "Features and labels must have the same number of rows"
        self.X = X.astype(np.float32)
        self.y = y
        self.seq_len = seq_len

        if len(self.X) < self.seq_len:
            raise ValueError("Not enough rows to create one sequence. Reduce seq_len or add data.")

    def __len__(self) -> int:
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx: int):
        x_seq = self.X[idx : idx + self.seq_len]  # (seq_len, feat)
        # Majority label in the window; you can change to last element label
        window_labels = self.y[idx : idx + self.seq_len]
        labels, counts = np.unique(window_labels, return_counts=True)
        label = labels[np.argmax(counts)]
        return torch.from_numpy(x_seq), torch.tensor(label, dtype=torch.long)


def _find_csv_files(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "*.csv")))


def _load_dataset_frames(dataset_name: str, base_data_dir: str) -> pd.DataFrame:
    dataset_dir = os.path.join(base_data_dir, dataset_name)
    files = _find_csv_files(dataset_dir)
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {dataset_dir}. Place CSVs in data/{dataset_name}/"
        )
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
    if not dfs:
        raise RuntimeError("Failed to read any CSV files.")
    return pd.concat(dfs, ignore_index=True)


def _select_features(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        # Heuristic: create a dummy label if missing (all zeros)
        # NOTE: creating dummy labels will lead to a single-class dataset and
        # therefore degenerate supervised training. We keep behavior but warn
        # loudly so users can fix their input files.
        print(f"Warning: Target column '{target_column}' not found. Creating dummy labels (all zeros). This will create a single-class dataset - supervised training will not be meaningful.")
        df[target_column] = 0

    # CRITICAL: Remove label column and attack_type column BEFORE selecting numeric features
    # to prevent label leakage (model must learn from traffic patterns, not labels!)
    label_related_cols = [target_column, 'attack_type', 'label', 'Label', 'class', 'Class', 'target', 'Target']
    df_features = df.drop(columns=[col for col in label_related_cols if col in df.columns], errors='ignore')
    
    # Keep only numeric columns as features
    numeric_df = df_features.select_dtypes(include=[np.number]).copy()

    # Robust handling for label column: support numeric labels and categorical/text labels
    y_raw = df[target_column]
    # If labels are non-numeric (object / string), factorize them into integer codes
    if y_raw.dtype == object or y_raw.dtype.name.startswith("category"):
        # pd.factorize returns (codes, uniques)
        codes, uniques = pd.factorize(y_raw)
        y = pd.Series(codes, index=df.index)
    else:
        # Try to convert numeric-like labels to int, otherwise fallback to factorize
        try:
            y = y_raw.astype(int)
        except Exception:
            codes, uniques = pd.factorize(y_raw)
            y = pd.Series(codes, index=df.index)

    # Fill NaNs and Infs
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return numeric_df, y


def create_dataloaders(
    dataset_name: str,
    batch_size: int = 64,
    seq_len: int = 100,
    data_dir: Optional[str] = None,
    num_workers: int = 0,
    test_size: float = 0.2,
    val_size: float = 0.1,
    target_column: str = "label",
    scaler_cache_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Loads dataset, splits, scales features, builds sequence datasets, and returns loaders.

    Returns:
        train_loader, val_loader, test_loader, input_dim, num_classes
    """
    base_data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    df = _load_dataset_frames(dataset_name, base_data_dir)
    X_df, y = _select_features(df, target_column)

    # Encode labels to 0..C-1 (ensure integer-coded)
    classes, y_idx = np.unique(y, return_inverse=True)
    num_classes = len(classes)

    # Sanity check: supervised training requires at least 2 classes
    if num_classes < 2:
        raise ValueError(
            f"Only {num_classes} class(es) found in target column '{target_column}'.\n"
            "Supervised training requires >= 2 classes.\n"
            "Check that your CSV files include a proper label column (e.g., 'label' or pass a different target_column)."
        )

    # Quick leakage detection: warn if any numeric feature is identical or nearly perfectly
    # correlated with the label (common sign of label leakage / label in features).
    try:
        numeric_cols = list(X_df.columns)
        for col in numeric_cols:
            col_vals = X_df[col].values
            # identical check
            if np.array_equal(col_vals, y):
                print(f"Warning: Feature column '{col}' is identical to target '{target_column}' (possible leakage).")
            else:
                # compute Pearson correlation where meaningful
                try:
                    if np.std(col_vals) > 0:
                        corr = np.corrcoef(col_vals.astype(float), y.astype(float))[0, 1]
                        if not np.isnan(corr) and abs(corr) > 0.95:
                            print(f"Warning: Feature column '{col}' has very high correlation ({corr:.3f}) with target '{target_column}' (possible leakage).")
                except Exception:
                    # ignore columns that can't be cast to float
                    pass
    except Exception as e:
        print(f"Leakage detection skipped due to error: {e}")

    # Train/test split first
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y_idx, test_size=test_size, random_state=42, stratify=y_idx if num_classes > 1 else None
    )

    # Then train/val split from train
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=42, stratify=y_train if num_classes > 1 else None
    )

    # Scale based on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Optionally cache the scaler
    if scaler_cache_path is None:
        scaler_cache_path = os.path.join(base_data_dir, f"scaler_{dataset_name}.json")
    try:
        with open(scaler_cache_path, "w") as f:
            # Ensure proper typing for static analysis and robustness
            mean_arr = np.asarray(getattr(scaler, "mean_", None), dtype=np.float64)
            scale_arr = np.asarray(getattr(scaler, "scale_", None), dtype=np.float64)
            json.dump({"mean": mean_arr.tolist(), "scale": scale_arr.tolist()}, f)
    except Exception as e:
        print(f"Warning: could not cache scaler to {scaler_cache_path}: {e}")

    input_dim = X_train.shape[1]

    train_ds = TimeSeriesCSVDataset(X_train, y_train, seq_len)
    val_ds = TimeSeriesCSVDataset(X_val, y_val, seq_len)
    test_ds = TimeSeriesCSVDataset(X_test, y_test, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, input_dim, num_classes


__all__ = ["DataConfig", "create_dataloaders", "TimeSeriesCSVDataset"]

def load_scaler_from_json(path: str) -> Optional[StandardScaler]:
    """Reconstruct a StandardScaler from a JSON cache created during training.

    Returns None if the file can't be read.
    """
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        scaler = StandardScaler()
        # Fit dummy to allocate attributes, then overwrite
        scaler.mean_ = np.array(obj["mean"], dtype=np.float64)
        scaler.scale_ = np.array(obj["scale"], dtype=np.float64)
        scaler.var_ = scaler.scale_ ** 2  # approximate, only used in inverse_transform rarely
        scaler.n_features_in_ = scaler.mean_.shape[0]
        return scaler
    except Exception as e:
        print(f"Warning: could not load scaler from {path}: {e}")
        return None

__all__.append("load_scaler_from_json")
