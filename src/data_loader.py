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
        print(f"Target column '{target_column}' not found. Creating dummy labels (all zeros).")
        df[target_column] = 0

    # Keep only numeric columns as features
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Remove the target column from features if it is numeric
    if target_column in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_column])

    y = df[target_column].astype(int)

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

    # Encode labels to 0..C-1
    classes, y_idx = np.unique(y, return_inverse=True)
    num_classes = len(classes)

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
