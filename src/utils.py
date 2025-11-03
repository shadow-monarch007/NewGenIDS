"""
utils.py - Utilities for Next-Gen IDS
------------------------------------
Seed fixing, metrics computation, and plotting helpers used across training/evaluation.
"""
from __future__ import annotations

import os
import random
from typing import Dict, Iterable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    """Compute common classification metrics (weighted)."""
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def plot_confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], out_path: str) -> str:
    """Create and save a confusion matrix plot. Returns the saved path."""
    cm = confusion_matrix(list(y_true), list(y_pred))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def save_metrics_csv(metrics: Dict[str, float], csv_path: str, append: bool = True) -> str:
    """Save metrics dict to a CSV file as key,value rows."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    line_sep = "\n"
    header_needed = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    mode = "a" if append else "w"
    with open(csv_path, mode) as f:
        if header_needed:
            f.write("metric,value" + line_sep)
        for k, v in metrics.items():
            f.write(f"{k},{v}" + line_sep)
    return csv_path
