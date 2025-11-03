"""
evaluate.py - Evaluation script for Next-Gen IDS
-----------------------------------------------
Loads a trained IDSModel checkpoint and evaluates on the test set, reporting metrics.
"""
import os
import argparse
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from src.model import IDSModel
from src.data_loader import create_dataloaders
from src.utils import plot_confusion_matrix, save_metrics_csv

def main():
    parser = argparse.ArgumentParser(description="Evaluate IDSModel")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., iot23, beth)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--results_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results'))
    args = parser.parse_args()

    # Load data
    _, _, test_loader, input_dim, num_classes = create_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=0
    )

    # Model
    model = IDSModel(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(args.device)
            logits = model(X)
            preds = logits.argmax(1).cpu().numpy()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    # Save metrics
    os.makedirs(args.results_dir, exist_ok=True)
    save_metrics_csv({
        'test_acc': float(acc),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1)
    }, os.path.join(args.results_dir, 'metrics.csv'))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(args.results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
