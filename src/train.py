"""
train.py - Training script for Next-Gen IDS
------------------------------------------
Trains the IDSModel on the specified dataset, logs metrics, and saves the best checkpoint.
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import IDSModel, NextGenIDS
from src.data_loader import create_dataloaders
from src.utils import set_seed as utils_set_seed, save_metrics_csv

def set_seed(seed=42):
    # Keep local for backward-compat, delegate to utils
    utils_set_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Train IDSModel")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., iot23, beth)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--save_path', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', 'best.pt'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--results_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results'))
    parser.add_argument('--use-arnn', action='store_true', help='Use NextGenIDS (A-RNN + S-LSTM + CNN) instead of IDSModel')
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    # Ensure results directory exists for CSV logging and artifacts
    os.makedirs(args.results_dir, exist_ok=True)

    # Load data
    train_loader, val_loader, _, input_dim, num_classes = create_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=0
    )

    # Model - choose architecture based on flag
    if args.use_arnn:
        print("🚀 Using NextGenIDS (A-RNN + S-LSTM + CNN)")
        model = NextGenIDS(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(args.device)
    else:
        print("📦 Using IDSModel (S-LSTM + CNN)")
        model = IDSModel(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(args.device)
    model.count_parameters()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        total_train_samples = 0
        train_preds, train_labels = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for X, y in pbar:
            X, y = X.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            batch_sz = int(X.size(0))
            train_loss += loss.item() * batch_sz
            total_train_samples += batch_sz
            train_preds.extend(logits.argmax(1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())
        if total_train_samples > 0:
            train_loss /= float(total_train_samples)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        # Log train metrics to CSV (append)
        save_metrics_csv({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'train_f1': float(train_f1)
        }, os.path.join(args.results_dir, 'metrics.csv'))

        # Validation
        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(args.device), y.to(args.device)
                logits = model(X)
                loss = criterion(logits, y)
                batch_sz = int(X.size(0))
                val_loss += loss.item() * batch_sz
                total_val_samples += batch_sz
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        if total_val_samples > 0:
            val_loss /= float(total_val_samples)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        scheduler.step(val_f1)
        # Log val metrics to CSV (append)
        save_metrics_csv({
            'epoch': epoch,
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'val_f1': float(val_f1)
        }, os.path.join(args.results_dir, 'metrics.csv'))

        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'state_dict': model.state_dict(), 'meta': {
                'input_dim': input_dim,
                'num_classes': num_classes,
                'epoch': epoch,
                'f1': val_f1
            }}, args.save_path)
            print(f"[Checkpoint] Saved best model to {args.save_path}")

    print(f"Training complete. Best val F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()
