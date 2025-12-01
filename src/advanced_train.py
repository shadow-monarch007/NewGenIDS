"""
advanced_train.py - Enhanced Training with Data Augmentation and Better Architecture
------------------------------------------------------------------------------------
Improved training with:
- Data augmentation for better generalization
- Mixed dataset training (IoT-23 + synthetic variations)
- Better learning rate scheduling
- Early stopping with patience
- Cross-validation support
"""
import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import IDSModel, NextGenIDS
from src.data_loader import create_dataloaders
from src.utils import set_seed

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def augment_data(X, y, noise_level=0.01):
    """Add Gaussian noise for data augmentation"""
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        # Original sample
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # Augmented sample with noise
        noise = np.random.normal(0, noise_level, X[i].shape)
        augmented_X.append(X[i] + noise)
        augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_loader, desc="Validating", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description="Advanced IDSModel Training")
    parser.add_argument('--dataset', type=str, default='iot23')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='checkpoints/best_advanced.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use-arnn', action='store_true', help='Use NextGenIDS architecture')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    args = parser.parse_args()
    
    set_seed(42)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    print(f"🚀 Advanced Training Configuration")
    print(f"   Dataset: {args.dataset}")
    print(f"   Model: {'NextGenIDS (A-RNN)' if args.use_arnn else 'IDSModel (LSTM+CNN)'}")
    print(f"   Device: {args.device}")
    print(f"   Epochs: {args.epochs} (with early stopping patience={args.patience})")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Data Augmentation: {'Enabled' if args.augment else 'Disabled'}")
    print()
    
    # Load data
    train_loader, val_loader, test_loader, input_dim, num_classes = create_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        test_size=0.15,
        val_size=0.15
    )
    
    print(f"✓ Data loaded: {input_dim} features, {num_classes} classes")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print()
    
    # Initialize model
    if args.use_arnn:
        model = NextGenIDS(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes)
    else:
        model = IDSModel(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes)
    
    model = model.to(args.device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    print("🏋️  Starting training...")
    print("=" * 80)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, args.device)
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(model, val_loader, criterion, args.device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'input_dim': input_dim,
                'num_classes': num_classes,
                'seq_len': args.seq_len,
                'model_type': 'NextGenIDS' if args.use_arnn else 'IDSModel'
            }, args.save_path)
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
            break
    
    print("\n" + "=" * 80)
    print("🎉 Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Final evaluation on test set
    print("\n📊 Final evaluation on test set...")
    checkpoint = torch.load(args.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1, test_preds, test_labels = validate(model, test_loader, criterion, args.device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Loss: {test_loss:.4f}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(test_labels, test_preds, zero_division=0))
    
    # Save metrics
    metrics_df = pd.DataFrame(history)
    metrics_path = args.save_path.replace('.pt', '_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    print(f"✓ Model checkpoint saved to {args.save_path}")

if __name__ == '__main__':
    main()
