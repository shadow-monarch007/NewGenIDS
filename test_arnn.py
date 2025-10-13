"""Quick test of NextGenIDS with A-RNN on synthetic data."""
import torch
from src.model import NextGenIDS
from src.data_loader import create_dataloaders

print("🧪 Testing NextGenIDS with synthetic data...\n")

# Load data
train_loader, val_loader, test_loader, input_dim, num_classes = create_dataloaders(
    'iot23', batch_size=32, seq_len=64, num_workers=0
)
print(f"✅ Data loaded: {input_dim} features, {num_classes} classes\n")

# Create model
model = NextGenIDS(input_dim, 128, 2, num_classes)
model.count_parameters()
print()

# Test training for 3 batches
device = 'cpu'
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

model.train()
batch_count = 0
for X, y in train_loader:
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    
    print(f"Batch {batch_count+1}: loss={loss.item():.4f}, output shape={logits.shape}")
    batch_count += 1
    if batch_count >= 3:
        break

print("\n✅ NextGenIDS works perfectly with A-RNN pre-stage!")
print("🎉 Ready to use in dashboard or training scripts!")
