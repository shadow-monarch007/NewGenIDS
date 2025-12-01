# Demo Test Script 2: Model Training
# Trains a new model on IoT23 dataset (quick 3 epochs for demo)

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "DEMO 2: Model Training" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

Write-Host "🚀 Training new model with A-RNN architecture..." -ForegroundColor Green
Write-Host "Parameters:" -ForegroundColor Gray
Write-Host "  - Dataset: IoT23" -ForegroundColor Gray
Write-Host "  - Epochs: 3 (quick demo)" -ForegroundColor Gray
Write-Host "  - Batch Size: 32" -ForegroundColor Gray
Write-Host "  - Sequence Length: 64" -ForegroundColor Gray
Write-Host "  - Architecture: A-RNN (Advanced Recurrent)" -ForegroundColor Gray
Write-Host ""

python src/train.py `
  --dataset iot23 `
  --epochs 3 `
  --batch_size 32 `
  --seq_len 64 `
  --use-arnn `
  --save_path checkpoints/demo_trained.pt

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "📊 Checking Trained Model Metadata:" -ForegroundColor Yellow
Write-Host "=" * 80 -ForegroundColor Cyan

python -c @"
import torch
ckpt = torch.load('checkpoints/demo_trained.pt', map_location='cpu')
meta = ckpt['meta']
print(f'  Input Features: {meta[\"input_dim\"]}')
print(f'  Attack Classes: {meta[\"num_classes\"]}')
print(f'  Final F1 Score: {meta[\"f1\"]:.4f}')
print(f'  Training Epoch: {meta[\"epoch\"]}')
"@

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "✅ Demo 2 Complete!" -ForegroundColor Green
Write-Host "Model saved to: checkpoints/demo_trained.pt" -ForegroundColor Gray
Write-Host "=" * 80 -ForegroundColor Cyan
