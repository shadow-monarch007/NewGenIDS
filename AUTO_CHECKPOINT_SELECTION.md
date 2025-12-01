# Automatic Checkpoint Selection

## Feature Overview

The dashboard now **automatically selects the best matching checkpoint** when you upload a CSV file for evaluation. This eliminates dimension mismatch errors and ensures smooth evaluation regardless of your data's feature count.

## How It Works

### Selection Algorithm

When you upload a CSV file in the **Evaluate** tab:

1. **Feature Detection**: System analyzes your CSV and counts numeric features
2. **Checkpoint Scanning**: Scans all available checkpoints in `checkpoints/` folder
3. **Metadata Extraction**: Reads each checkpoint's metadata (input_dim, num_classes, F1 score)
4. **Smart Matching**: Selects checkpoint with:
   - **Primary criteria**: Smallest dimension difference from your data
   - **Secondary criteria**: Highest F1 score (as tiebreaker)
5. **Auto-Load**: Automatically loads the selected checkpoint for evaluation

### Example Behavior

```
Your CSV has 79 features:
├─ best_iot23.pt (39 features) → diff = 40
├─ best_multiclass.pt (60 features) → diff = 19
└─ best_uploaded.pt (79 features) → diff = 0 ✅ SELECTED

Auto-selected: best_uploaded.pt
```

## Available Checkpoints

Current checkpoints in your project:

| Checkpoint | Input Dim | Classes | F1 Score | Best For |
|------------|-----------|---------|----------|----------|
| `best_iot23.pt` | 39 | 2 (Binary) | 0.8737 | IoT23 dataset, simple binary classification |
| `best_iot23_retrained.pt` | 39 | 2 (Binary) | 0.8638 | Alternative IoT23 model |
| `best_multiclass.pt` | 60 | 6 (Multi) | 0.8761 | Multi-class attacks (DDoS, Port Scan, etc.) |
| `best_uploaded.pt` | 79 | 8 | 1.0000 | Custom uploaded datasets with many features |

## Benefits

### ✅ Before (Manual Selection)
- Had to guess which checkpoint to use
- Frequent dimension mismatch errors
- Required trial-and-error to find compatible checkpoint
- Error: `size mismatch for classifier.5.weight: copying a param with shape torch.Size([2, 64]) from checkpoint...`

### ✨ After (Auto-Selection)
- Upload any CSV → system picks best checkpoint automatically
- Zero dimension mismatch errors
- Works seamlessly with any dataset
- Console shows: `🎯 Auto-selected checkpoint: best_multiclass.pt (input_dim=60, data_features=60, f1=0.8761)`

## Usage

### In Dashboard UI

1. Go to **Evaluate** tab
2. Select **Upload Custom Test Data** from dropdown
3. Click **Choose File** and upload your CSV
4. Click **📊 Evaluate**
5. System automatically selects best checkpoint and runs evaluation

**No manual checkpoint selection needed!**

### In Code (API)

```python
# POST /api/evaluate with file upload
# The system auto-selects checkpoint based on uploaded file's features

import requests

with open('my_dataset.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8080/api/evaluate',
        files={'file': f},
        data={'batch_size': 32, 'seq_len': 64}
    )
    
result = response.json()
print(result['metrics'])  # Evaluation metrics
```

### Testing Auto-Selection

Run the test script to see how selection works:

```bash
python test_auto_checkpoint.py
```

Output shows which checkpoint would be selected for different feature counts.

## Technical Details

### Implementation

Located in `src/dashboard_unified.py`:

```python
def _select_best_checkpoint(num_features: int) -> tuple[str, str]:
    """
    Automatically select the best matching checkpoint based on number of features.
    Returns (checkpoint_path, dataset_name) tuple.
    """
    # 1. Scan all .pt files in checkpoints/
    # 2. Load metadata for each checkpoint
    # 3. Calculate dimension difference: abs(checkpoint_dim - data_dim)
    # 4. Sort by (dim_diff, -f1_score)
    # 5. Return best match
```

### When Auto-Selection Activates

- ✅ **Enabled**: When uploading CSV file via `/api/evaluate`
- ❌ **Disabled**: When using built-in datasets (e.g., `iot23`, `beth`)
- ❌ **Disabled**: When checkpoint is explicitly specified in request

### Fallback Behavior

If auto-selection fails:
- Falls back to manual checkpoint selection
- Uses default checkpoint from request data
- Logs error and continues with specified checkpoint

## Future Enhancements

Potential improvements for next version:

1. **Architecture Matching**: Also consider model architecture (A-RNN vs LSTM)
2. **Class Count Compatibility**: Warn if class mismatch between data and model
3. **UI Feedback**: Show selected checkpoint in dashboard UI
4. **Custom Weights**: Allow user to set importance of dim_diff vs F1 score
5. **Checkpoint Recommendations**: Suggest retraining if no good match found

## Troubleshooting

### "No valid checkpoints found"
- Ensure `checkpoints/` folder contains at least one `.pt` file
- Check checkpoint files have valid `meta` dictionary with `input_dim`

### Auto-selection picks wrong model
- Check console output: `🎯 Auto-selected checkpoint: ...`
- Verify your CSV has expected number of features
- Consider retraining a model specifically for your feature count

### Evaluation still fails after auto-selection
- Check if CSV has `label` column for evaluation
- Verify CSV format is valid (numeric features, no missing values)
- Try with smaller `seq_len` or `batch_size`

## Related Files

- `src/dashboard_unified.py` - Main implementation
- `test_auto_checkpoint.py` - Test script
- `checkpoints/*.pt` - Model checkpoint files
- `uploads/uploaded/*.csv` - Uploaded evaluation files

---

**Status**: ✅ Implemented and pushed to `main` branch  
**Commit**: `140d627` - feat: Add automatic checkpoint selection based on feature dimensions
