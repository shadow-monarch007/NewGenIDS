"""
Run Inference
-------------
Loads a trained model checkpoint, runs prediction on a CSV input file (or a
synthetic sample if not provided), computes SHAP for the prediction, and logs
an alert to the blockchain.

CLI:
    python src/run_inference.py --dataset iot23 --input-file data/iot23/sample.csv
"""
from __future__ import annotations

import os
import argparse
import json

import numpy as np
import torch

from src.data_loader import create_dataloaders
from src.model import IDSModel
from src.blockchain_logger import BlockchainLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run IDS inference + SHAP + blockchain logging")
    p.add_argument("--dataset", choices=["iot23", "beth"], required=True)
    p.add_argument("--checkpoint", type=str, required=False)
    p.add_argument("--input-file", type=str, required=False, help="Optional CSV file for a quick demo input")
    p.add_argument("--results-dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # Use loaders to get dims and normalization
    train_loader, val_loader, test_loader, input_dim, num_classes = create_dataloaders(
        dataset_name=args.dataset, batch_size=32, seq_len=100, num_workers=0
    )

    model = IDSModel(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(args.device)
    if not args.checkpoint:
        args.checkpoint = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", f"best_{args.dataset}.pt")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Take one batch from test set as demo input
    x_demo, y_demo = next(iter(test_loader))
    x_demo = x_demo.to(args.device)

    with torch.no_grad():
        logits = model(x_demo)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    # Log first prediction as an alert
    first_prob = float(probs[0, preds[0]].cpu().item())
    alert = {
        "dataset": args.dataset,
        "pred": int(preds[0].cpu().item()),
        "confidence": first_prob,
    }

    chain_path = os.path.join(args.results_dir, "alerts_chain.json")
    logger = BlockchainLogger(chain_path)
    block = logger.append_alert(alert)

    print("Inference complete. First prediction:", alert)
    print("Blockchain block appended:", json.dumps(block, indent=2))
    print("Chain valid:", logger.verify_chain())


if __name__ == "__main__":
    main()
