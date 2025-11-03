"""
SHAP Explainability Script
--------------------------
Loads the best checkpoint, computes SHAP values for a sample of the test set,
and saves plots and CSVs under results/shap_plots.

CLI:
    python src/explain_shap.py --dataset iot23 --checkpoint checkpoints/best_iot23.pt
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

from src.data_loader import create_dataloaders
from src.model import IDSModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explain IDS predictions with SHAP")
    p.add_argument("--dataset", choices=["iot23", "beth"], required=True)
    p.add_argument("--checkpoint", type=str, required=False)
    p.add_argument("--results-dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "shap_plots"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-samples", type=int, default=128, help="Number of test sequences to explain")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    train_loader, val_loader, test_loader, input_dim, num_classes = create_dataloaders(
        dataset_name=args.dataset,
        batch_size=64,
        seq_len=100,
        num_workers=0,
    )

    model = IDSModel(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(args.device)

    if not args.checkpoint:
        args.checkpoint = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", f"best_{args.dataset}.pt")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Collect a small batch from test set
    xs = []
    ys = []
    for i, (x, y) in enumerate(test_loader):
        xs.append(x)
        ys.append(y)
        if sum(len(t) for t in xs) >= args.num_samples:
            break
    x_batch = torch.cat(xs, dim=0)[: args.num_samples].to(args.device)

    # Use a small background dataset for KernelExplainer if on CPU; on GPU use DeepExplainer
    try:
        if args.device.startswith("cuda"):
            explainer = shap.DeepExplainer(model, x_batch[:32])
            shap_values = explainer.shap_values(x_batch)
        else:
            f = lambda inp: model(torch.tensor(inp, dtype=torch.float32, device=args.device)).detach().cpu().numpy()
            background = x_batch[:32].cpu().numpy()
            explainer = shap.KernelExplainer(f, background)
            shap_values = explainer.shap_values(x_batch[: args.num_samples].cpu().numpy(), nsamples=100)
    except Exception as e:
        print(f"SHAP explanation failed or fell back: {e}. Trying GradientExplainer.")
        explainer = shap.GradientExplainer((model, x_batch[:32]), x_batch[:32])
        shap_values = explainer.shap_values(x_batch)

    # Save a summary plot for the top class
    plot_path = os.path.join(args.results_dir, "shap_summary.png")
    try:
        # shap_values may be list for multiclass
        sv = shap_values if isinstance(shap_values, np.ndarray) else shap_values[0]
        shap.summary_plot(sv, features=x_batch.detach().cpu().numpy(), show=False)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved SHAP summary plot to {plot_path}")
    except Exception as e:
        print(f"Could not save SHAP plot: {e}")


if __name__ == "__main__":
    main()
