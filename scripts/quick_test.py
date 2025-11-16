"""
Quick Test Script (Final-Year Project)
-------------------------------------
- Ensures the model can run on demo CSV
- Sends a sample alert to the dashboard ingest endpoint (simulating realtime)
- Verifies the blockchain chain integrity via API

Run:
  python scripts/quick_test.py --checkpoint checkpoints/best_iot23.pt
Prereqs:
  - Dashboard running: python src/dashboard_live.py
  - If no checkpoint, train first: python src/train.py --dataset iot23 --epochs 3 --use-arnn
"""
from __future__ import annotations

import os
import json
import argparse
from datetime import datetime

import requests

from src.predict import predict_traffic, get_severity
from src.blockchain_logger import BlockchainLogger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='checkpoints/best_iot23.pt')
    p.add_argument('--dashboard', type=str, default='http://localhost:5000')
    p.add_argument('--demo', type=str, default=os.path.join('data','iot23','demo_attacks.csv'))
    return p.parse_args()


def main():
    args = parse_args()

    print("== Quick Test ==")
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found at {args.checkpoint}.\nTrain first: python src/train.py --dataset iot23 --epochs 3 --use-arnn")
        return

    if not os.path.exists(args.demo):
        print(f"Demo CSV not found at {args.demo}")
        return

    # 1) Run prediction on demo CSV
    print(f"Running prediction on {args.demo} ...")
    preds = predict_traffic(
        csv_path=args.demo,
        checkpoint_path=args.checkpoint,
        dataset_name='iot23',
        device='cpu',
        seq_len=100,
    )
    if not preds:
        print("No predictions generated; demo too small?")
        return

    last = preds[-1]
    print("Prediction sample:", json.dumps(last, indent=2))

    # 2) Append to blockchain log directly
    results_dir = os.path.join('results')
    os.makedirs(results_dir, exist_ok=True)
    chain_path = os.path.join(results_dir, 'alerts_chain.json')
    logger = BlockchainLogger(chain_path)

    alert = {
        "attack_type": last["attack_type"],
        "severity": last["severity"],
        "confidence": float(last["confidence"]),
        "features": last.get("features", {}),
        "timestamp": last.get("timestamp") or datetime.utcnow().isoformat()+"Z",
        "source": "QuickTest"
    }
    logger.append_alert(alert)
    print("Appended one alert to blockchain.")

    # 3) Ingest to dashboard (simulates realtime post)
    try:
        r = requests.post(args.dashboard.rstrip('/') + '/api/dashboard/ingest', json=alert, timeout=3)
        print("Ingest status:", r.status_code, r.text)
    except Exception as e:
        print("Failed to POST to dashboard (is it running?):", e)

    # 4) Verify blockchain via API
    try:
        vr = requests.get(args.dashboard.rstrip('/') + '/api/blockchain/verify', timeout=3)
        print("Verify:", vr.status_code, vr.text)
    except Exception as e:
        print("Failed to verify via API:", e)

    print("== Done. Check dashboard Recent Alerts and stats.")


if __name__ == '__main__':
    main()
