"""
Realtime Packet Capture -> Feature Windows -> Model Inference -> Alerts
---------------------------------------------------------------------
Lightweight realtime pipeline for Windows using Scapy. Aggregates packets
into short time windows, computes the same 20 features as pcap_converter.py,
reuses predict.py helpers, and emits alerts to:
  - results/alerts_chain.json (tamper-evident)
  - dashboard (POST /api/dashboard/ingest) for live UI via SSE

Run (PowerShell, Administrator):
  python src/realtime.py --iface "Ethernet" --window 3 --checkpoint checkpoints/best_iot23.pt

Requirements:
  - Windows: Install Npcap (https://nmap.org/npcap/)
  - Python deps: scapy (already in requirements), requests
"""
from __future__ import annotations

import argparse
import time
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from scapy.all import sniff, IP, TCP, UDP, Raw
except Exception as e:
    raise SystemExit("Scapy is required for realtime capture. Install with: pip install scapy")

# Local imports
from src.predict import (
    load_model_and_scaler,
    preprocess_traffic_data,
    create_sequences,
    get_severity,
)
from src.blockchain_logger import BlockchainLogger

try:
    import requests
except Exception:
    requests = None  # We'll fall back to file logging only if requests is missing


FEATURE_COLUMNS = [
    'packet_rate', 'packet_size', 'byte_rate', 'flow_duration',
    'total_packets', 'total_bytes', 'entropy', 'port_scan_score',
    'syn_flag_count', 'ack_flag_count', 'fin_flag_count', 'rst_flag_count',
    'psh_flag_count', 'urg_flag_count', 'unique_src_ports', 'unique_dst_ports',
    'payload_entropy', 'dns_query_count', 'http_request_count', 'ssl_handshake_count'
]


@dataclass
class WindowStats:
    start_ts: float
    total_packets: int = 0
    total_bytes: int = 0
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    urg_count: int = 0
    src_ports: set = None
    dst_ports: set = None
    dns_queries: int = 0
    http_requests: int = 0
    ssl_handshakes: int = 0
    payload_data: bytearray = None
    pkt_sizes: List[int] = None

    def __post_init__(self):
        self.src_ports = set()
        self.dst_ports = set()
        self.payload_data = bytearray()
        self.pkt_sizes = []


def shannon_entropy(buf: bytes) -> float:
    if not buf:
        return 0.0
    counts = np.bincount(np.frombuffer(buf, dtype=np.uint8), minlength=256).astype(np.float64)
    probs = counts / counts.sum() if counts.sum() > 0 else counts
    nz = probs[probs > 0]
    return float(-(nz * np.log2(nz)).sum())


class RealtimeIDS:
    def __init__(self, iface: str, window_sec: float, checkpoint: str, dataset: str, device: str, seq_len: int,
                 dashboard_url: str = "http://localhost:5000"):
        self.iface = iface
        self.window_sec = window_sec
        self.seq_len = seq_len
        self.dashboard_url = dashboard_url.rstrip('/')

        # Load model and scaler once
        self.model, self.scaler, self.input_dim, self.num_classes = load_model_and_scaler(
            checkpoint, dataset, device
        )

        # Alert chain logger
        self.chain_logger = BlockchainLogger(chain_path=self._results_path('alerts_chain.json'))

        # Current window stats
        self.cur = WindowStats(start_ts=time.time())

        # Control
        self._stop = threading.Event()

    @staticmethod
    def _results_path(name: str) -> str:
        import os
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, name)

    def _reset_window(self, now: float):
        self.cur = WindowStats(start_ts=now)

    def _accumulate(self, pkt):
        try:
            self.cur.total_packets += 1
            size = int(len(pkt))
            self.cur.total_bytes += size
            self.cur.pkt_sizes.append(size)

            if TCP in pkt:
                flags = pkt[TCP].flags
                # Flag counters
                if flags & 0x02:
                    self.cur.syn_count += 1
                if flags & 0x10:
                    self.cur.ack_count += 1
                if flags & 0x01:
                    self.cur.fin_count += 1
                if flags & 0x04:
                    self.cur.rst_count += 1
                if flags & 0x08:
                    self.cur.psh_count += 1
                if flags & 0x20:
                    self.cur.urg_count += 1

                # Ports
                self.cur.src_ports.add(int(pkt[TCP].sport))
                self.cur.dst_ports.add(int(pkt[TCP].dport))

                # SSL heuristic
                if int(pkt[TCP].dport) == 443 or int(pkt[TCP].sport) == 443:
                    self.cur.ssl_handshakes += 1

            if UDP in pkt:
                self.cur.src_ports.add(int(pkt[UDP].sport))
                self.cur.dst_ports.add(int(pkt[UDP].dport))

            # DNS heuristic
            # Lightweight: count UDP/53 or TCP/53 traffic as DNS query indicators
            try:
                if (UDP in pkt and (pkt[UDP].sport == 53 or pkt[UDP].dport == 53)) or \
                   (TCP in pkt and (pkt[TCP].sport == 53 or pkt[TCP].dport == 53)):
                    self.cur.dns_queries += 1
            except Exception:
                pass

            # HTTP heuristic (port 80)
            try:
                if TCP in pkt and (pkt[TCP].sport == 80 or pkt[TCP].dport == 80):
                    self.cur.http_requests += 1
            except Exception:
                pass

            # Payload for entropy (cap to 100 bytes per packet)
            if Raw in pkt:
                payload = bytes(pkt[Raw].load[:100])
                self.cur.payload_data.extend(payload)
        except Exception:
            # Swallow any scapy field parsing issues
            pass

    def _window_features(self, now: float) -> Dict[str, float]:
        dur = max(now - self.cur.start_ts, 1e-6)
        avg_pkt_size = float(np.mean(self.cur.pkt_sizes)) if self.cur.pkt_sizes else 0.0
        packet_rate = self.cur.total_packets / dur
        byte_rate = self.cur.total_bytes / dur
        entropy = shannon_entropy(bytes(self.cur.payload_data)) if self.cur.payload_data else 0.0
        port_scan_score = len(self.cur.dst_ports) / max(self.cur.total_packets, 1)

        return {
            'packet_rate': packet_rate,
            'packet_size': avg_pkt_size,
            'byte_rate': byte_rate,
            'flow_duration': dur,
            'total_packets': self.cur.total_packets,
            'total_bytes': self.cur.total_bytes,
            'entropy': entropy,
            'port_scan_score': port_scan_score,
            'syn_flag_count': self.cur.syn_count,
            'ack_flag_count': self.cur.ack_count,
            'fin_flag_count': self.cur.fin_count,
            'rst_flag_count': self.cur.rst_count,
            'psh_flag_count': self.cur.psh_count,
            'urg_flag_count': self.cur.urg_count,
            'unique_src_ports': len(self.cur.src_ports),
            'unique_dst_ports': len(self.cur.dst_ports),
            'payload_entropy': entropy,
            'dns_query_count': self.cur.dns_queries,
            'http_request_count': self.cur.http_requests,
            'ssl_handshake_count': self.cur.ssl_handshakes,
        }

    def _emit_alert(self, features: Dict[str, float]):
        # Convert to DataFrame with the exact column order used in training
        row = pd.DataFrame([[features.get(col, 0.0) for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

        # Preprocess -> sequence -> predict
        X = preprocess_traffic_data(row, self.scaler, self.input_dim)
        X_seq = create_sequences(X, self.seq_len)

        import torch
        with torch.no_grad():
            seq = torch.from_numpy(X_seq[0]).unsqueeze(0)  # [1, T, F]
            logits = self.model(seq)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())

        # Map to severity string using existing helper
        from src.predict import ATTACK_TYPES
        attack_type = ATTACK_TYPES.get(pred, f"Unknown_{pred}")
        severity = get_severity(attack_type, conf)

        alert = {
            "attack_type": attack_type,
            "severity": severity,
            "confidence": conf,
            "features": {
                # Subset used by dashboard explanations
                "packet_rate": features.get("packet_rate", 0.0),
                "packet_size": features.get("packet_size", 0.0),
                "byte_rate": features.get("byte_rate", 0.0),
                "flow_duration": features.get("flow_duration", 0.0),
                "entropy": features.get("entropy", 0.0),
                "src_port": 0,
                "dst_port": 0,
                "total_packets": features.get("total_packets", 0.0),
            },
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')
        }

        # Append to blockchain
        self.chain_logger.append_alert(alert)

        # Try to send to dashboard ingest endpoint for live SSE + Threat DB
        try:
            if requests is not None:
                requests.post(f"{self.dashboard_url}/api/dashboard/ingest", json=alert, timeout=1.0)
        except Exception:
            # Dashboard not running; ignore
            pass

        # Also print to console
        print(f"[ALERT] {severity:<8} type={attack_type:<12} conf={conf:.3f} rate={features.get('packet_rate',0):.1f}pps")

    def _sniff_loop(self):
        def on_pkt(pkt):
            # Only process IP traffic
            if IP not in pkt:
                return
            self._accumulate(pkt)

        # Start scapy sniff
        sniff(iface=self.iface, store=False, prn=on_pkt, stop_filter=lambda x: self._stop.is_set())

    def _window_loop(self):
        # Periodically compute features and run inference
        last_emit = time.time()
        while not self._stop.is_set():
            time.sleep(0.1)
            now = time.time()
            if now - last_emit >= self.window_sec:
                # Skip empty windows
                if self.cur.total_packets > 0:
                    feats = self._window_features(now)
                    try:
                        self._emit_alert(feats)
                    except Exception as e:
                        print(f"[WARN] Inference failed: {e}")
                # Reset window
                self._reset_window(now)
                last_emit = now

    def start(self):
        t1 = threading.Thread(target=self._sniff_loop, daemon=True)
        t2 = threading.Thread(target=self._window_loop, daemon=True)
        t1.start(); t2.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping realtime IDS...")
            self._stop.set()


def main():
    p = argparse.ArgumentParser(description="Realtime IDS (Scapy) -> Model -> Dashboard")
    p.add_argument('--iface', type=str, default='Ethernet', help='Network interface name (e.g., Ethernet, Wi-Fi)')
    p.add_argument('--window', type=float, default=3.0, help='Aggregation window in seconds')
    p.add_argument('--seq-len', type=int, default=100, help='Sequence length for model')
    p.add_argument('--checkpoint', type=str, default='checkpoints/best_iot23.pt', help='Model checkpoint path')
    p.add_argument('--dataset', type=str, default='iot23', help='Dataset name for scaler lookup')
    p.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    p.add_argument('--dashboard', type=str, default='http://localhost:5000', help='Dashboard base URL for ingest')
    args = p.parse_args()

    print("=" * 70)
    print("🛰️  Realtime IDS starting...")
    print(f"🧩 Interface : {args.iface}")
    print(f"⏱️  Window    : {args.window}s")
    print(f"🧠 Checkpoint: {args.checkpoint}")
    print(f"🖥️  Device    : {args.device}")
    print(f"📡 Dashboard : {args.dashboard}")
    print("=" * 70)

    ids = RealtimeIDS(
        iface=args.iface,
        window_sec=args.window,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        device=args.device,
        seq_len=args.seq_len,
        dashboard_url=args.dashboard,
    )
    ids.start()


if __name__ == '__main__':
    main()
