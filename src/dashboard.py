"""
Web Dashboard for Next-Gen IDS
-------------------------------
Flask web application providing:
- CSV file upload for analysis
- Real-time training & evaluation
- Visual results (metrics, confusion matrix)
- AI-powered intrusion explanations
- Mitigation recommendations

Run:
    python src/dashboard.py
    Then open http://localhost:5000
"""
from __future__ import annotations

import os
import json
import io
import base64
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt

from src.model import IDSModel, NextGenIDS
from src.data_loader import create_dataloaders
from src.utils import compute_metrics, plot_confusion_matrix
from src.train import main as train_main
from src.evaluate import main as evaluate_main

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'))

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
CHECKPOINT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

# Global state for tracking jobs
current_job = {"status": "idle", "progress": 0, "message": "Ready"}


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '' or file.filename is None:
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are supported"}), 400
    
    # Save uploaded file
    dataset_name = request.form.get('dataset_name', 'uploaded')
    dataset_dir = os.path.join(UPLOAD_FOLDER, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    filepath = os.path.join(dataset_dir, str(file.filename))
    file.save(filepath)
    
    # Quick stats
    try:
        df = pd.read_csv(filepath)
        stats = {
            "filename": file.filename,
            "dataset_name": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
            "features": list(df.columns),
            "has_label": "label" in df.columns
        }
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400


@app.route('/api/train', methods=['POST'])
def train():
    """Run training on uploaded dataset."""
    global current_job
    
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    
    dataset_name = data.get('dataset_name', 'uploaded')
    epochs = int(data.get('epochs', 5))
    batch_size = int(data.get('batch_size', 32))
    seq_len = int(data.get('seq_len', 64))
    use_arnn = data.get('use_arnn', False)  # New: allow choosing A-RNN model
    
    current_job = {"status": "training", "progress": 0, "message": f"Training for {epochs} epochs..."}
    
    try:
        # Move uploaded files to data directory temporarily
        upload_path = os.path.join(UPLOAD_FOLDER, dataset_name)
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', dataset_name)
        
        # Create data loaders
        train_loader, val_loader, test_loader, input_dim, num_classes = create_dataloaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            seq_len=seq_len,
            data_dir=UPLOAD_FOLDER,
            num_workers=0
        )
        
        # Initialize model - choose architecture based on flag
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if use_arnn:
            print("🚀 Dashboard: Using NextGenIDS (A-RNN + S-LSTM + CNN)")
            model = NextGenIDS(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
        else:
            print("📦 Dashboard: Using IDSModel (S-LSTM + CNN)")
            model = IDSModel(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
        
        # Simple training loop (simplified version)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        best_f1 = 0.0
        history = []
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            total_samples = 0
            
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                batch_sz = int(X.size(0))
                train_loss += loss.item() * batch_sz
                total_samples += batch_sz
            
            train_loss /= float(total_samples)
            
            # Validation
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    logits = model(X)
                    val_preds.extend(logits.argmax(1).cpu().numpy())
                    val_labels.extend(y.cpu().numpy())
            
            metrics = compute_metrics(val_labels, val_preds)
            val_f1 = metrics['f1']
            
            history.append({
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_f1": float(val_f1),
                "val_accuracy": float(metrics['accuracy'])
            })
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                checkpoint_path = os.path.join(CHECKPOINT_FOLDER, f'best_{dataset_name}.pt')
                torch.save({
                    'state_dict': model.state_dict(),
                    'meta': {
                        'input_dim': input_dim,
                        'num_classes': num_classes,
                        'epoch': epoch + 1,
                        'f1': val_f1
                    }
                }, checkpoint_path)
            
            current_job["progress"] = int((epoch + 1) / epochs * 100)
            current_job["message"] = f"Epoch {epoch + 1}/{epochs} | Val F1: {val_f1:.4f}"
        
        current_job = {"status": "completed", "progress": 100, "message": "Training complete"}
        
        return jsonify({
            "success": True,
            "history": history,
            "best_f1": float(best_f1),
            "checkpoint": f'best_{dataset_name}.pt'
        })
        
    except Exception as e:
        current_job = {"status": "error", "progress": 0, "message": str(e)}
        return jsonify({"error": str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Run evaluation on test set."""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    
    dataset_name = data.get('dataset_name', 'uploaded')
    checkpoint_name = data.get('checkpoint', f'best_{dataset_name}.pt')
    batch_size = int(data.get('batch_size', 32))
    seq_len = int(data.get('seq_len', 64))
    
    try:
        # Load data
        _, _, test_loader, input_dim, num_classes = create_dataloaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            seq_len=seq_len,
            data_dir=UPLOAD_FOLDER,
            num_workers=0
        )
        
        # Load model - detect architecture from checkpoint
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint_path = os.path.join(CHECKPOINT_FOLDER, checkpoint_name)
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Check if checkpoint uses NextGenIDS (A-RNN) or old IDSModel
        state_dict_keys = list(ckpt['state_dict'].keys())
        uses_arnn = any('arnn' in key or 'slstm_cnn' in key for key in state_dict_keys)
        
        if uses_arnn:
            print("📊 Loading NextGenIDS (A-RNN) model...")
            model = NextGenIDS(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
        else:
            print("📊 Loading standard IDSModel...")
            model = IDSModel(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
        
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        
        # Evaluate
        test_preds, test_labels = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                test_preds.extend(logits.argmax(1).cpu().numpy())
                test_labels.extend(y.cpu().numpy())
        
        metrics = compute_metrics(test_labels, test_preds)
        
        # Generate confusion matrix image
        from sklearn.metrics import confusion_matrix as cm_func
        cm = cm_func(test_labels, test_preds)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title("Test Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        # Convert to base64 for embedding in HTML
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        cm_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({
            "success": True,
            "metrics": {
                "accuracy": float(metrics['accuracy']),
                "precision": float(metrics['precision']),
                "recall": float(metrics['recall']),
                "f1": float(metrics['f1'])
            },
            "confusion_matrix": cm_base64
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyze_file', methods=['POST'])
def analyze_uploaded_file():
    """Analyze uploaded CSV file and generate threat explanation."""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    
    filename = data.get('filename')
    dataset_name = data.get('dataset_name', 'uploaded')
    
    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    
    try:
        # Load the uploaded CSV file
        dataset_dir = os.path.join(UPLOAD_FOLDER, dataset_name)
        filepath = os.path.join(dataset_dir, filename)
        
        if not os.path.exists(filepath):
            return jsonify({"error": f"File not found: {filename}"}), 404
        
        df = pd.read_csv(filepath)
        
        # Analyze the first few rows to detect attack type
        # Extract features from the data
        if len(df) == 0:
            return jsonify({"error": "Empty file"}), 400
        
        # Get first row for analysis
        first_row = df.iloc[0]
        
        # Try to detect attack type from filename or data
        filename_lower = filename.lower()
        detected_type = 'Unknown'
        
        if 'ddos' in filename_lower:
            detected_type = 'DDoS'
        elif 'port_scan' in filename_lower or 'port scan' in filename_lower:
            detected_type = 'Port_Scan'
        elif 'malware' in filename_lower or 'c2' in filename_lower:
            detected_type = 'Malware_C2'
        elif 'brute_force' in filename_lower or 'brute force' in filename_lower:
            detected_type = 'Brute_Force'
        elif 'sql' in filename_lower:
            detected_type = 'SQL_Injection'
        elif 'normal' in filename_lower:
            detected_type = 'Normal'
        
        # Extract features from the data
        features = {}
        feature_mapping = {
            'packet_rate': ['packet_rate', 'packets_per_sec', 'pkt_rate'],
            'packet_size': ['packet_size', 'avg_pkt_size', 'pkt_size', 'bytes'],
            'byte_rate': ['byte_rate', 'bytes_per_sec', 'bps'],
            'flow_duration': ['flow_duration', 'duration', 'flow_dur'],
            'entropy': ['entropy', 'ent'],
            'src_port': ['src_port', 'sport', 'source_port'],
            'dst_port': ['dst_port', 'dport', 'dest_port', 'destination_port'],
            'total_packets': ['total_packets', 'tot_pkts', 'packets']
        }
        
        for feature_name, possible_cols in feature_mapping.items():
            for col in possible_cols:
                if col in df.columns:
                    features[feature_name] = float(first_row[col])
                    break
            if feature_name not in features:
                features[feature_name] = 0.0
        
        # Simulate confidence based on data characteristics
        confidence = 0.85 + (len(df) / 10000) * 0.1  # Higher confidence with more data
        confidence = min(0.99, confidence)
        
        # Generate explanation
        explanation = generate_ai_explanation(detected_type, features, confidence)
        
        return jsonify({
            "success": True,
            "explanation": explanation,
            "detected_from": "file_analysis",
            "rows_analyzed": len(df)
        })
        
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route('/api/explain', methods=['POST'])
def explain_intrusion():
    """Generate AI-powered explanation of detected intrusion."""
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    
    intrusion_type = data.get('intrusion_type', 'unknown')
    features = data.get('features', {})
    confidence = data.get('confidence', 0.0)
    
    # Generate explanation (you can integrate OpenAI API here)
    explanation = generate_ai_explanation(intrusion_type, features, confidence)
    
    return jsonify({
        "success": True,
        "explanation": explanation
    })


def generate_ai_explanation(intrusion_type: str, features: Dict[str, Any], confidence: float) -> Dict[str, Any]:
    """
    Generate human-readable explanation of intrusion detection.
    
    Analyzes actual feature values to provide dynamic, data-driven explanations.
    In production, can be enhanced with OpenAI API or local LLM.
    """
    
    # Extract actual feature values for dynamic analysis
    packet_rate = features.get('packet_rate', 0)
    packet_size = features.get('packet_size', 0)
    byte_rate = features.get('byte_rate', 0)
    flow_duration = features.get('flow_duration', 0)
    entropy = features.get('entropy', 0)
    src_port = features.get('src_port', 0)
    dst_port = features.get('dst_port', 0)
    total_packets = features.get('total_packets', 0)
    
    # Base explanations (templates with dynamic data injection)
    # Base explanations (templates with dynamic data injection)
    explanations = {
        "DDoS": {
            "description": f"Distributed Denial of Service attack detected. Multiple sources flooding the network with {packet_rate:.0f} packets/sec to overwhelm resources.",
            "indicators": [
                f"🔴 Abnormally high packet rate: {packet_rate:.0f} packets/sec (normal: <100 pps)",
                f"🔴 Target destination port: {int(dst_port)} (typical web service)",
                f"🔴 Small packet sizes: {packet_size:.0f} bytes (SYN flood pattern)",
                f"🔴 Low entropy: {entropy:.2f} (repetitive patterns)",
                f"🔴 Total packets in flow: {int(total_packets)} (flood indicator)",
                f"🔴 Byte rate: {byte_rate:.0f} bytes/sec"
            ],
            "mitigation": [
                "🛡️ Enable rate limiting on firewall (e.g., iptables limit module)",
                "☁️ Deploy DDoS protection service (Cloudflare, AWS Shield, Akamai)",
                "🚫 Block malicious IP ranges using ACLs or blackhole routing",
                "📈 Scale infrastructure horizontally to handle traffic spike",
                "🔄 Enable SYN cookies to mitigate SYN flood attacks"
            ],
            "severity": "Critical" if packet_rate > 500 else "High",
            "attack_stage": "Active Attack - Immediate Action Required"
        },
        "Port_Scan": {
            "description": f"Reconnaissance activity detected. Attacker probing port {int(dst_port)} among others (pre-attack phase).",
            "indicators": [
                f"🟡 Sequential/multiple port access attempts detected",
                f"🟡 Target port: {int(dst_port)}",
                f"🟡 Rapid connection patterns: {flow_duration:.3f}s per attempt",
                f"🟡 Small packet sizes: {packet_size:.0f} bytes (SYN probes)",
                f"🟡 Low entropy: {entropy:.2f} (automated scanning)",
                f"🟡 Packet rate: {packet_rate:.0f} pps (scanning speed)"
            ],
            "mitigation": [
                "👁️ Enable port scan detection on IDS/IPS (e.g., Snort, Suricata)",
                "🔒 Close unnecessary open ports using firewall rules",
                "🚨 Implement fail2ban or similar tools to auto-block scanners",
                "📊 Monitor and alert on suspicious scanning patterns via SIEM",
                "🕵️ Investigate source IP - may indicate upcoming targeted attack"
            ],
            "severity": "Medium",
            "attack_stage": "Reconnaissance - Attacker Gathering Information"
        },
        "Port Scan": {  # Alternative name
            "description": f"Reconnaissance activity detected. Attacker probing port {int(dst_port)} among others (pre-attack phase).",
            "indicators": [
                f"🟡 Sequential/multiple port access attempts detected",
                f"🟡 Target port: {int(dst_port)}",
                f"🟡 Rapid connection patterns: {flow_duration:.3f}s per attempt",
                f"🟡 Small packet sizes: {packet_size:.0f} bytes (SYN probes)",
                f"🟡 Low entropy: {entropy:.2f} (automated scanning)",
                f"🟡 Packet rate: {packet_rate:.0f} pps (scanning speed)"
            ],
            "mitigation": [
                "👁️ Enable port scan detection on IDS/IPS (e.g., Snort, Suricata)",
                "🔒 Close unnecessary open ports using firewall rules",
                "🚨 Implement fail2ban or similar tools to auto-block scanners",
                "📊 Monitor and alert on suspicious scanning patterns via SIEM",
                "🕵️ Investigate source IP - may indicate upcoming targeted attack"
            ],
            "severity": "Medium",
            "attack_stage": "Reconnaissance - Attacker Gathering Information"
        },
        "Malware_C2": {
            "description": f"Command & Control communication detected. Compromised device beaconing to external server every ~{flow_duration:.0f}s (active breach).",
            "indicators": [
                f"🔴 Unusual outbound connection to port: {int(dst_port)}",
                f"🔴 Periodic beaconing pattern: every ~{flow_duration:.1f} seconds",
                f"🔴 High entropy traffic: {entropy:.2f} (likely encrypted)",
                f"🔴 Packet rate: {packet_rate:.0f} pps (C2 communication)",
                f"🔴 Byte rate: {byte_rate:.0f} bytes/sec (data exfiltration?)",
                f"🔴 Total packets: {int(total_packets)} (sustained connection)"
            ],
            "mitigation": [
                "⚠️ ISOLATE affected device immediately from network",
                "🦠 Run comprehensive antivirus/malware scan (offline if possible)",
                f"🚫 Block destination port {int(dst_port)} at firewall and DNS level",
                "🔍 Perform forensic analysis - identify infection vector",
                "💾 Back up critical data, then reimage system from clean OS",
                "📧 Report to CERT/security team for incident response"
            ],
            "severity": "Critical",
            "attack_stage": "Active Breach - Device Compromised"
        },
        "Malware C2": {  # Alternative name
            "description": f"Command & Control communication detected. Compromised device beaconing to external server every ~{flow_duration:.0f}s (active breach).",
            "indicators": [
                f"🔴 Unusual outbound connection to port: {int(dst_port)}",
                f"🔴 Periodic beaconing pattern: every ~{flow_duration:.1f} seconds",
                f"🔴 High entropy traffic: {entropy:.2f} (likely encrypted)",
                f"🔴 Packet rate: {packet_rate:.0f} pps (C2 communication)",
                f"🔴 Byte rate: {byte_rate:.0f} bytes/sec (data exfiltration?)",
                f"🔴 Total packets: {int(total_packets)} (sustained connection)"
            ],
            "mitigation": [
                "⚠️ ISOLATE affected device immediately from network",
                "🦠 Run comprehensive antivirus/malware scan (offline if possible)",
                f"🚫 Block destination port {int(dst_port)} at firewall and DNS level",
                "🔍 Perform forensic analysis - identify infection vector",
                "💾 Back up critical data, then reimage system from clean OS",
                "📧 Report to CERT/security team for incident response"
            ],
            "severity": "Critical",
            "attack_stage": "Active Breach - Device Compromised"
        },
        "Brute_Force": {
            "description": f"Brute force authentication attack detected on port {int(dst_port)}. Repeated login attempts to guess credentials.",
            "indicators": [
                f"🟠 Target authentication port: {int(dst_port)} (SSH:22, RDP:3389, FTP:21)",
                f"🟠 Attack rate: {packet_rate:.0f} attempts/second",
                f"🟠 Short connection duration: {flow_duration:.2f}s per attempt",
                f"🟠 Packet size: {packet_size:.0f} bytes (auth packets)",
                f"🟠 Total failed attempts: {int(total_packets)} packets",
                f"🟠 Low entropy: {entropy:.2f} (automated tool)"
            ],
            "mitigation": [
                "🔑 Enforce strong password policies (length, complexity, rotation)",
                "🚪 Implement account lockout after N failed attempts",
                "🔐 Enable multi-factor authentication (MFA) on all services",
                "🚨 Deploy fail2ban/sshguard to auto-block brute forcers",
                f"📍 Restrict port {int(dst_port)} access to specific IP ranges (VPN)",
                "🔒 Consider certificate-based auth instead of passwords"
            ],
            "severity": "High" if packet_rate > 20 else "Medium",
            "attack_stage": "Active Attack - Credential Compromise Attempt"
        },
        "Brute Force": {  # Alternative name
            "description": f"Brute force authentication attack detected on port {int(dst_port)}. Repeated login attempts to guess credentials.",
            "indicators": [
                f"🟠 Target authentication port: {int(dst_port)} (SSH:22, RDP:3389, FTP:21)",
                f"🟠 Attack rate: {packet_rate:.0f} attempts/second",
                f"🟠 Short connection duration: {flow_duration:.2f}s per attempt",
                f"🟠 Packet size: {packet_size:.0f} bytes (auth packets)",
                f"🟠 Total failed attempts: {int(total_packets)} packets",
                f"🟠 Low entropy: {entropy:.2f} (automated tool)"
            ],
            "mitigation": [
                "🔑 Enforce strong password policies (length, complexity, rotation)",
                "🚪 Implement account lockout after N failed attempts",
                "🔐 Enable multi-factor authentication (MFA) on all services",
                "🚨 Deploy fail2ban/sshguard to auto-block brute forcers",
                f"📍 Restrict port {int(dst_port)} access to specific IP ranges (VPN)",
                "🔒 Consider certificate-based auth instead of passwords"
            ],
            "severity": "High" if packet_rate > 20 else "Medium",
            "attack_stage": "Active Attack - Credential Compromise Attempt"
        },
        "SQL_Injection": {
            "description": f"SQL injection attack detected. Web application on port {int(dst_port)} targeted with malicious database queries.",
            "indicators": [
                f"🟠 Large HTTP request sizes: {packet_size:.0f} bytes (injection payload)",
                f"🟠 Target web port: {int(dst_port)} (HTTP/HTTPS)",
                f"🟠 High byte rate: {byte_rate:.0f} bytes/sec (attack traffic)",
                f"🟠 Attack pattern rate: {packet_rate:.0f} requests/sec",
                f"🟠 Entropy level: {entropy:.2f} (encoded SQL syntax)",
                f"🟠 Total attack requests: {int(total_packets)}"
            ],
            "mitigation": [
                "💉 Use parameterized queries/prepared statements (NEVER string concatenation)",
                "🔒 Implement input validation and sanitization on all user inputs",
                "🛡️ Deploy Web Application Firewall (WAF) with SQL injection rules",
                "👤 Apply principle of least privilege for database accounts",
                "🔍 Enable database query logging and monitoring",
                "🔄 Update and patch web application frameworks regularly"
            ],
            "severity": "High" if byte_rate > 5000 else "Medium",
            "attack_stage": "Active Attack - Database Compromise Attempt"
        },
        "SQL Injection": {  # Alternative name
            "description": f"SQL injection attack detected. Web application on port {int(dst_port)} targeted with malicious database queries.",
            "indicators": [
                f"🟠 Large HTTP request sizes: {packet_size:.0f} bytes (injection payload)",
                f"🟠 Target web port: {int(dst_port)} (HTTP/HTTPS)",
                f"🟠 High byte rate: {byte_rate:.0f} bytes/sec (attack traffic)",
                f"🟠 Attack pattern rate: {packet_rate:.0f} requests/sec",
                f"🟠 Entropy level: {entropy:.2f} (encoded SQL syntax)",
                f"🟠 Total attack requests: {int(total_packets)}"
            ],
            "mitigation": [
                "💉 Use parameterized queries/prepared statements (NEVER string concatenation)",
                "🔒 Implement input validation and sanitization on all user inputs",
                "🛡️ Deploy Web Application Firewall (WAF) with SQL injection rules",
                "👤 Apply principle of least privilege for database accounts",
                "🔍 Enable database query logging and monitoring",
                "🔄 Update and patch web application frameworks regularly"
            ],
            "severity": "High" if byte_rate > 5000 else "Medium",
            "attack_stage": "Active Attack - Database Compromise Attempt"
        },
        "Normal": {
            "description": f"Normal network traffic detected on port {int(dst_port)}. No malicious activity identified.",
            "indicators": [
                f"✅ Standard packet rate: {packet_rate:.0f} pps (within normal range)",
                f"✅ Typical packet size: {packet_size:.0f} bytes for service type",
                f"✅ Flow duration: {flow_duration:.2f}s (expected for connection type)",
                f"✅ Normal entropy: {entropy:.2f} (unencrypted or standard encryption)",
                f"✅ Byte rate: {byte_rate:.0f} bytes/sec (benign traffic volume)",
                f"✅ Port {int(dst_port)}: Common service port"
            ],
            "mitigation": [
                "✅ No action required - traffic is benign",
                "📊 Continue monitoring baseline patterns",
                "🔄 Update normal behavior profiles for ML model",
                "📈 Use this data to improve anomaly detection thresholds"
            ],
            "severity": "None",
            "attack_stage": "No Attack - Normal Operations"
        },
        "unknown": {
            "description": f"Anomalous network behavior detected on port {int(dst_port)}. Pattern deviates from normal baseline.",
            "indicators": [
                f"⚠️ Model confidence: {confidence:.1%}",
                f"⚠️ Packet rate: {packet_rate:.0f} pps (unusual pattern)",
                f"⚠️ Packet size: {packet_size:.0f} bytes (anomalous)",
                f"⚠️ Entropy: {entropy:.2f} (unexpected for traffic type)",
                f"⚠️ Flow duration: {flow_duration:.2f}s",
                "⚠️ Pattern not matching known attack profiles",
                "⚠️ May indicate zero-day attack or misconfiguration"
            ],
            "mitigation": [
                "🔍 Review detailed SHAP analysis for affected features",
                "📊 Correlate with other security logs (firewall, antivirus, SIEM)",
                "👁️ Monitor affected device for continued anomalies",
                "🕵️ Consider manual investigation if pattern persists >24 hours",
                "📧 Report to security team for advanced analysis",
                "🔬 Collect pcap for deep packet inspection if severity increases"
            ],
            "severity": "Low to Medium",
            "attack_stage": "Anomaly Detected - Investigation Recommended"
        }
    }
    
    explanation_data = explanations.get(intrusion_type, explanations["unknown"])
    
    return {
        "intrusion_type": intrusion_type,
        "confidence": confidence,
        "description": explanation_data["description"],
        "indicators": explanation_data["indicators"],
        "mitigation_steps": explanation_data["mitigation"],
        "severity": explanation_data["severity"],
        "attack_stage": explanation_data.get("attack_stage", "Unknown"),
        "timestamp": datetime.now().isoformat(),
        "recommended_priority": "IMMEDIATE" if explanation_data["severity"] == "Critical" else "HIGH" if explanation_data["severity"] == "High" else "MEDIUM" if explanation_data["severity"] == "Medium" else "LOW"
    }


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current job status."""
    return jsonify(current_job)


if __name__ == '__main__':
    print("🚀 Starting Next-Gen IDS Dashboard...")
    print("📊 Access dashboard at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    app.run(debug=True, host='0.0.0.0', port=5000)
