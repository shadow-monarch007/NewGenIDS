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
    
    In production, integrate with OpenAI API or local LLM.
    For now, uses rule-based templates.
    """
    
    # Rule-based explanations (can be replaced with GPT API call)
    explanations = {
        "DDoS": {
            "description": "Distributed Denial of Service attack detected. Multiple sources flooding the network with traffic to overwhelm resources.",
            "indicators": [
                "🔴 Abnormally high packet rate (>500 packets/sec)",
                "🔴 Multiple source IPs targeting single destination port (typically 80/443)",
                "🔴 Small packet sizes (~64 bytes - SYN flood pattern)",
                "🔴 Repetitive packet patterns with low entropy",
                "🔴 Incomplete TCP handshakes (SYN without ACK)"
            ],
            "mitigation": [
                "🛡️ Enable rate limiting on firewall (e.g., iptables limit module)",
                "☁️ Deploy DDoS protection service (Cloudflare, AWS Shield, Akamai)",
                "🚫 Block malicious IP ranges using ACLs or blackhole routing",
                "📈 Scale infrastructure horizontally to handle traffic spike",
                "🔄 Enable SYN cookies to mitigate SYN flood attacks"
            ],
            "severity": "Critical",
            "attack_stage": "Active Attack - Immediate Action Required"
        },
        "Port_Scan": {
            "description": "Reconnaissance activity detected. Attacker systematically probing network to identify open ports and services (pre-attack phase).",
            "indicators": [
                "🟡 Sequential port access attempts (ports 1-65535)",
                "🟡 Connection attempts to multiple closed ports (RST responses)",
                "🟡 Rapid connection/disconnection patterns (<0.1s per port)",
                "🟡 Small packet sizes (~60 bytes - SYN probes)",
                "🟡 Single source IP targeting multiple destination ports"
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
            "description": "Reconnaissance activity detected. Attacker systematically probing network to identify open ports and services (pre-attack phase).",
            "indicators": [
                "🟡 Sequential port access attempts (ports 1-65535)",
                "🟡 Connection attempts to multiple closed ports (RST responses)",
                "🟡 Rapid connection/disconnection patterns (<0.1s per port)",
                "🟡 Small packet sizes (~60 bytes - SYN probes)",
                "🟡 Single source IP targeting multiple destination ports"
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
            "description": "Command & Control communication detected. Compromised device communicating with external attacker server (active breach).",
            "indicators": [
                "🔴 Unusual outbound connections to unknown servers",
                "🔴 Periodic beaconing patterns (every ~60 seconds)",
                "🔴 Encrypted traffic to suspicious domains/IPs",
                "🔴 High entropy data (encrypted payloads)",
                "🔴 Persistent PSH+ACK flags (data transmission)"
            ],
            "mitigation": [
                "⚠️ ISOLATE affected device immediately from network",
                "🦠 Run comprehensive antivirus/malware scan (offline if possible)",
                "🚫 Block C2 server IPs/domains at firewall and DNS level",
                "🔍 Perform forensic analysis - identify infection vector",
                "💾 Back up critical data, then reimage system from clean OS",
                "📧 Report to CERT/security team for incident response"
            ],
            "severity": "Critical",
            "attack_stage": "Active Breach - Device Compromised"
        },
        "Malware C2": {  # Alternative name
            "description": "Command & Control communication detected. Compromised device communicating with external attacker server (active breach).",
            "indicators": [
                "🔴 Unusual outbound connections to unknown servers",
                "🔴 Periodic beaconing patterns (every ~60 seconds)",
                "🔴 Encrypted traffic to suspicious domains/IPs",
                "🔴 High entropy data (encrypted payloads)",
                "🔴 Persistent PSH+ACK flags (data transmission)"
            ],
            "mitigation": [
                "⚠️ ISOLATE affected device immediately from network",
                "🦠 Run comprehensive antivirus/malware scan (offline if possible)",
                "🚫 Block C2 server IPs/domains at firewall and DNS level",
                "🔍 Perform forensic analysis - identify infection vector",
                "💾 Back up critical data, then reimage system from clean OS",
                "📧 Report to CERT/security team for incident response"
            ],
            "severity": "Critical",
            "attack_stage": "Active Breach - Device Compromised"
        },
        "Brute_Force": {
            "description": "Brute force authentication attack detected. Attacker attempting to guess credentials through repeated login attempts.",
            "indicators": [
                "🟠 Repeated failed authentication attempts from same source",
                "🟠 Targeting authentication services (SSH:22, RDP:3389, FTP:21)",
                "🟠 High connection termination rate (FIN/RST flags)",
                "🟠 Moderate packet rate (~30 packets/sec)",
                "🟠 Short connection durations (~2 seconds per attempt)"
            ],
            "mitigation": [
                "🔑 Enforce strong password policies (length, complexity, rotation)",
                "🚪 Implement account lockout after N failed attempts",
                "🔐 Enable multi-factor authentication (MFA) on all services",
                "🚨 Deploy fail2ban/sshguard to auto-block brute forcers",
                "📍 Restrict authentication services to specific IP ranges (VPN)",
                "🔒 Consider certificate-based auth instead of passwords"
            ],
            "severity": "High",
            "attack_stage": "Active Attack - Credential Compromise Attempt"
        },
        "Brute Force": {  # Alternative name
            "description": "Brute force authentication attack detected. Attacker attempting to guess credentials through repeated login attempts.",
            "indicators": [
                "🟠 Repeated failed authentication attempts from same source",
                "🟠 Targeting authentication services (SSH:22, RDP:3389, FTP:21)",
                "🟠 High connection termination rate (FIN/RST flags)",
                "🟠 Moderate packet rate (~30 packets/sec)",
                "🟠 Short connection durations (~2 seconds per attempt)"
            ],
            "mitigation": [
                "🔑 Enforce strong password policies (length, complexity, rotation)",
                "🚪 Implement account lockout after N failed attempts",
                "🔐 Enable multi-factor authentication (MFA) on all services",
                "🚨 Deploy fail2ban/sshguard to auto-block brute forcers",
                "📍 Restrict authentication services to specific IP ranges (VPN)",
                "🔒 Consider certificate-based auth instead of passwords"
            ],
            "severity": "High",
            "attack_stage": "Active Attack - Credential Compromise Attempt"
        },
        "SQL_Injection": {
            "description": "SQL injection attack detected. Attacker attempting to manipulate database queries through malicious input.",
            "indicators": [
                "🟠 Abnormally large HTTP request sizes (>800 bytes)",
                "🟠 Suspicious patterns in web traffic (quotes, semicolons, SQL keywords)",
                "🟠 Targeting web application ports (80, 443, 8080)",
                "🟠 Multiple requests to same endpoint with varying payloads",
                "🟠 Unusual character encoding or URL-encoded SQL syntax"
            ],
            "mitigation": [
                "💉 Use parameterized queries/prepared statements (NEVER string concatenation)",
                "🔒 Implement input validation and sanitization on all user inputs",
                "🛡️ Deploy Web Application Firewall (WAF) with SQL injection rules",
                "👤 Apply principle of least privilege for database accounts",
                "🔍 Enable database query logging and monitoring",
                "🔄 Update and patch web application frameworks regularly"
            ],
            "severity": "High",
            "attack_stage": "Active Attack - Database Compromise Attempt"
        },
        "SQL Injection": {  # Alternative name
            "description": "SQL injection attack detected. Attacker attempting to manipulate database queries through malicious input.",
            "indicators": [
                "🟠 Abnormally large HTTP request sizes (>800 bytes)",
                "🟠 Suspicious patterns in web traffic (quotes, semicolons, SQL keywords)",
                "🟠 Targeting web application ports (80, 443, 8080)",
                "🟠 Multiple requests to same endpoint with varying payloads",
                "🟠 Unusual character encoding or URL-encoded SQL syntax"
            ],
            "mitigation": [
                "💉 Use parameterized queries/prepared statements (NEVER string concatenation)",
                "🔒 Implement input validation and sanitization on all user inputs",
                "🛡️ Deploy Web Application Firewall (WAF) with SQL injection rules",
                "👤 Apply principle of least privilege for database accounts",
                "🔍 Enable database query logging and monitoring",
                "🔄 Update and patch web application frameworks regularly"
            ],
            "severity": "High",
            "attack_stage": "Active Attack - Database Compromise Attempt"
        },
        "Normal": {
            "description": "Normal network traffic detected. No malicious activity identified.",
            "indicators": [
                "✅ Standard packet rates within normal range",
                "✅ Complete TCP handshakes (SYN-SYN/ACK-ACK)",
                "✅ Typical packet sizes for service type",
                "✅ Established connections with proper termination",
                "✅ Expected entropy levels for data type"
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
            "description": "Anomalous network behavior detected. Pattern deviates from normal baseline but doesn't match known attack signatures.",
            "indicators": [
                f"⚠️ Model confidence: {confidence:.1%}",
                "⚠️ Multiple feature anomalies detected",
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
