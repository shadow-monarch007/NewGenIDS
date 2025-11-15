"""
Production IDS Dashboard
-----------------------
Real intrusion detection system dashboard with:
- Upload unlabeled traffic CSV files
- Real-time threat prediction
- Threat tracking and statistics
- AI-powered explanations
"""
import os
import sys
import json
import tempfile
import hashlib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import queue
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_traffic, get_severity, extract_key_features, ATTACK_TYPES
from src.threat_db import ThreatDatabase
from src.blockchain_logger import BlockchainLogger
from src.dashboard import generate_ai_explanation  # Reuse explanation function
from src.pcap_converter import PCAPConverter

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'))

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoints', 'best_iot23.pt')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialize threat database
threat_db = ThreatDatabase(os.path.join(DATA_FOLDER, 'threats.json'))

# SSE state
_clients: list[queue.Queue] = []

def _broadcast_alert(alert: dict):
    # Push to all connected clients
    for q in list(_clients):
        try:
            q.put_nowait(alert)
        except Exception:
            pass


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard_new.html')


@app.route('/api/dashboard/stats', methods=['GET'])
def get_stats():
    """Get dashboard statistics."""
    stats = threat_db.get_statistics()
    return jsonify(stats)


@app.route('/api/dashboard/recent_threats', methods=['GET'])
def get_recent_threats():
    """Get recent threat detections."""
    limit = int(request.args.get('limit', 10))
    threats = threat_db.get_recent_threats(limit)
    return jsonify(threats)


@app.route('/api/dashboard/timeline', methods=['GET'])
def get_timeline():
    """Get threat timeline data."""
    days = int(request.args.get('days', 7))
    timeline = threat_db.get_timeline_data(days)
    return jsonify(timeline)


@app.route('/events')
def sse_events():
    """Server-Sent Events stream for live alerts."""
    def gen():
        q = queue.Queue()
        _clients.append(q)
        try:
            while True:
                alert = q.get()
                yield f"data: {json.dumps(alert)}\n\n"
        except GeneratorExit:
            pass
        finally:
            try:
                _clients.remove(q)
            except ValueError:
                pass

    return Response(stream_with_context(gen()), mimetype='text/event-stream')


@app.route('/api/analyze_traffic', methods=['POST'])
def analyze_traffic():
    """
    Analyze uploaded traffic file (CSV or PCAP).
    This is the REAL IDS - predicts threats from unlabeled data!
    """
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    # Check file type
    filename_lower = file.filename.lower()
    is_pcap = filename_lower.endswith(('.pcap', '.pcapng'))
    is_csv = filename_lower.endswith('.csv')
    
    if not (is_pcap or is_csv):
        return jsonify({
            "success": False, 
            "error": "Invalid file. Please upload CSV or PCAP file"
        }), 400
    
    try:
        # Save uploaded file temporarily
        suffix = '.pcap' if is_pcap else '.csv'
        with tempfile.NamedTemporaryFile(mode='w+b', suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Convert PCAP to CSV if needed
        csv_path = tmp_path
        if is_pcap:
            print(f"Converting PCAP file: {file.filename}")
            converter = PCAPConverter()
            csv_path = tmp_path.replace('.pcap', '_features.csv')
            try:
                converter.convert_pcap_to_csv(tmp_path, csv_path, window_size=5.0, max_packets=10000)
            except Exception as e:
                os.unlink(tmp_path)
                return jsonify({
                    "success": False,
                    "error": f"PCAP conversion failed: {str(e)}"
                }), 400
        
        # Check if model exists
        if not os.path.exists(CHECKPOINT_PATH):
            os.unlink(tmp_path)
            if is_pcap and os.path.exists(csv_path):
                os.unlink(csv_path)
            return jsonify({
                "success": False, 
                "error": f"Model not found at {CHECKPOINT_PATH}. Please train the model first."
            }), 400
        
        # Run prediction
        predictions = predict_traffic(
            csv_path=csv_path,
            checkpoint_path=CHECKPOINT_PATH,
            dataset_name='iot23',
            device='cpu',
            seq_len=100
        )
        
        # Clean up temp files
        os.unlink(tmp_path)
        if is_pcap and os.path.exists(csv_path):
            os.unlink(csv_path)
        
        if not predictions:
            return jsonify({
                "success": False,
                "error": "No predictions generated. File may be too small or invalid."
            }), 400
        
        # Take the most recent/relevant prediction (usually first or last sequence)
        main_prediction = predictions[-1]  # Last sequence usually most representative
        
        # Add threats to database
        for pred in predictions:
            threat_db.add_threat({
                "attack_type": pred["attack_type"],
                "severity": pred["severity"],
                "confidence": pred["confidence"],
                "features": pred["features"],
                "timestamp": pred["timestamp"],
                "status": "Active" if pred["attack_type"] != "Normal" else "False Positive",
                "source": "File Upload",
                "filename": file.filename
            })
        
        # Generate AI explanation
        explanation = generate_ai_explanation(
            main_prediction["attack_type"],
            main_prediction["features"],
            main_prediction["confidence"]
        )
        
        return jsonify({
            "success": True,
            "prediction": {
                "attack_type": main_prediction["attack_type"],
                "confidence": main_prediction["confidence"],
                "severity": main_prediction["severity"],
                "indicators": explanation.get("indicators", []),
                "mitigation_steps": explanation.get("mitigation_steps", []),
                "description": explanation.get("description", ""),
                "attack_stage": explanation.get("attack_stage", ""),
                "recommended_priority": explanation.get("recommended_priority", "MEDIUM")
            },
            "sequences_analyzed": len(predictions),
            "threats_detected": sum(1 for p in predictions if p["attack_type"] != "Normal")
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Analysis failed: {str(e)}"
        }), 500


@app.route('/api/dashboard/ingest', methods=['POST'])
def ingest_alert():
    """Receive alert JSON from realtime agent; store and broadcast."""
    try:
        alert = request.get_json(force=True)
        if not isinstance(alert, dict):
            return jsonify({"error": "Invalid alert payload"}), 400

        # Normalize and store
        threat_db.add_threat({
            "attack_type": alert.get("attack_type", "Unknown"),
            "severity": alert.get("severity", "Unknown"),
            "confidence": float(alert.get("confidence", 0.0)),
            "features": alert.get("features", {}),
            "timestamp": alert.get("timestamp") or datetime.now().isoformat(),
            "status": "Active" if alert.get("attack_type") != "Normal" else "False Positive",
            "source": "Realtime Capture"
        })

        # Broadcast via SSE
        _broadcast_alert(alert)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/blockchain/verify', methods=['GET'])
def verify_blockchain():
    """Verify integrity of alerts blockchain and return summary."""
    chain_path = os.path.join(RESULTS_FOLDER, 'alerts_chain.json')
    if not os.path.exists(chain_path):
        return jsonify({"error": "alerts_chain.json not found", "exists": False}), 404
    try:
        logger = BlockchainLogger(chain_path)
        with open(chain_path, 'r') as f:
            chain = json.load(f)
        valid = logger.verify_chain()
        length = len(chain)
        last = chain[-1] if chain else {}
        return jsonify({
            "valid": valid,
            "length": length,
            "last_hash": last.get('hash'),
            "last_index": last.get('index'),
            "exists": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/threat/<int:threat_id>', methods=['GET'])
def get_threat_details(threat_id):
    """Get detailed information about a specific threat."""
    threats = threat_db.get_recent_threats(1000)  # Get all
    threat = next((t for t in threats if t.get('id') == threat_id), None)
    
    if not threat:
        return jsonify({"error": "Threat not found"}), 404
    
    # Generate explanation if not present
    if 'explanation' not in threat:
        explanation = generate_ai_explanation(
            threat.get('attack_type', 'Unknown'),
            threat.get('features', {}),
            threat.get('confidence', 0.5)
        )
        threat['explanation'] = explanation
    
    return jsonify(threat)


@app.route('/api/threat/<int:threat_id>/update_status', methods=['POST'])
def update_threat_status(threat_id):
    """Update threat status (Active, Investigating, Remediated, False_positive)."""
    data = request.json
    new_status = data.get('status')
    
    if not new_status:
        return jsonify({"error": "Status not provided"}), 400
    
    success = threat_db.update_status(threat_id, new_status)
    
    if success:
        return jsonify({"success": True, "message": f"Threat {threat_id} status updated to {new_status}"})
    else:
        return jsonify({"error": "Threat not found"}), 404


@app.route('/api/dashboard/clear_threats', methods=['POST'])
def clear_threats():
    """Clear all threats (for testing/demo purposes)."""
    threat_db.clear_all()
    return jsonify({"success": True, "message": "All threats cleared"})


if __name__ == '__main__':
    print("=" * 70)
    print("🛡️  Next-Gen IDS Dashboard - Production Mode")
    print("=" * 70)
    print(f"📊 Dashboard URL: http://localhost:5000")
    print(f"🔍 Model: {CHECKPOINT_PATH}")
    print(f"💾 Threat DB: {threat_db.db_path}")
    print("=" * 70)
    print("\n✨ Upload any network traffic CSV file to detect threats!")
    print("   The system will analyze traffic patterns WITHOUT needing labels.\n")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
