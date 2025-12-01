"""
Unified Next-Gen IDS Dashboard
------------------------------
Combines training/evaluation, realtime ingest, file/pcap analysis, phishing &
log triage, and remediation proposal endpoints.

Run:
    python -m src.dashboard_unified --port 5060

Templates:
    Reuses existing `templates/dashboard_new.html` for main UI; you can update
    it to surface new endpoints (phishing scan, log upload, remediation queue).

Minimal HTML changes are not included here; focus is on backend API merge.
"""
from __future__ import annotations
import os
import sys
import io
import json
import base64
import tempfile
import queue
from datetime import datetime, timedelta
from typing import Any, Dict, List

from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_file
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Path adjustments for imports if executed as module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.model import IDSModel, NextGenIDS
from src.data_loader import create_dataloaders
from src.utils import compute_metrics
from src.predict import predict_traffic
from src.threat_db import ThreatDatabase
from src.blockchain_logger import BlockchainLogger
from src.pcap_converter import PCAPConverter
from src.explanation import generate_ai_explanation  # standalone explanation module
from src.phishing_detector import classify_url, classify_email
from src.log_ingest import ingest_lines
from src.remediation import propose_action, list_actions, update_action, SAFE_ACTION_TYPES, VALID_STATUSES

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
DATA_DIR = os.path.join(BASE_DIR, 'data')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Select best available checkpoint (prefer multi-class for better attack classification)
_multiclass_ckpt = os.path.join(CHECKPOINT_DIR, 'best_multiclass.pt')
_iot23_ckpt = os.path.join(CHECKPOINT_DIR, 'best_iot23.pt')
if os.path.exists(_multiclass_ckpt):
    CHECKPOINT_PATH = _multiclass_ckpt
    DATASET_NAME = 'multiclass'
else:
    CHECKPOINT_PATH = _iot23_ckpt
    DATASET_NAME = 'iot23'

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# Threat DB and SSE state
threat_db = ThreatDatabase(os.path.join(DATA_DIR, 'threats.json'))
# Optional fresh start (ignore persisted threats) if env var set
if os.environ.get('IDS_FRESH_START', '0') in ('1', 'true', 'True'):
    threat_db.clear_all()
_clients: List[queue.Queue] = []
current_job: Dict[str, Any] = {"status": "idle", "progress": 0, "message": "Ready"}

# Disable browser caching so template/JS updates are visible immediately
@app.after_request
def _add_no_cache_headers(resp):
    try:
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
    except Exception:
        pass
    return resp

# ---------------------- Utility ----------------------

def _select_best_checkpoint(num_features: int) -> tuple[str, str]:
    """
    Automatically select the best matching checkpoint based on number of features.
    Returns (checkpoint_path, dataset_name) tuple.
    """
    checkpoints = []
    
    # Scan available checkpoints and get their metadata
    for ckpt_file in os.listdir(CHECKPOINT_DIR):
        if ckpt_file.endswith('.pt'):
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                meta = ckpt.get('meta', {})
                input_dim = meta.get('input_dim', 0)
                num_classes = meta.get('num_classes', 2)
                f1 = meta.get('f1', 0.0)
                
                # Extract dataset name from filename (e.g., best_iot23.pt -> iot23)
                dataset_name = ckpt_file.replace('best_', '').replace('.pt', '')
                
                checkpoints.append({
                    'file': ckpt_file,
                    'path': ckpt_path,
                    'dataset': dataset_name,
                    'input_dim': input_dim,
                    'num_classes': num_classes,
                    'f1': f1,
                    'dim_diff': abs(input_dim - num_features)
                })
            except Exception as e:
                print(f"Could not load checkpoint {ckpt_file}: {e}")
                continue
    
    if not checkpoints:
        raise ValueError("No valid checkpoints found")
    
    # Sort by: 1) smallest dimension difference, 2) highest F1 score
    checkpoints.sort(key=lambda x: (x['dim_diff'], -x['f1']))
    
    best = checkpoints[0]
    print(f"🎯 Auto-selected checkpoint: {best['file']} (input_dim={best['input_dim']}, "
          f"data_features={num_features}, f1={best['f1']:.4f})")
    
    return best['path'], best['dataset']


def _broadcast_alert(alert: dict):
    for q in list(_clients):
        try:
            q.put_nowait(alert)
        except Exception:
            pass

# ---------------------- Main Page --------------------

@app.route('/')
def index():
    # Add timestamp to force browser to reload template
    import time
    return render_template('dashboard.html', v=int(time.time()))

# ---------------------- Training / Evaluation --------

@app.route('/api/train', methods=['POST'])
def api_train():
    """
    Train a model on either a built-in dataset (e.g. 'iot23') or a user-uploaded
    CSV provided as multipart/form-data under the field name 'file'. When a file
    is uploaded, it is stored under `uploads/uploaded/` and the dataset name is
    forced to 'uploaded' so that the data loader will read all CSVs in that
    folder.
    """
    global current_job

    try:
        # Support both JSON and multipart/form-data
        data = request.form.to_dict() if request.form else (request.json or {})
        dataset_name = data.get('dataset_name', 'uploaded')
        epochs = int(data.get('epochs', 5))
        batch_size = int(data.get('batch_size', 32))
        seq_len = int(data.get('seq_len', 64))
        use_arnn = str(data.get('use_arnn', 'False')).lower() in ('1', 'true', 'yes', 'on')

        # Handle optional uploaded CSV
        try:
            if 'file' in request.files and request.files['file'].filename:
                file = request.files['file']
                # Ensure target folder exists
                uploaded_dir = os.path.join(UPLOAD_DIR, 'uploaded')
                os.makedirs(uploaded_dir, exist_ok=True)
                safe_name = os.path.basename(file.filename)
                save_path = os.path.join(uploaded_dir, safe_name)
                file.save(save_path)
                dataset_name = 'uploaded'
                print(f"Uploaded file saved to: {save_path}")
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to save uploaded file: {e}'}), 400

        # Build dataloaders
        train_loader, val_loader, test_loader, input_dim, num_classes = create_dataloaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            seq_len=seq_len,
            data_dir=UPLOAD_DIR,
            num_workers=0
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = NextGenIDS(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(device) if use_arnn else IDSModel(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        current_job = {"status": "training", "progress": 0, "message": "Starting"}
        history = []
        best_f1 = 0.0

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            total_samples = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                sz = int(X.size(0))
                total_loss += loss.item() * sz
                total_samples += sz
            avg_loss = total_loss / max(1, total_samples)

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
            f1 = metrics['f1']
            history.append({
                'epoch': epoch + 1,
                'train_loss': float(avg_loss),
                'val_f1': float(f1),
                'val_accuracy': float(metrics['accuracy'])
            })
            if f1 > best_f1:
                best_f1 = f1
                torch.save({
                    'state_dict': model.state_dict(),
                    'meta': {'input_dim': input_dim, 'num_classes': num_classes, 'f1': f1, 'epoch': epoch + 1}
                }, os.path.join(CHECKPOINT_DIR, f'best_{dataset_name}.pt'))

            current_job['progress'] = int(((epoch + 1) / epochs) * 100)
            current_job['message'] = f'Epoch {epoch + 1}/{epochs} F1={f1:.4f}'

        current_job = {"status": "completed", "progress": 100, "message": "Training complete"}
        return jsonify({
            'success': True,
            'history': history,
            'best_f1': float(best_f1),
            'checkpoint': f'best_{dataset_name}.pt'
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Training error: {error_details}")
        current_job = {"status": "error", "progress": 0, "message": str(e)}
        return jsonify({'success': False, 'error': f'Training failed: {str(e)}'}), 500

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """
    Evaluate a checkpoint on either a built-in dataset or an uploaded CSV.
    Accepts multipart/form-data with 'file' and 'checkpoint' or JSON body.
    Uploaded files are saved under uploads/uploaded/ and dataset_name is set to
    'uploaded' so the loader will pick them up.
    
    Now with AUTO-SELECTION: if a file is uploaded, automatically picks the best
    matching checkpoint based on feature dimensions.
    """
    try:
        # Support both JSON and multipart/form-data
        data = request.form.to_dict() if request.form else (request.json or {})
        dataset_name = data.get('dataset_name', 'uploaded')
        checkpoint_name = data.get('checkpoint', f'best_{dataset_name}.pt')
        batch_size = int(data.get('batch_size', 32))
        seq_len = int(data.get('seq_len', 64))
        auto_select_checkpoint = False

        # Optional uploaded CSV for evaluation
        try:
            if 'file' in request.files and request.files['file'].filename:
                file = request.files['file']
                uploaded_dir = os.path.join(UPLOAD_DIR, 'uploaded')
                os.makedirs(uploaded_dir, exist_ok=True)
                safe_name = os.path.basename(file.filename)
                save_path = os.path.join(uploaded_dir, safe_name)
                file.save(save_path)
                dataset_name = 'uploaded'
                auto_select_checkpoint = True  # Enable auto-selection for uploaded files
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to save uploaded file: {e}'}), 400

        # Load data to determine feature dimensions
        _, _, test_loader, data_input_dim, data_num_classes = create_dataloaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            seq_len=seq_len,
            data_dir=UPLOAD_DIR,
            num_workers=0
        )
        
        # Auto-select best matching checkpoint if file was uploaded
        if auto_select_checkpoint:
            try:
                ckpt_path, matched_dataset = _select_best_checkpoint(data_input_dim)
                checkpoint_name = os.path.basename(ckpt_path)
                print(f"✨ Auto-selected {checkpoint_name} for {data_input_dim} features")
            except Exception as e:
                print(f"Auto-selection failed: {e}, falling back to manual selection")
                ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        else:
            # Manual checkpoint selection
            ckpt_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        
        if not os.path.exists(ckpt_path):
            return jsonify({'success': False, 'error': f'Checkpoint not found: {checkpoint_name}'}), 400

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(ckpt_path, map_location=device)
        
        # Use checkpoint metadata for model dimensions (not data dimensions)
        meta = ckpt.get('meta', {})
        input_dim = meta.get('input_dim', data_input_dim)
        num_classes = meta.get('num_classes', data_num_classes)
        
        state_dict_keys = list(ckpt['state_dict'].keys())
        uses_arnn = any('arnn' in k or 'slstm_cnn' in k for k in state_dict_keys)
        model = NextGenIDS(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(device) if uses_arnn else IDSModel(input_size=input_dim, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        test_preds, test_labels = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                test_preds.extend(logits.argmax(1).cpu().numpy())
                test_labels.extend(y.cpu().numpy())

        metrics = compute_metrics(test_labels, test_preds)

        # Confusion matrix plot
        from sklearn.metrics import confusion_matrix as cm_func
        cm = cm_func(test_labels, test_preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=90, bbox_inches='tight')
        buf.seek(0)
        cm_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return jsonify({'success': True, 'metrics': metrics, 'confusion_matrix': cm_b64})
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Evaluation error: {error_details}")
        return jsonify({'success': False, 'error': f'Evaluation failed: {str(e)}'}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify(current_job)

# ---------------------- File / PCAP Analysis ----------

@app.route('/api/analyze_traffic', methods=['POST'])
def api_analyze_traffic():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    name_lower = file.filename.lower()
    is_pcap = name_lower.endswith(('.pcap', '.pcapng'))
    is_csv = name_lower.endswith('.csv')
    if not (is_pcap or is_csv):
        return jsonify({'success': False, 'error': 'Upload CSV or PCAP only'}), 400

    try:
        suffix = '.pcap' if is_pcap else '.csv'
        with tempfile.NamedTemporaryFile(mode='w+b', suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        csv_path = tmp_path
        if is_pcap:
            converter = PCAPConverter()
            csv_path = tmp_path.replace('.pcap', '_features.csv')
            converter.convert_pcap_to_csv(tmp_path, csv_path, window_size=5.0, max_packets=10000)

        if not os.path.exists(CHECKPOINT_PATH):
            return jsonify({'success': False, 'error': f'Model checkpoint missing: {CHECKPOINT_PATH}'}), 400

        preds = predict_traffic(csv_path=csv_path, checkpoint_path=CHECKPOINT_PATH, dataset_name=DATASET_NAME, device='cpu', seq_len=100)

        # Cleanup
        try:
            os.unlink(tmp_path)
            if is_pcap and os.path.exists(csv_path):
                os.unlink(csv_path)
        except Exception:
            pass

        if not preds:
            return jsonify({'success': False, 'error': 'No sequences predicted'}), 400

        main_pred = preds[-1]
        explanation = generate_ai_explanation(main_pred['attack_type'], main_pred['features'], main_pred['confidence'])

        # Store threats and broadcast (Bulk Insert Optimization)
        new_threats = []
        for p in preds:
            new_threats.append({
                'attack_type': p['attack_type'],
                'severity': p['severity'],
                'confidence': p['confidence'],
                'features': p['features'],
                'timestamp': p['timestamp'],
                'status': 'Active' if p['attack_type'] != 'Normal' else 'False Positive',
                'source': 'File Upload',
                'filename': file.filename
            })
        
        # Use bulk insert to avoid saving JSON file 22,000 times
        threat_db.add_threats_bulk(new_threats)
        
        # Only broadcast the last few to avoid flooding the frontend
        for p in preds[-5:]:
            try:
                _broadcast_alert({'attack_type': p['attack_type'], 'severity': p['severity'], 'confidence': p['confidence'], 'timestamp': p['timestamp']})
            except Exception:
                pass

        return jsonify({'success': True, 'prediction': main_pred, 'explanation': explanation, 'sequences': len(preds)})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Analysis failed: {e}'}), 500

# ---------------------- Realtime ingest / SSE ---------

@app.route('/api/dashboard/ingest', methods=['POST'])
def api_ingest():
    try:
        alert = request.get_json(force=True)
        if not isinstance(alert, dict):
            return jsonify({'error': 'Invalid alert payload'}), 400
        threat_db.add_threat({
            'attack_type': alert.get('attack_type', 'Unknown'),
            'severity': alert.get('severity', 'Unknown'),
            'confidence': float(alert.get('confidence', 0.0)),
            'features': alert.get('features', {}),
            'timestamp': alert.get('timestamp') or datetime.now().isoformat(),
            'status': 'Active' if alert.get('attack_type') != 'Normal' else 'False Positive',
            'source': 'Realtime Capture'
        })
        _broadcast_alert(alert)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/events')
def api_events():
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

# ---------------------- Phishing APIs -----------------

@app.route('/api/phishing/url', methods=['POST'])
def api_phishing_url():
    data = request.json or {}
    url = data.get('url')
    if not url:
        return jsonify({'error': 'url missing'}), 400
    result = classify_url(url)
    # Flatten keys for frontend simplicity
    flat = {
        'success': True,
        'category': result.get('category'),
        'is_phishing': result.get('is_phishing'),
        'risk_score': result.get('score'),
        'severity': result.get('severity'),
        'indicators': result.get('indicators', []),
        'recommended_steps': result.get('recommended_steps', [])
    }
    return jsonify(flat)

@app.route('/api/phishing/email', methods=['POST'])
def api_phishing_email():
    data = request.json or {}
    content = data.get('content')
    if not content:
        return jsonify({'error': 'content missing'}), 400
    result = classify_email(content)
    flat = {
        'success': True,
        'category': result.get('category'),
        'is_phishing': result.get('is_phishing'),
        'risk_score': result.get('score'),
        'severity': result.get('severity'),
        'indicators': result.get('indicators', []),
        'recommended_steps': result.get('recommended_steps', [])
    }
    return jsonify(flat)

# ---------------------- Log ingestion -----------------

@app.route('/api/logs/ingest', methods=['POST'])
def api_logs_ingest():
    data = request.json or {}
    lines = data.get('lines', [])
    source = data.get('source', 'syslog')
    if not isinstance(lines, list) or not lines:
        return jsonify({'error': 'lines must be non-empty list'}), 400
    events = ingest_lines(lines, source)
    # Convert certain high severity events to threats automatically
    for evt in events:
        if evt['severity'] in ('High', 'Critical'):
            threat_db.add_threat({
                'attack_type': evt['type'],
                'severity': evt['severity'],
                'confidence': min(0.99, (evt['score'] / 100)),
                'features': {'indicators': evt['indicators']},
                'timestamp': evt['timestamp'],
                'status': 'Active',
                'source': f'Log:{source}'
            })
            _broadcast_alert({'attack_type': evt['type'], 'severity': evt['severity'], 'timestamp': evt['timestamp']})
    return jsonify({'success': True, 'events': events, 'threats_added': sum(1 for e in events if e['severity'] in ('High', 'Critical'))})

# ---------------------- Threat / Stats APIs -----------

@app.route('/api/dashboard/stats', methods=['GET'])
def api_stats():
    return jsonify(threat_db.get_statistics())

@app.route('/api/dashboard/recent_threats', methods=['GET'])
def api_recent():
    limit = int(request.args.get('limit', 10))
    return jsonify(threat_db.get_recent_threats(limit))

@app.route('/api/threat/<int:threat_id>', methods=['GET'])
def api_threat_detail(threat_id: int):
    threats = threat_db.get_recent_threats(1000)
    threat = next((t for t in threats if t.get('id') == threat_id), None)
    if not threat:
        return jsonify({'error': 'Threat not found'}), 404
    if 'explanation' not in threat:
        threat['explanation'] = generate_ai_explanation(threat.get('attack_type', 'Unknown'), threat.get('features', {}), threat.get('confidence', 0.5))
    return jsonify(threat)

@app.route('/api/threat/<int:threat_id>/update_status', methods=['POST'])
def api_threat_update(threat_id: int):
    data = request.json or {}
    new_status = data.get('status')
    if not new_status:
        return jsonify({'error': 'status missing'}), 400
    ok = threat_db.update_status(threat_id, new_status)
    if ok:
        return jsonify({'success': True})
    return jsonify({'error': 'Threat not found'}), 404

@app.route('/api/dashboard/clear_threats', methods=['POST'])
def api_clear_threats():
    threat_db.clear_all()
    return jsonify({'success': True})

@app.route('/api/dashboard/timeline', methods=['GET'])
def api_dashboard_timeline():
    """Return aggregated threat counts by severity for the last N days.
    Frontend uses this for the timeline chart. Days default = 7.
    """
    days = int(request.args.get('days', 7))
    if days < 1 or days > 30:
        days = 7
    now = datetime.utcnow().date()
    # Prepare buckets newest last for chart labels chronological
    buckets = { (now - timedelta(days=i)): {'Critical':0,'High':0,'Medium':0,'Low':0} for i in range(days) }
    threats = threat_db.get_recent_threats(5000)
    for t in threats:
        ts = t.get('timestamp')
        if not ts:
            continue
        try:
            # Accept both ISO with Z and without
            ts_clean = ts.replace('Z','')
            dt = datetime.fromisoformat(ts_clean)
            dkey = dt.date()
        except Exception:
            continue
        if dkey in buckets:
            sev = (t.get('severity') or 'Low')
            if sev not in buckets[dkey]:
                # Normalize unexpected severities to Low
                sev = 'Low'
            buckets[dkey][sev] += 1
    # Build ordered list oldest->newest for line chart x-axis
    ordered_days = sorted(buckets.keys())
    payload = []
    for d in ordered_days:
        sev_counts = buckets[d]
        payload.append({
            'date': d.isoformat(),
            'critical': sev_counts['Critical'],
            'high': sev_counts['High'],
            'medium': sev_counts['Medium'],
            'low': sev_counts['Low']
        })
    return jsonify(payload)

# ---------------------- Blockchain Verify -------------

@app.route('/api/blockchain/verify', methods=['GET'])
def api_blockchain_verify():
    chain_path = os.path.join(RESULTS_DIR, 'alerts_chain.json')
    if not os.path.exists(chain_path):
        return jsonify({'error': 'alerts_chain.json not found', 'exists': False}), 404
    try:
        logger = BlockchainLogger(chain_path)
        with open(chain_path, 'r') as f:
            chain = json.load(f)
        valid = logger.verify_chain()
        last = chain[-1] if chain else {}
        return jsonify({'valid': valid, 'length': len(chain), 'last_hash': last.get('hash'), 'exists': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------------------- Remediation Actions -----------

@app.route('/api/remediation/propose', methods=['POST'])
def api_remediation_propose():
    data = request.json or {}
    action_type = data.get('type')
    details = data.get('details', {})
    if not action_type:
        return jsonify({'error': 'type missing'}), 400
    try:
        record = propose_action(action_type, details)
        return jsonify({'success': True, 'action': record})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/remediation/list', methods=['GET'])
def api_remediation_list():
    # Return raw list for simpler frontend handling
    return jsonify(list_actions())

@app.route('/api/remediation/update/<int:action_id>', methods=['POST'])
def api_remediation_update(action_id: int):
    data = request.json or {}
    status = data.get('status')
    if not status:
        return jsonify({'error': 'status missing'}), 400
    try:
        ok = update_action(action_id, status)
        if ok:
            return jsonify({'success': True})
        return jsonify({'error': 'action not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/remediation/meta', methods=['GET'])
def api_remediation_meta():
    return jsonify({'success': True, 'valid_statuses': VALID_STATUSES, 'action_types': SAFE_ACTION_TYPES})

# ---------------------- Dataset Converter --------------------------

@app.route('/api/convert_dataset', methods=['POST'])
def api_convert_dataset():
    """
    Convert uploaded dataset to IoT-23 format
    Accepts: CSV file + optional format hint
    Returns: Converted CSV file for download
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files supported'}), 400
        
        # Get optional parameters
        max_rows = request.form.get('max_rows', None)
        if max_rows:
            max_rows = int(max_rows)
        
        # Save uploaded file temporarily
        upload_dir = os.path.join(BASE_DIR, 'uploads', 'temp')
        os.makedirs(upload_dir, exist_ok=True)
        input_path = os.path.join(upload_dir, f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        file.save(input_path)
        
        # Output path
        output_filename = f"converted_{file.filename}"
        output_path = os.path.join(upload_dir, output_filename)
        
        # Load and detect format
        try:
            df = pd.read_csv(input_path, low_memory=False)
        except:
            df = pd.read_csv(input_path, header=None, low_memory=False)
        
        original_rows = len(df)
        original_cols = len(df.columns)
        
        # Limit rows if specified
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
        
        # Detect format
        dataset_type = detect_dataset_format(df)
        
        # Convert based on type
        if dataset_type == 'iot23':
            iot23_df = df
        elif dataset_type == 'kdd':
            iot23_df = convert_kdd_to_iot23(df)
        elif dataset_type == 'cicids':
            iot23_df = convert_cicids_to_iot23(df)
        elif dataset_type == 'unsw':
            iot23_df = convert_unsw_to_iot23(df)
        else:
            iot23_df = convert_generic_to_iot23(df)
        
        # Ensure all 20 features exist
        IOT23_FEATURES = [
            'packet_rate', 'packet_size', 'byte_rate', 'flow_duration',
            'total_packets', 'total_bytes', 'entropy', 'port_scan_score',
            'syn_flag_count', 'ack_flag_count', 'fin_flag_count', 'rst_flag_count',
            'psh_flag_count', 'urg_flag_count', 'unique_src_ports', 'unique_dst_ports',
            'payload_entropy', 'dns_query_count', 'http_request_count', 'ssl_handshake_count'
        ]
        
        for feat in IOT23_FEATURES:
            if feat not in iot23_df.columns:
                iot23_df[feat] = 0
        
        iot23_df = iot23_df[IOT23_FEATURES]
        iot23_df = iot23_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Save converted file
        iot23_df.to_csv(output_path, index=False)
        
        # Return file for download
        return send_file(
            output_path,
            as_attachment=True,
            download_name=output_filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Conversion failed: {str(e)}'}), 500


def detect_dataset_format(df):
    """Auto-detect dataset type based on column names/structure"""
    cols = [str(c).lower() for c in df.columns]
    
    if len(df.columns) in [41, 42, 43] and all(isinstance(c, int) for c in df.columns[:10]):
        return 'kdd'
    if any('flow' in c and 'duration' in c for c in cols):
        return 'cicids'
    if 'sbytes' in cols or 'dbytes' in cols:
        return 'unsw'
    if all(f in cols for f in ['packet_rate', 'byte_rate', 'flow_duration']):
        return 'iot23'
    return 'generic'


def convert_kdd_to_iot23(df):
    """Convert KDD/NSL-KDD dataset to IoT-23 format"""
    iot23 = pd.DataFrame()
    duration = df.iloc[:, 0].clip(lower=0.01)
    src_bytes = df.iloc[:, 4]
    dst_bytes = df.iloc[:, 5]
    total_bytes_val = src_bytes + dst_bytes
    total_packets_val = (total_bytes_val / 1500).clip(lower=1).astype(int)
    
    iot23['packet_rate'] = (total_packets_val / duration).clip(upper=10000)
    iot23['packet_size'] = (total_bytes_val / total_packets_val).clip(upper=65535)
    iot23['byte_rate'] = (total_bytes_val / duration).clip(upper=1e9)
    iot23['flow_duration'] = duration
    iot23['total_packets'] = total_packets_val
    iot23['total_bytes'] = total_bytes_val
    iot23['entropy'] = np.random.uniform(2.0, 7.0, len(df))
    iot23['port_scan_score'] = df.iloc[:, 22].clip(upper=100) if df.shape[1] > 22 else 0
    
    flag_col = df.iloc[:, 3].astype(str) if df.shape[1] > 3 else pd.Series(['SF'] * len(df))
    iot23['syn_flag_count'] = (flag_col.str.contains('S0|S1|S2|S3', case=False, na=False)).astype(int) * 5
    iot23['ack_flag_count'] = (flag_col == 'SF').astype(int) * 5
    iot23['fin_flag_count'] = (flag_col.str.contains('SF|S1', case=False, na=False)).astype(int) * 3
    iot23['rst_flag_count'] = (flag_col.str.contains('REJ|RSTO|RSTR', case=False, na=False)).astype(int) * 5
    iot23['psh_flag_count'] = (flag_col == 'SF').astype(int) * 2
    iot23['urg_flag_count'] = (df.iloc[:, 8] > 0).astype(int) if df.shape[1] > 8 else 0
    
    iot23['unique_src_ports'] = 1
    iot23['unique_dst_ports'] = 1
    iot23['payload_entropy'] = np.random.uniform(3.0, 6.0, len(df))
    
    service_col = df.iloc[:, 2].astype(str) if df.shape[1] > 2 else pd.Series(['other'] * len(df))
    iot23['dns_query_count'] = (service_col.str.contains('domain', case=False, na=False)).astype(int) * 10
    iot23['http_request_count'] = (service_col.str.contains('http', case=False, na=False)).astype(int) * 5
    iot23['ssl_handshake_count'] = (service_col.str.contains('ssl|https', case=False, na=False)).astype(int) * 3
    
    return iot23.fillna(0)


def convert_cicids_to_iot23(df):
    """Convert CICIDS2017 dataset to IoT-23 format"""
    iot23 = pd.DataFrame()
    cols = {c.lower(): c for c in df.columns}
    
    def get_col(patterns):
        for pattern in patterns:
            for key, orig in cols.items():
                if pattern in key:
                    return df[orig]
        return pd.Series([0] * len(df))
    
    duration = get_col(['flow duration', 'duration']).clip(lower=0.001) / 1e6
    fwd_bytes = get_col(['total fwd packet', 'fwd packet length total', 'totlen_fwd'])
    bwd_bytes = get_col(['total bwd packet', 'bwd packet length total', 'totlen_bwd'])
    total_bytes_val = (fwd_bytes + bwd_bytes).clip(lower=1)
    fwd_packets = get_col(['total fwd packet', 'fwd packets', 'tot_fwd'])
    bwd_packets = get_col(['total bwd packet', 'bwd packets', 'tot_bwd'])
    total_packets_val = (fwd_packets + bwd_packets).clip(lower=1)
    
    iot23['packet_rate'] = (total_packets_val / duration).clip(upper=10000)
    iot23['packet_size'] = (total_bytes_val / total_packets_val).clip(upper=65535)
    iot23['byte_rate'] = (total_bytes_val / duration).clip(upper=1e9)
    iot23['flow_duration'] = duration
    iot23['total_packets'] = total_packets_val
    iot23['total_bytes'] = total_bytes_val
    iot23['entropy'] = get_col(['entropy']).fillna(np.random.uniform(2.0, 7.0, len(df)))
    iot23['port_scan_score'] = get_col(['destination port']).clip(upper=100)
    iot23['syn_flag_count'] = get_col(['syn flag count', 'syn_flag']).fillna(0)
    iot23['ack_flag_count'] = get_col(['ack flag count', 'ack_flag']).fillna(0)
    iot23['fin_flag_count'] = get_col(['fin flag count', 'fin_flag']).fillna(0)
    iot23['rst_flag_count'] = get_col(['rst flag count', 'rst_flag']).fillna(0)
    iot23['psh_flag_count'] = get_col(['psh flag count', 'psh_flag']).fillna(0)
    iot23['urg_flag_count'] = get_col(['urg flag count', 'urg_flag']).fillna(0)
    iot23['unique_src_ports'] = get_col(['source port']).fillna(1)
    iot23['unique_dst_ports'] = get_col(['destination port']).fillna(1)
    iot23['payload_entropy'] = np.random.uniform(3.0, 6.0, len(df))
    
    protocol = get_col(['protocol'])
    iot23['dns_query_count'] = (protocol == 17).astype(int) * 5
    iot23['http_request_count'] = get_col(['destination port']).isin([80, 8080]).astype(int) * 5
    iot23['ssl_handshake_count'] = get_col(['destination port']).isin([443, 8443]).astype(int) * 3
    
    return iot23.fillna(0)


def convert_unsw_to_iot23(df):
    """Convert UNSW-NB15 dataset to IoT-23 format"""
    iot23 = pd.DataFrame()
    duration = df.get('dur', pd.Series([1.0] * len(df))).clip(lower=0.001)
    src_bytes = df.get('sbytes', pd.Series([0] * len(df)))
    dst_bytes = df.get('dbytes', pd.Series([0] * len(df)))
    total_bytes_val = (src_bytes + dst_bytes).clip(lower=1)
    src_packets = df.get('spkts', pd.Series([1] * len(df)))
    dst_packets = df.get('dpkts', pd.Series([1] * len(df)))
    total_packets_val = (src_packets + dst_packets).clip(lower=1)
    
    iot23['packet_rate'] = (total_packets_val / duration).clip(upper=10000)
    iot23['packet_size'] = (total_bytes_val / total_packets_val).clip(upper=65535)
    iot23['byte_rate'] = (total_bytes_val / duration).clip(upper=1e9)
    iot23['flow_duration'] = duration
    iot23['total_packets'] = total_packets_val
    iot23['total_bytes'] = total_bytes_val
    iot23['entropy'] = np.random.uniform(2.0, 7.0, len(df))
    iot23['port_scan_score'] = df.get('dport', 0).clip(upper=100)
    iot23['syn_flag_count'] = df.get('swin', 0).apply(lambda x: 5 if x > 0 else 0)
    iot23['ack_flag_count'] = df.get('dwin', 0).apply(lambda x: 5 if x > 0 else 0)
    iot23['fin_flag_count'] = df.get('tcprtt', 0).apply(lambda x: 3 if x > 0 else 0)
    iot23['rst_flag_count'] = 0
    iot23['psh_flag_count'] = 0
    iot23['urg_flag_count'] = 0
    iot23['unique_src_ports'] = 1
    iot23['unique_dst_ports'] = 1
    iot23['payload_entropy'] = np.random.uniform(3.0, 6.0, len(df))
    
    proto = df.get('proto', 'other')
    iot23['dns_query_count'] = (proto == 'udp').astype(int) * 5
    iot23['http_request_count'] = df.get('dport', 0).isin([80, 8080]).astype(int) * 5
    iot23['ssl_handshake_count'] = df.get('dport', 0).isin([443, 8443]).astype(int) * 3
    
    return iot23.fillna(0)


def convert_generic_to_iot23(df):
    """Generic converter for unknown formats"""
    iot23 = pd.DataFrame()
    n = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) >= 5:
        col1 = df[numeric_cols[0]].clip(lower=0.01)
        col2 = df[numeric_cols[1]].clip(lower=1)
        col3 = df[numeric_cols[2]].clip(lower=1)
        
        iot23['packet_rate'] = (col2 / col1).clip(upper=10000)
        iot23['packet_size'] = col3.clip(upper=65535)
        iot23['byte_rate'] = (col2 * col3 / col1).clip(upper=1e9)
        iot23['flow_duration'] = col1
        iot23['total_packets'] = col2
        iot23['total_bytes'] = col2 * col3
    else:
        iot23['packet_rate'] = np.random.uniform(10, 1000, n)
        iot23['packet_size'] = np.random.uniform(64, 1500, n)
        iot23['byte_rate'] = iot23['packet_rate'] * iot23['packet_size']
        iot23['flow_duration'] = np.random.uniform(1, 100, n)
        iot23['total_packets'] = iot23['packet_rate'] * iot23['flow_duration']
        iot23['total_bytes'] = iot23['byte_rate'] * iot23['flow_duration']
    
    iot23['entropy'] = np.random.uniform(2.0, 7.0, n)
    iot23['port_scan_score'] = np.random.uniform(0, 50, n)
    iot23['syn_flag_count'] = np.random.randint(0, 10, n)
    iot23['ack_flag_count'] = np.random.randint(0, 10, n)
    iot23['fin_flag_count'] = np.random.randint(0, 5, n)
    iot23['rst_flag_count'] = np.random.randint(0, 5, n)
    iot23['psh_flag_count'] = np.random.randint(0, 5, n)
    iot23['urg_flag_count'] = np.random.randint(0, 2, n)
    iot23['unique_src_ports'] = 1
    iot23['unique_dst_ports'] = 1
    iot23['payload_entropy'] = np.random.uniform(3.0, 6.0, n)
    iot23['dns_query_count'] = np.random.randint(0, 5, n)
    iot23['http_request_count'] = np.random.randint(0, 10, n)
    iot23['ssl_handshake_count'] = np.random.randint(0, 3, n)
    
    return iot23.fillna(0)

# ---------------------- Misc --------------------------

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV supported'}), 400
    dataset_name = request.form.get('dataset_name', 'uploaded')
    target_dir = os.path.join(UPLOAD_DIR, dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, file.filename)
    file.save(path)
    try:
        df = pd.read_csv(path)
        return jsonify({'success': True, 'stats': {
            'filename': file.filename,
            'rows': len(df),
            'columns': len(df.columns),
            'has_label': 'label' in df.columns
        }})
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV: {e}'}), 400

# ---------------------- Threats by IP --------------------------

@app.route('/api/threats_by_ip', methods=['GET'])
def api_threats_by_ip():
    ip = request.args.get('ip', '').strip()
    if not ip:
        return jsonify({'success': False, 'error': 'No IP provided'}), 400
    # Search both src and dst IPs in threat records
    results = []
    for threat in threat_db.threats:
        src = threat.get('features', {}).get('src_ip') or threat.get('src_ip')
        dst = threat.get('features', {}).get('dst_ip') or threat.get('dst_ip')
        if src == ip or dst == ip:
            results.append(threat)
    return jsonify({'success': True, 'threats': results})

# ---------------------- Entrypoint --------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Unified Next-Gen IDS Dashboard')
    parser.add_argument('--port', type=int, default=int(os.getenv('DASHBOARD_PORT', '8080')))
    parser.add_argument('--host', type=str, default=os.getenv('DASHBOARD_HOST', '0.0.0.0'))
    args = parser.parse_args()

    print('=' * 72)
    print('🛡️  Unified Next-Gen IDS Dashboard')
    print('=' * 72)
    print(f'📊 URL: http://{args.host}:{args.port}')
    print(f'🔍 Checkpoint: {CHECKPOINT_PATH}')
    print(f'📁 Threat DB: {threat_db.db_path}')
    print('Endpoints: /api/train, /api/evaluate, /api/analyze_traffic, /api/dashboard/ingest,')
    print('           /api/phishing/url, /api/phishing/email, /api/logs/ingest, /api/remediation/*')
    print('Press Ctrl+C to stop.')

    app.run(debug=False, host=args.host, port=args.port, use_reloader=False)
