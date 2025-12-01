import os
import time
import json
import requests
from datetime import datetime

BASE = os.environ.get('DASHBOARD_URL', 'http://127.0.0.1:8080')
SLEEP = float(os.environ.get('EVENT_INTERVAL', '1.0'))

SAMPLES = [
    {"attack_type":"DDoS","severity":"Critical","confidence":0.97},
    {"attack_type":"Port Scan","severity":"High","confidence":0.88},
    {"attack_type":"Malware C2","severity":"High","confidence":0.91},
    {"attack_type":"Brute Force","severity":"Medium","confidence":0.73},
    {"attack_type":"SQL Injection","severity":"High","confidence":0.82},
    {"attack_type":"Normal","severity":"Low","confidence":0.10},
]

if __name__ == '__main__':
    print(f"Streaming {len(SAMPLES)} events to {BASE} ...")
    for e in SAMPLES:
        e['timestamp'] = datetime.utcnow().isoformat()
        r = requests.post(f"{BASE}/api/dashboard/ingest", json=e, timeout=10)
        if r.ok:
            print("→", e['attack_type'], e['severity'])
        else:
            print("! post failed", r.status_code, r.text[:200])
        time.sleep(SLEEP)
    print("Done.")
