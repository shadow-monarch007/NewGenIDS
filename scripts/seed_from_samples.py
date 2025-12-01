import os
import requests

BASE = os.environ.get('DASHBOARD_URL', 'http://127.0.0.1:8080')
SAMPLES = [
    'data/iot23/demo_samples/ddos.csv',
    'data/iot23/demo_samples/malware_c2.csv',
    'data/iot23/demo_samples/port_scan.csv',
    'data/iot23/demo_samples/normal_web_browsing.csv'
]

if __name__ == '__main__':
    for path in SAMPLES:
        if not os.path.exists(path):
            print('Missing:', path)
            continue
        print('Uploading', path)
        with open(path, 'rb') as f:
            r = requests.post(f"{BASE}/api/analyze_traffic", files={'file': (os.path.basename(path), f)})
        try:
            j = r.json()
        except Exception:
            print('Bad response:', r.status_code)
            continue
        if j.get('success'):
            print('✔ analyzed:', path, '→', j.get('prediction',{}).get('attack_type'))
        else:
            print('✖ failed:', path, j)
