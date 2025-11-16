# Final-Year Testing Guide (Windows)

These steps validate the system end-to-end without production tooling.

Prerequisites
- Python venv active and dependencies installed:
  - `pip install -r requirements.txt`
- Npcap installed (required for live capture): https://nmap.org/npcap/

1) Train (or reuse existing checkpoint)
```powershell
python src\train.py --dataset iot23 --epochs 3 --use-arnn
```

2) Start the production dashboard
```powershell
python src\dashboard_live.py
```
Open http://localhost:5000 in your browser.

3) Quick test (no admin required): one alert + chain verify
```powershell
python scripts\quick_test.py --checkpoint checkpoints\best_iot23.pt
```
What this does:
- Runs model on `data/iot23/demo_attacks.csv`
- Appends one alert to `results/alerts_chain.json`
- POSTs the alert to `/api/dashboard/ingest` (shows instantly on dashboard)
- Calls `/api/blockchain/verify` and prints status

4) Realtime capture demo (Administrator PowerShell)
```powershell
# Find your interface name (e.g., Ethernet, Wi-Fi)
Get-NetAdapter | Select-Object Name, Status

# Run realtime capture (replace interface as needed)
python src\realtime.py --iface "Ethernet" --window 3 --checkpoint checkpoints\best_iot23.pt
```
You should see live toasts on the dashboard and stat cards updating.
Click “Verify Chain” at the top-right to confirm integrity anytime.

Troubleshooting
- Realtime shows no packets: ensure Administrator shell and correct interface name.
- Dashboard not updating: confirm `src/dashboard_live.py` is running; open browser devtools and check SSE to `/events`.
- Missing checkpoint: run step (1) to train.
- Port conflicts: change Flask port in `dashboard_live.py` (app.run(..., port=5000)).
