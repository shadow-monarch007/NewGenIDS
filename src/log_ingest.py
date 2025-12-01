"""
Log Ingestion Module
--------------------
Parses generic system / application logs and emits threat candidates.
Focus: lightweight pattern + anomaly cues (no heavy ML). Designed so a future
ML classifier can replace `classify_log_line()` while keeping interface.

Supported sources (conceptual identifiers in events):
  - windows_event (raw exported text lines)
  - syslog (RFC 5424 or classic format)
  - app (custom app logs with key=value pairs)

Public API:
  ingest_lines(lines: list[str], source: str = 'syslog') -> list[dict]
  classify_log_line(line: str, source: str) -> dict | None

Event dict keys:
  type          : High-level category (AUTH_FAILURE, EXECUTION, NETWORK, PRIV_ESC, GENERIC)
  severity      : Low | Medium | High | Critical
  indicators    : list[str] triggered heuristics
  raw           : original line
  timestamp     : extracted or generated ISO timestamp
  recommended   : list[str] next-step actions

Heuristics (examples):
  - Multiple failed auth attempts (AUTH_FAILURE)
  - Powershell / cmd suspicious flags (EXECUTION)
  - Unexpected admin group modification (PRIV_ESC)
  - Suspicious network port scanning phrases (NETWORK)
  - Encoded command patterns (base64) (EXECUTION)
"""
from __future__ import annotations
import re
from datetime import datetime
from typing import List, Dict, Optional

AUTH_FAIL_PAT = re.compile(r'failed|authentication failure|invalid password', re.IGNORECASE)
AUTH_SUCCESS_PAT = re.compile(r'successful login', re.IGNORECASE)
POWERSHELL_PAT = re.compile(r'powershell.exe', re.IGNORECASE)
SUSP_PS_FLAGS = re.compile(r'-nop|-w hidden|-enc|-encodedcommand', re.IGNORECASE)
CMD_PAT = re.compile(r'cmd.exe', re.IGNORECASE)
BASE64_PAT = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
PRIV_ESC_PAT = re.compile(r'added to (Administrators|sudoers)|privilege escalation', re.IGNORECASE)
PORT_SCAN_PAT = re.compile(r'nmap|masscan|zmap|port scan', re.IGNORECASE)
IP_ADDR_PAT = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
TIME_PAT = re.compile(r'(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})')

RECOMMENDED_ACTIONS = {
    'AUTH_FAILURE': [
        'Review source IP for brute force pattern.',
        'Consider temporary IP block or MFA enforcement.',
        'Check for lateral movement attempts.'
    ],
    'EXECUTION': [
        'Capture full process tree for forensic review.',
        'Isolate host if malicious execution confirmed.',
        'Check for further persistence mechanisms.'
    ],
    'PRIV_ESC': [
        'Validate if group change was authorized.',
        'Audit recent administrative actions.',
        'Scan for credential dump tools.'
    ],
    'NETWORK': [
        'Monitor outbound connections for exfiltration.',
        'Filter suspicious scanning IPs at firewall.',
        'Search for correlated IDS alerts.'
    ],
    'GENERIC': [
        'Correlate with other log sources (SIEM).',
        'No immediate action; continue monitoring.'
    ]
}

SEVERITY_ORDER = ['Low', 'Medium', 'High', 'Critical']


def _severity(score: int) -> str:
    if score >= 80:
        return 'Critical'
    if score >= 55:
        return 'High'
    if score >= 30:
        return 'Medium'
    return 'Low'


def classify_log_line(line: str, source: str) -> Optional[Dict[str, any]]:
    raw = line.strip()
    if not raw:
        return None
    lower = raw.lower()
    indicators: List[str] = []
    score = 0
    event_type = 'GENERIC'

    # Timestamp extraction
    ts_match = TIME_PAT.search(raw)
    timestamp = ts_match.group(1) if ts_match else datetime.utcnow().isoformat()

    # Auth failures
    if AUTH_FAIL_PAT.search(raw):
        event_type = 'AUTH_FAILURE'
        indicators.append('auth_failure')
        score += 35
        # Multiple ip addresses may indicate distributed brute force
        ips = IP_ADDR_PAT.findall(raw)
        if len(ips) >= 2:
            indicators.append('multi_source_attempt')
            score += 10

    # Suspicious execution
    if POWERSHELL_PAT.search(raw) or CMD_PAT.search(raw):
        event_type = 'EXECUTION'
        indicators.append('shell_execution')
        score += 25
        if SUSP_PS_FLAGS.search(raw):
            indicators.append('susp_ps_flags')
            score += 25
        if BASE64_PAT.search(raw):
            indicators.append('encoded_payload')
            score += 20

    # Privilege escalation
    if PRIV_ESC_PAT.search(raw):
        event_type = 'PRIV_ESC'
        indicators.append('privilege_change')
        score += 40

    # Network scanning
    if PORT_SCAN_PAT.search(raw):
        event_type = 'NETWORK'
        indicators.append('scan_tool_detected')
        score += 30

    # Raise severity if both exec + enc payload + priv esc in same line (rare)
    if 'shell_execution' in indicators and 'privilege_change' in indicators and 'encoded_payload' in indicators:
        score += 30
        indicators.append('compound_attack_chain')

    result = {
        'raw': raw,
        'type': event_type,
        'severity': _severity(score),
        'indicators': indicators,
        'score': score,
        'timestamp': timestamp,
        'recommended': RECOMMENDED_ACTIONS.get(event_type, RECOMMENDED_ACTIONS['GENERIC'])
    }
    return result


def ingest_lines(lines: List[str], source: str = 'syslog') -> List[Dict[str, any]]:
    events = []
    for line in lines:
        evt = classify_log_line(line, source)
        if evt:
            events.append(evt)
    return events

__all__ = ['ingest_lines', 'classify_log_line']
