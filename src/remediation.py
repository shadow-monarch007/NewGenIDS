"""
Remediation Module
------------------
Provides safe, non-destructive action registration for automated / semi-
automated response. Instead of actually blocking IPs or killing processes, it
records intent so an operator can approve execution later or an external
orchestrator can pick it up.

Future integration could call OS commands (e.g., firewall rules) behind a
safety flag. For now we keep side effects limited to writing JSON.

Public API:
    propose_action(action_type: str, details: dict) -> dict
    list_actions() -> list[dict]
    update_action(id: int, status: str) -> bool

Action statuses:
    PENDING -> proposed by detector
    APPROVED -> operator approved
    EXECUTED -> carried out by external system
    REJECTED -> dismissed

Stored at results/remediation_actions.json
"""
from __future__ import annotations
import os
import json
from datetime import datetime
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
STORE_PATH = os.path.join(RESULTS_DIR, 'remediation_actions.json')

os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize store if missing
if not os.path.exists(STORE_PATH):
    with open(STORE_PATH, 'w', encoding='utf-8') as f:
        json.dump([], f)

SAFE_ACTION_TYPES = [
    'BLOCK_IP', 'UNBLOCK_IP', 'ISOLATE_HOST', 'RELEASE_HOST', 'DISABLE_USER',
    'ENABLE_USER', 'QUARANTINE_FILE', 'RESTORE_FILE', 'REVOKE_TOKEN', 'ROTATE_CREDENTIALS'
]

VALID_STATUSES = ['PENDING', 'APPROVED', 'EXECUTED', 'REJECTED']


def _load() -> List[Dict[str, any]]:
    with open(STORE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save(actions: List[Dict[str, any]]):
    with open(STORE_PATH, 'w', encoding='utf-8') as f:
        json.dump(actions, f, indent=2)


def propose_action(action_type: str, details: Dict[str, any]) -> Dict[str, any]:
    if action_type not in SAFE_ACTION_TYPES:
        raise ValueError(f'Unsupported action_type: {action_type}')
    actions = _load()
    action_id = (actions[-1]['id'] + 1) if actions else 1
    record = {
        'id': action_id,
        'type': action_type,
        'details': details,
        'status': 'PENDING',
        'timestamp': datetime.utcnow().isoformat()
    }
    actions.append(record)
    _save(actions)
    return record


def list_actions() -> List[Dict[str, any]]:
    return _load()


def update_action(action_id: int, status: str) -> bool:
    if status not in VALID_STATUSES:
        raise ValueError('Invalid status')
    actions = _load()
    for a in actions:
        if a['id'] == action_id:
            a['status'] = status
            a['updated_at'] = datetime.utcnow().isoformat()
            _save(actions)
            return True
    return False

__all__ = ['propose_action', 'list_actions', 'update_action', 'SAFE_ACTION_TYPES', 'VALID_STATUSES']
