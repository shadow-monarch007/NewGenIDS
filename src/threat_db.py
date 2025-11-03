"""
Threat Database Manager
----------------------
Stores and retrieves detected threats for dashboard display.
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict


class ThreatDatabase:
    def __init__(self, db_path: str = "data/threats.json"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.threats = self._load()
    
    def _load(self) -> List[Dict]:
        """Load threats from JSON file."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load threat database: {e}")
                return []
        return []
    
    def _save(self):
        """Save threats to JSON file."""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.threats, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save threat database: {e}")
    
    def add_threat(self, threat: Dict):
        """Add a new threat detection."""
        threat['id'] = len(self.threats) + 1
        threat['timestamp'] = threat.get('timestamp', datetime.now().isoformat())
        threat['status'] = threat.get('status', 'Active')
        self.threats.append(threat)
        self._save()
    
    def add_threats_bulk(self, threats: List[Dict]):
        """Add multiple threats at once."""
        for threat in threats:
            threat['id'] = len(self.threats) + 1
            threat['timestamp'] = threat.get('timestamp', datetime.now().isoformat())
            threat['status'] = threat.get('status', 'Active')
            self.threats.append(threat)
        self._save()
    
    def get_recent_threats(self, limit: int = 10) -> List[Dict]:
        """Get most recent threats."""
        return sorted(self.threats, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
    
    def get_by_severity(self, severity: str) -> List[Dict]:
        """Get threats by severity level."""
        return [t for t in self.threats if t.get('severity', '').lower() == severity.lower()]
    
    def get_statistics(self) -> Dict:
        """Calculate threat statistics for dashboard."""
        if not self.threats:
            return {
                "total": 0,
                "critical": 0,
                "active": 0,
                "remediated": 0,
                "by_severity": {},
                "by_status": {},
                "by_type": {}
            }
        
        stats = {
            "total": len(self.threats),
            "critical": 0,
            "active": 0,
            "remediated": 0,
            "false_positive": 0,
            "investigating": 0,
            "by_severity": defaultdict(int),
            "by_status": defaultdict(int),
            "by_type": defaultdict(int)
        }
        
        for threat in self.threats:
            severity = threat.get('severity', 'Unknown')
            status = threat.get('status', 'Unknown')
            attack_type = threat.get('attack_type', 'Unknown')
            
            # Count by severity
            stats['by_severity'][severity] += 1
            if severity.lower() == 'critical':
                stats['critical'] += 1
            
            # Count by status
            stats['by_status'][status] += 1
            if status.lower() == 'active':
                stats['active'] += 1
            elif status.lower() in ['remediated', 'resolved']:
                stats['remediated'] += 1
            elif status.lower() == 'false_positive':
                stats['false_positive'] += 1
            elif status.lower() == 'investigating':
                stats['investigating'] += 1
            
            # Count by type
            stats['by_type'][attack_type] += 1
        
        return stats
    
    def update_status(self, threat_id: int, new_status: str):
        """Update threat status."""
        for threat in self.threats:
            if threat.get('id') == threat_id:
                threat['status'] = new_status
                threat['updated_at'] = datetime.now().isoformat()
                self._save()
                return True
        return False
    
    def clear_all(self):
        """Clear all threats (for testing)."""
        self.threats = []
        self._save()
    
    def get_timeline_data(self, days: int = 7) -> List[Dict]:
        """Get threat counts per day for timeline chart."""
        from collections import defaultdict
        from datetime import datetime, timedelta
        
        timeline = defaultdict(lambda: defaultdict(int))
        
        # Initialize last N days
        end_date = datetime.now()
        for i in range(days):
            date = (end_date - timedelta(days=i)).strftime('%Y-%m-%d')
            timeline[date] = defaultdict(int)
        
        # Count threats per day by severity
        for threat in self.threats:
            try:
                timestamp = datetime.fromisoformat(threat.get('timestamp', ''))
                date = timestamp.strftime('%Y-%m-%d')
                severity = threat.get('severity', 'Unknown')
                timeline[date][severity] += 1
            except Exception:
                continue
        
        # Convert to list format for charts
        result = []
        for date in sorted(timeline.keys()):
            result.append({
                'date': date,
                'critical': timeline[date]['Critical'],
                'high': timeline[date]['High'],
                'medium': timeline[date]['Medium'],
                'low': timeline[date]['Low']
            })
        
        return result
