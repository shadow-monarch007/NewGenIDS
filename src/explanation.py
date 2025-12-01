"""Explanation Module
---------------------
Provides generate_ai_explanation() previously embedded in legacy dashboard.
Separated so other components can import without pulling full dashboard code.
"""
from __future__ import annotations
from datetime import datetime
from typing import Dict, Any

def generate_ai_explanation(intrusion_type: str, features: Dict[str, Any], confidence: float) -> Dict[str, Any]:
    packet_rate = features.get('packet_rate', 0)
    packet_size = features.get('packet_size', 0)
    byte_rate = features.get('byte_rate', 0)
    flow_duration = features.get('flow_duration', 0)
    entropy = features.get('entropy', 0)
    dst_port = features.get('dst_port', 0)
    total_packets = features.get('total_packets', 0)

    explanations = {
        "Attack": {
            "description": f"Malicious network activity detected with {confidence:.1%} confidence. High packet rate ({packet_rate:.0f}/s) and anomalous traffic patterns suggest intrusion attempt.",
            "indicators": [
                f"Anomalous packet rate: {packet_rate:.0f} packets/sec",
                f"Destination port: {int(dst_port)}",
                f"Packet size: {packet_size:.0f} bytes",
                f"Traffic entropy: {entropy:.2f}",
                f"Total packets: {int(total_packets)}",
                f"Byte rate: {byte_rate:.0f} bytes/sec"
            ],
            "mitigation": [
                "Block source IP at firewall",
                "Enable rate limiting",
                "Review network logs for attack patterns",
                "Alert security team",
                "Consider deploying additional IDS rules"
            ],
            "severity": "Critical" if confidence > 0.9 else "High" if confidence > 0.7 else "Medium",
            "attack_stage": "Active Attack"
        },
        "DDoS": {
            "description": f"Distributed Denial of Service attack detected. Multiple sources flooding the network with {packet_rate:.0f} packets/sec to overwhelm resources.",
            "indicators": [
                f"High packet rate: {packet_rate:.0f} packets/sec",
                f"Destination port: {int(dst_port)}",
                f"Small packet size: {packet_size:.0f} bytes",
                f"Entropy: {entropy:.2f}",
                f"Total packets: {int(total_packets)}",
                f"Byte rate: {byte_rate:.0f} bytes/sec"
            ],
            "mitigation": [
                "Enable rate limiting on firewall",
                "Deploy upstream DDoS protection service",
                "Block abusive IP ranges",
                "Enable SYN cookies",
                "Scale horizontally if possible"
            ],
            "severity": "Critical" if packet_rate > 500 else "High",
            "attack_stage": "Active Attack"
        },
        "Port_Scan": {
            "description": f"Reconnaissance activity probing port {int(dst_port)}.",
            "indicators": [
                "Sequential/multiple port access attempts",
                f"Target port: {int(dst_port)}",
                f"Conn duration: {flow_duration:.3f}s",
                f"Packet size: {packet_size:.0f} bytes",
                f"Entropy: {entropy:.2f}",
                f"Packet rate: {packet_rate:.0f} pps"
            ],
            "mitigation": [
                "Enable port scan detection rules",
                "Close unnecessary open ports",
                "Deploy fail2ban / similar",
                "Investigate source host"
            ],
            "severity": "Medium",
            "attack_stage": "Reconnaissance"
        },
        "Malware_C2": {
            "description": f"Possible command & control beacon every ~{flow_duration:.0f}s.",
            "indicators": [
                f"Outbound port: {int(dst_port)}",
                f"Beacon interval: {flow_duration:.1f}s",
                f"Entropy: {entropy:.2f}",
                f"Packet rate: {packet_rate:.0f} pps",
                f"Byte rate: {byte_rate:.0f} bytes/sec",
                f"Total packets: {int(total_packets)}"
            ],
            "mitigation": [
                "Isolate suspected host",
                "Run malware scan",
                "Block destination at firewall/DNS",
                "Perform forensic analysis"
            ],
            "severity": "Critical",
            "attack_stage": "Active Breach"
        },
        "Brute_Force": {
            "description": f"Repeated auth attempts on service port {int(dst_port)}.",
            "indicators": [
                f"Authentication port: {int(dst_port)}",
                f"Attempt rate: {packet_rate:.0f}/s",
                f"Conn duration: {flow_duration:.2f}s",
                f"Packet size: {packet_size:.0f} bytes",
                f"Total packets: {int(total_packets)}",
                f"Entropy: {entropy:.2f}"
            ],
            "mitigation": [
                "Enforce strong passwords",
                "Enable account lockout policies",
                "Deploy MFA",
                "Restrict port to trusted IPs"
            ],
            "severity": "High" if packet_rate > 20 else "Medium",
            "attack_stage": "Credential Attack"
        },
        "SQL_Injection": {
            "description": f"Potential SQL injection targeting web port {int(dst_port)}.",
            "indicators": [
                f"Request size: {packet_size:.0f} bytes",
                f"Web port: {int(dst_port)}",
                f"Byte rate: {byte_rate:.0f} bytes/sec",
                f"Request rate: {packet_rate:.0f}/s",
                f"Entropy: {entropy:.2f}",
                f"Total requests: {int(total_packets)}"
            ],
            "mitigation": [
                "Use parameterized queries",
                "Sanitize user inputs",
                "Deploy WAF rules",
                "Least privilege DB accounts"
            ],
            "severity": "High" if byte_rate > 5000 else "Medium",
            "attack_stage": "Application Attack"
        },
        "Normal": {
            "description": f"Normal traffic observed on port {int(dst_port)}.",
            "indicators": [
                f"Packet rate {packet_rate:.0f} pps within baseline",
                f"Packet size {packet_size:.0f} bytes",
                f"Duration {flow_duration:.2f}s",
                f"Entropy {entropy:.2f}",
                f"Byte rate {byte_rate:.0f} bytes/sec",
                f"Port {int(dst_port)} typical"
            ],
            "mitigation": [
                "No action required", "Continue monitoring"
            ],
            "severity": "None",
            "attack_stage": "None"
        },
        "unknown": {
            "description": f"Anomalous pattern deviating from baseline on port {int(dst_port)}.",
            "indicators": [
                f"Confidence {confidence:.1%}",
                f"Packet rate {packet_rate:.0f} pps", f"Packet size {packet_size:.0f} bytes", f"Entropy {entropy:.2f}",
                f"Flow duration {flow_duration:.2f}s", "No match to known signatures"
            ],
            "mitigation": [
                "Review SHAP feature impact", "Correlate with firewall/AV logs", "Monitor host", "Escalate if persists"
            ],
            "severity": "Low",
            "attack_stage": "Anomaly"
        }
    }
    data = explanations.get(intrusion_type, explanations['unknown'])
    return {
        'intrusion_type': intrusion_type,
        'confidence': confidence,
        'description': data['description'],
        'indicators': data['indicators'],
        'mitigation_steps': data['mitigation'],
        'severity': data['severity'],
        'attack_stage': data['attack_stage'],
        'timestamp': datetime.now().isoformat(),
        'recommended_priority': 'IMMEDIATE' if data['severity'] == 'Critical' else 'HIGH' if data['severity'] == 'High' else 'MEDIUM' if data['severity'] == 'Medium' else 'LOW'
    }

__all__ = ['generate_ai_explanation']