"""
auto_response.py - Automated Threat Response System
-------------------------------------------------
Implements automated responses to detected threats with safety checks
"""
import os
import json
import subprocess
import platform
from datetime import datetime
from enum import Enum

class ResponseAction(Enum):
    """Available response actions"""
    BLOCK_IP = "block_ip"
    ALLOW_IP = "allow_ip"
    BLOCK_PORT = "block_port"
    ISOLATE_HOST = "isolate_host"
    KILL_PROCESS = "kill_process"
    DISABLE_USER = "disable_user"
    ALERT_ONLY = "alert_only"
    LOG_EVENT = "log_event"

class AutoResponseSystem:
    """Manages automated responses to security threats"""
    
    def __init__(self, enabled=True, dry_run=True, config_file='config/auto_response.json'):
        self.enabled = enabled
        self.dry_run = dry_run  # Safety: don't actually execute by default
        self.config_file = config_file
        self.actions_log_file = 'results/auto_response_log.json'
        self.blocked_ips = set()
        self.is_windows = platform.system() == 'Windows'
        self._load_config()
        self._ensure_logs()
    
    def _load_config(self):
        """Load configuration"""
        default_config = {
            'enabled': True,
            'dry_run': True,
            'auto_block_threshold': 0.90,  # Auto-block if confidence >= 90%
            'allowed_actions': ['BLOCK_IP', 'LOG_EVENT', 'ALERT_ONLY'],
            'whitelist_ips': ['127.0.0.1', '::1', '192.168.1.1'],  # Never block these
            'max_blocks_per_hour': 50,
            'notification_email': None
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except:
                self.config = default_config
        else:
            self.config = default_config
            self._save_config()
    
    def _save_config(self):
        """Save configuration"""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _ensure_logs(self):
        """Ensure log file exists"""
        os.makedirs(os.path.dirname(self.actions_log_file), exist_ok=True)
        if not os.path.exists(self.actions_log_file):
            with open(self.actions_log_file, 'w') as f:
                json.dump([], f)
    
    def _log_action(self, action, target, result, details=None):
        """Log action to file"""
        try:
            with open(self.actions_log_file, 'r') as f:
                logs = json.load(f)
        except:
            logs = []
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'target': target,
            'result': result,
            'dry_run': self.dry_run,
            'details': details or {}
        }
        
        logs.append(log_entry)
        
        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(self.actions_log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return log_entry
    
    def _is_whitelisted(self, ip):
        """Check if IP is whitelisted"""
        return ip in self.config.get('whitelist_ips', [])
    
    def _execute_windows_firewall_block(self, ip):
        """Block IP using Windows Firewall"""
        if self.dry_run:
            return True, f"[DRY RUN] Would block IP {ip} via Windows Firewall"
        
        try:
            rule_name = f"IDS_Block_{ip.replace('.', '_')}"
            cmd = f'netsh advfirewall firewall add rule name="{rule_name}" dir=in action=block remoteip={ip}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            return True, f"Successfully blocked {ip}"
        except subprocess.CalledProcessError as e:
            return False, f"Failed to block {ip}: {e.stderr}"
    
    def _execute_linux_iptables_block(self, ip):
        """Block IP using iptables (Linux)"""
        if self.dry_run:
            return True, f"[DRY RUN] Would block IP {ip} via iptables"
        
        try:
            cmd = f'iptables -A INPUT -s {ip} -j DROP'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            return True, f"Successfully blocked {ip}"
        except subprocess.CalledProcessError as e:
            return False, f"Failed to block {ip}: {e}"
    
    def block_ip(self, ip, reason=None):
        """
        Block an IP address
        Returns: (success: bool, message: str)
        """
        if not self.enabled:
            return False, "Auto-response system is disabled"
        
        # Safety checks
        if self._is_whitelisted(ip):
            msg = f"IP {ip} is whitelisted, skipping block"
            self._log_action('BLOCK_IP', ip, 'SKIPPED_WHITELIST', {'reason': reason})
            return False, msg
        
        if ip in self.blocked_ips:
            return False, f"IP {ip} is already blocked"
        
        # Execute block
        if self.is_windows:
            success, message = self._execute_windows_firewall_block(ip)
        else:
            success, message = self._execute_linux_iptables_block(ip)
        
        if success:
            self.blocked_ips.add(ip)
        
        self._log_action('BLOCK_IP', ip, 'SUCCESS' if success else 'FAILED', {
            'reason': reason,
            'message': message
        })
        
        return success, message
    
    def unblock_ip(self, ip):
        """
        Unblock an IP address
        Returns: (success: bool, message: str)
        """
        if not self.enabled:
            return False, "Auto-response system is disabled"
        
        if self.dry_run:
            msg = f"[DRY RUN] Would unblock IP {ip}"
            self._log_action('ALLOW_IP', ip, 'DRY_RUN')
            return True, msg
        
        try:
            if self.is_windows:
                rule_name = f"IDS_Block_{ip.replace('.', '_')}"
                cmd = f'netsh advfirewall firewall delete rule name="{rule_name}"'
                subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            else:
                cmd = f'iptables -D INPUT -s {ip} -j DROP'
                subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            
            self.blocked_ips.discard(ip)
            msg = f"Successfully unblocked {ip}"
            self._log_action('ALLOW_IP', ip, 'SUCCESS', {'message': msg})
            return True, msg
            
        except subprocess.CalledProcessError as e:
            msg = f"Failed to unblock {ip}: {e}"
            self._log_action('ALLOW_IP', ip, 'FAILED', {'error': str(e)})
            return False, msg
    
    def respond_to_threat(self, threat_type, source_ip, confidence, details=None):
        """
        Automated response to detected threat
        Returns: (action_taken: str, success: bool, message: str)
        """
        if not self.enabled:
            return 'NONE', False, "Auto-response disabled"
        
        # Determine action based on threat type and confidence
        action_taken = 'ALERT_ONLY'
        
        # High confidence threats
        if confidence >= self.config.get('auto_block_threshold', 0.90):
            if threat_type in ['DDoS', 'Port_Scan', 'Brute_Force', 'SQL_Injection']:
                # Block the source IP
                success, message = self.block_ip(source_ip, reason=f"{threat_type} detected with {confidence:.2%} confidence")
                action_taken = 'BLOCK_IP' if success else 'ALERT_ONLY'
                
                return action_taken, success, message
        
        # Medium confidence - log and alert only
        elif confidence >= 0.70:
            msg = f"Medium confidence {threat_type} from {source_ip} (conf={confidence:.2%}) - monitoring"
            self._log_action('ALERT_ONLY', source_ip, 'LOGGED', {
                'threat_type': threat_type,
                'confidence': confidence,
                'details': details
            })
            return 'ALERT_ONLY', True, msg
        
        # Low confidence - just log
        else:
            msg = f"Low confidence {threat_type} from {source_ip} (conf={confidence:.2%}) - logged"
            self._log_action('LOG_EVENT', source_ip, 'LOGGED', {
                'threat_type': threat_type,
                'confidence': confidence
            })
            return 'LOG_EVENT', True, msg
    
    def get_blocked_ips(self):
        """Get list of currently blocked IPs"""
        return list(self.blocked_ips)
    
    def get_action_logs(self, limit=100):
        """Get recent action logs"""
        try:
            with open(self.actions_log_file, 'r') as f:
                logs = json.load(f)
            return logs[-limit:]
        except:
            return []
    
    def set_dry_run(self, enabled):
        """Enable/disable dry run mode"""
        self.dry_run = enabled
        self.config['dry_run'] = enabled
        self._save_config()
        return f"Dry run mode {'enabled' if enabled else 'disabled'}"
    
    def clear_all_blocks(self):
        """Clear all IP blocks (emergency use)"""
        if self.dry_run:
            return f"[DRY RUN] Would clear {len(self.blocked_ips)} blocked IPs"
        
        cleared = []
        for ip in list(self.blocked_ips):
            success, msg = self.unblock_ip(ip)
            if success:
                cleared.append(ip)
        
        return f"Cleared {len(cleared)} IP blocks"

# Global instance
auto_response = AutoResponseSystem(enabled=True, dry_run=True)
