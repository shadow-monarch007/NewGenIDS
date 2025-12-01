"""
security.py - Security middleware and input validation
---------------------------------------------------
Implements rate limiting, CSRF protection, and input sanitization
"""
import re
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, session
from collections import defaultdict

class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self):
        self.requests = defaultdict(list)  # ip -> [timestamp, timestamp, ...]
        self.limits = {
            'default': (100, 60),  # 100 requests per 60 seconds
            'login': (5, 300),     # 5 login attempts per 5 minutes
            'upload': (10, 60),    # 10 uploads per minute
            'analysis': (20, 60),  # 20 analysis requests per minute
        }
    
    def _clean_old_requests(self, ip, limit_key, window):
        """Remove expired timestamps"""
        cutoff = datetime.now() - timedelta(seconds=window)
        self.requests[f"{ip}:{limit_key}"] = [
            ts for ts in self.requests[f"{ip}:{limit_key}"]
            if ts > cutoff
        ]
    
    def is_rate_limited(self, ip, limit_key='default'):
        """Check if IP has exceeded rate limit"""
        max_requests, window = self.limits.get(limit_key, self.limits['default'])
        key = f"{ip}:{limit_key}"
        
        # Clean old requests
        self._clean_old_requests(ip, limit_key, window)
        
        # Check limit
        if len(self.requests[key]) >= max_requests:
            return True
        
        # Add current request
        self.requests[key].append(datetime.now())
        return False
    
    def get_remaining(self, ip, limit_key='default'):
        """Get remaining requests for IP"""
        max_requests, window = self.limits.get(limit_key, self.limits['default'])
        key = f"{ip}:{limit_key}"
        self._clean_old_requests(ip, limit_key, window)
        return max(0, max_requests - len(self.requests[key]))

class CSRFProtection:
    """CSRF token management"""
    
    def __init__(self):
        self.tokens = {}  # session_id -> token
        self.token_lifetime = 3600  # 1 hour
    
    def generate_token(self, session_id):
        """Generate CSRF token for session"""
        token = secrets.token_urlsafe(32)
        self.tokens[session_id] = {
            'token': token,
            'created_at': datetime.now()
        }
        return token
    
    def validate_token(self, session_id, token):
        """Validate CSRF token"""
        if session_id not in self.tokens:
            return False
        
        token_data = self.tokens[session_id]
        
        # Check expiration
        if datetime.now() - token_data['created_at'] > timedelta(seconds=self.token_lifetime):
            del self.tokens[session_id]
            return False
        
        return token_data['token'] == token

class InputValidator:
    """Input validation and sanitization"""
    
    @staticmethod
    def validate_ip(ip):
        """Validate IP address format"""
        if not ip:
            return False, "IP address is required"
        
        # IPv4 pattern
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(ipv4_pattern, ip):
            # Check octets are 0-255
            octets = ip.split('.')
            if all(0 <= int(octet) <= 255 for octet in octets):
                return True, ip
        
        # IPv6 pattern (simplified)
        ipv6_pattern = r'^([0-9a-fA-F]{0,4}:){7}[0-9a-fA-F]{0,4}$'
        if re.match(ipv6_pattern, ip):
            return True, ip
        
        return False, "Invalid IP address format"
    
    @staticmethod
    def validate_filename(filename, allowed_extensions=None):
        """Validate filename for security"""
        if not filename:
            return False, "Filename is required"
        
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return False, "Invalid filename: path traversal detected"
        
        # Check extension if provided
        if allowed_extensions:
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            if ext not in allowed_extensions:
                return False, f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
        
        # Check length
        if len(filename) > 255:
            return False, "Filename too long (max 255 characters)"
        
        return True, filename
    
    @staticmethod
    def validate_url(url):
        """Validate URL format"""
        if not url:
            return False, "URL is required"
        
        # Basic URL pattern
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, url):
            return False, "Invalid URL format"
        
        # Check length
        if len(url) > 2048:
            return False, "URL too long (max 2048 characters)"
        
        # Check for suspicious patterns
        suspicious = ['javascript:', 'data:', 'file:', 'vbscript:']
        if any(url.lower().startswith(s) for s in suspicious):
            return False, "Suspicious URL protocol detected"
        
        return True, url
    
    @staticmethod
    def sanitize_string(text, max_length=1000):
        """Sanitize text input"""
        if not text:
            return ""
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if char.isprintable() or char in ['\n', '\t'])
        
        # Trim length
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()
    
    @staticmethod
    def validate_port(port):
        """Validate port number"""
        try:
            port_num = int(port)
            if 1 <= port_num <= 65535:
                return True, port_num
            return False, "Port must be between 1 and 65535"
        except ValueError:
            return False, "Port must be a number"
    
    @staticmethod
    def validate_threat_type(threat_type):
        """Validate threat type"""
        valid_types = ['DDoS', 'Port_Scan', 'Malware_C2', 'Brute_Force', 'SQL_Injection', 'Normal']
        if threat_type not in valid_types:
            return False, f"Invalid threat type. Valid: {', '.join(valid_types)}"
        return True, threat_type

# Global instances
rate_limiter = RateLimiter()
csrf_protection = CSRFProtection()
input_validator = InputValidator()

# Decorators
def rate_limit(limit_key='default'):
    """Decorator for rate limiting"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            ip = request.remote_addr
            
            if rate_limiter.is_rate_limited(ip, limit_key):
                return jsonify({
                    'error': 'Rate limit exceeded. Please try again later.',
                    'code': 429
                }), 429
            
            # Add remaining count to response headers
            remaining = rate_limiter.get_remaining(ip, limit_key)
            response = f(*args, **kwargs)
            
            # If response is tuple (for status codes)
            if isinstance(response, tuple):
                return response
            
            return response
        return decorated_function
    return decorator

def require_csrf():
    """Decorator for CSRF protection"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            session_id = session.get('session_id')
            csrf_token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
            
            if not session_id or not csrf_token:
                return jsonify({
                    'error': 'CSRF token missing',
                    'code': 403
                }), 403
            
            if not csrf_protection.validate_token(session_id, csrf_token):
                return jsonify({
                    'error': 'Invalid CSRF token',
                    'code': 403
                }), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_input(**validators):
    """Decorator for input validation"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get data from request
            data = request.get_json() or request.form.to_dict() or request.args.to_dict()
            
            # Validate each field
            for field, validator_func in validators.items():
                if field in data:
                    valid, result = validator_func(data[field])
                    if not valid:
                        return jsonify({
                            'error': f'Validation failed for {field}: {result}',
                            'code': 400
                        }), 400
                    data[field] = result
            
            # Add validated data to request
            request.validated_data = data
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator
