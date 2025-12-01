"""
auth.py - Simple authentication system for dashboard
--------------------------------------------------
Provides basic username/password authentication with session management
"""
import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import session, request, jsonify

# Session configuration
SESSION_TIMEOUT = 3600  # 1 hour in seconds

# Default credentials (CHANGE THESE IN PRODUCTION!)
DEFAULT_USERS = {
    'admin': hashlib.sha256('admin123'.encode()).hexdigest(),
    'demo': hashlib.sha256('demo123'.encode()).hexdigest()
}

class AuthManager:
    """Manages user authentication and sessions"""
    
    def __init__(self, users_file='config/users.json'):
        self.users_file = users_file
        self.users = self._load_users()
        self.sessions = {}  # session_id -> user_info
        
    def _load_users(self):
        """Load users from file or use defaults"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return DEFAULT_USERS.copy()
    
    def _save_users(self):
        """Save users to file"""
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def add_user(self, username, password):
        """Add new user"""
        if username in self.users:
            return False
        self.users[username] = self.hash_password(password)
        self._save_users()
        return True
    
    def verify_credentials(self, username, password):
        """Verify username and password"""
        if username not in self.users:
            return False
        return self.users[username] == self.hash_password(password)
    
    def create_session(self, username):
        """Create new session for user"""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'username': username,
            'created_at': datetime.now(),
            'last_active': datetime.now()
        }
        return session_id
    
    def validate_session(self, session_id):
        """Check if session is valid and not expired"""
        if session_id not in self.sessions:
            return False
        
        session_info = self.sessions[session_id]
        last_active = session_info['last_active']
        
        # Check if session expired
        if datetime.now() - last_active > timedelta(seconds=SESSION_TIMEOUT):
            del self.sessions[session_id]
            return False
        
        # Update last active time
        session_info['last_active'] = datetime.now()
        return True
    
    def get_session_user(self, session_id):
        """Get username from session"""
        if session_id in self.sessions:
            return self.sessions[session_id]['username']
        return None
    
    def end_session(self, session_id):
        """End session (logout)"""
        if session_id in self.sessions:
            del self.sessions[session_id]

# Global auth manager instance
auth_manager = AuthManager()

def require_auth(f):
    """Decorator to require authentication for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for session cookie
        session_id = session.get('session_id')
        
        if not session_id or not auth_manager.validate_session(session_id):
            return jsonify({'error': 'Unauthorized. Please login.', 'code': 401}), 401
        
        # Add username to request context
        request.username = auth_manager.get_session_user(session_id)
        
        return f(*args, **kwargs)
    return decorated_function

def login_user(username, password):
    """
    Login user and create session
    Returns: (success: bool, message: str, session_id: str or None)
    """
    if not username or not password:
        return False, "Username and password required", None
    
    if not auth_manager.verify_credentials(username, password):
        return False, "Invalid credentials", None
    
    session_id = auth_manager.create_session(username)
    return True, "Login successful", session_id

def logout_user(session_id):
    """Logout user and end session"""
    auth_manager.end_session(session_id)
    return True, "Logout successful"

def change_password(username, old_password, new_password):
    """Change user password"""
    if not auth_manager.verify_credentials(username, old_password):
        return False, "Invalid current password"
    
    auth_manager.users[username] = auth_manager.hash_password(new_password)
    auth_manager._save_users()
    return True, "Password changed successfully"

def get_current_user():
    """Get currently logged in user from session"""
    session_id = session.get('session_id')
    if session_id:
        return auth_manager.get_session_user(session_id)
    return None
