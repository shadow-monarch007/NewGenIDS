"""
Production deployment configuration and launcher
"""
import os
import sys

# Gunicorn configuration
bind = "0.0.0.0:8080"
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Process naming
proc_name = "nextgen_ids"

# Server mechanics
daemon = False
pidfile = "logs/gunicorn.pid"
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (uncomment and configure for HTTPS)
# keyfile = "/path/to/key.pem"
# certfile = "/path/to/cert.pem"

def on_starting(server):
    """Called just before the master process is initialized"""
    os.makedirs("logs", exist_ok=True)
    print("🚀 Starting NextGen IDS Production Server...")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP"""
    print("🔄 Reloading workers...")

def when_ready(server):
    """Called just after the server is started"""
    print(f"✓ Server ready at http://{bind}")
    print(f"✓ Workers: {workers}")
    print(f"✓ Worker class: {worker_class}")
    print("✓ Press CTRL+C to quit")

def on_exit(server):
    """Called just before exiting Gunicorn"""
    print("👋 Shutting down NextGen IDS...")
