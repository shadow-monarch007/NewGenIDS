"""
start_production.py - Production Launcher
---------------------------------------
Launch the IDS dashboard in production mode with proper configuration
"""
import os
import sys
import subprocess
import platform

def check_gunicorn():
    """Check if Gunicorn is installed"""
    try:
        import gunicorn
        return True
    except ImportError:
        return False

def install_gunicorn():
    """Install Gunicorn"""
    print("📦 Installing Gunicorn...")
    subprocess.run([sys.executable, "-m", "pip", "install", "gunicorn"], check=True)
    print("✓ Gunicorn installed")

def main():
    """Launch production server"""
    print("=" * 80)
    print("NextGen IDS - Production Deployment")
    print("=" * 80)
    print()
    
    # Check platform
    is_windows = platform.system() == 'Windows'
    
    if is_windows:
        print("⚠️  Note: Gunicorn is not officially supported on Windows")
        print("   Using Flask development server with multi-threading...")
        print()
        
        # Use Flask with threading on Windows
        from src.dashboard_unified import app
        
        print("🚀 Starting server on http://0.0.0.0:8080")
        print("✓ Press CTRL+C to quit")
        print()
        
        app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
        
    else:
        # Linux/Mac - use Gunicorn
        if not check_gunicorn():
            install_gunicorn()
        
        print("🚀 Starting Gunicorn server...")
        print()
        
        # Launch with Gunicorn
        os.execvp('gunicorn', [
            'gunicorn',
            '-c', 'gunicorn_config.py',
            'src.dashboard_unified:app'
        ])

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
