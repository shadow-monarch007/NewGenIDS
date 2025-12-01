"""
Quick Start - Launch NextGen IDS Dashboard
----------------------------------------
Simple script to get the system running quickly
"""
import subprocess
import sys
import os

def main():
    print("=" * 80)
    print("🛡️  NextGen IDS - Quick Start")
    print("=" * 80)
    print()
    print("Starting unified dashboard on http://localhost:8080")
    print()
    print("📋 Default Login Credentials:")
    print("   Username: admin")
    print("   Password: admin123")
    print()
    print("   OR")
    print()
    print("   Username: demo")
    print("   Password: demo123")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 80)
    print()
    
    try:
        # Launch dashboard
        subprocess.run([sys.executable, "-m", "src.dashboard_unified"], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure virtual environment is activated")
        print("2. Check all dependencies are installed: pip install -r requirements.txt")
        print("3. Verify Python version >= 3.8")
        sys.exit(1)

if __name__ == '__main__':
    main()
