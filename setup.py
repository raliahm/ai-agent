#!/usr/bin/env python3
"""
Air Mouse Setup Script
Checks system requirements and installs dependencies
"""

import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera detected and accessible")
            cap.release()
            return True
        else:
            print("❌ Camera not accessible")
            return False
    except ImportError:
        print("⚠️  OpenCV not installed - will be installed with requirements")
        return True

def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def main():
    print("🚀 Air Mouse Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check camera after installation
    if not check_camera():
        print("⚠️  Warning: Camera issues detected")
        print("   Make sure your camera is connected and not used by other apps")
    
    print("\n🎉 Setup complete!")
    print("\nTo start the Air Mouse:")
    print("  python main.py")
    print("\nGesture controls:")
    print("  👉 Point index finger → Move cursor")
    print("  🤏 Pinch thumb+index → Click")
    print("  ✊ Curl fingers → Scroll down")
    print("  🖐️ Extend fingers → Scroll up")
    print("  ❌ Press 'q' → Quit")

if __name__ == "__main__":
    main()
