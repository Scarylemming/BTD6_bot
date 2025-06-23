#!/usr/bin/env python3
"""
Script to install required dependencies for the BTD6 Bot
"""

import subprocess
import sys

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies for BTD6 Bot...")
    
    try:
        # Install pywin32 for Windows API access
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32==306"])
        print("✓ pywin32 installed successfully")
        
        # Install other dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All dependencies installed successfully")
        
        print("\nBot is ready to run in background mode!")
        print("You can now use your computer while the bot works behind the scenes.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("Please try running: pip install -r requirements.txt")
        return False
    
    return True

if __name__ == "__main__":
    install_dependencies() 