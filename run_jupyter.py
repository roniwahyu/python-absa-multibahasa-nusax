#!/usr/bin/env python3
"""
Script to launch Jupyter notebook for Instagram sentiment analysis.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_jupyter_installed():
    """Check if Jupyter is installed."""
    try:
        subprocess.run(['jupyter', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_jupyter():
    """Install Jupyter if not available."""
    print("📦 Installing Jupyter...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'jupyter'], check=True)
        print("✅ Jupyter installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Jupyter: {e}")
        return False

def launch_jupyter():
    """Launch Jupyter notebook."""
    notebook_file = "instagram_sentiment_analysis.ipynb"
    
    if not Path(notebook_file).exists():
        print(f"❌ Notebook file {notebook_file} not found!")
        return False
    
    print(f"🚀 Launching Jupyter notebook: {notebook_file}")
    print("📝 The notebook will open in your default browser")
    print("⚠️  Make sure to run cells in order from top to bottom")
    print("🔄 Some cells may take time to complete (especially model loading and scraping)")
    
    try:
        # Launch Jupyter notebook
        subprocess.run([
            'jupyter', 'notebook', 
            notebook_file,
            '--NotebookApp.open_browser=True'
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Jupyter: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Jupyter notebook session ended by user")
        return True

def main():
    """Main function."""
    print("Instagram Reviews Sentiment Analysis - Jupyter Launcher")
    print("=" * 60)
    
    # Check if Jupyter is installed
    if not check_jupyter_installed():
        print("📦 Jupyter not found. Installing...")
        if not install_jupyter():
            print("❌ Could not install Jupyter. Please install manually:")
            print("   pip install jupyter")
            return 1
    
    # Launch Jupyter
    if launch_jupyter():
        print("✅ Jupyter session completed successfully!")
        return 0
    else:
        print("❌ Failed to launch Jupyter notebook")
        return 1

if __name__ == "__main__":
    sys.exit(main())
