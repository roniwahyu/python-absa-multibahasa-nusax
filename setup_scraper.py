#!/usr/bin/env python3
"""
Setup script for Google Play Store scraper
Installs required dependencies and tests the setup.
"""

import subprocess
import sys
import importlib
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def main():
    """Main setup function."""
    print("="*60)
    print("GOOGLE PLAY STORE SCRAPER SETUP")
    print("="*60)
    
    # Required packages
    packages = [
        ('google-play-scraper', 'google_play_scraper'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('tqdm', 'tqdm'),
        ('openpyxl', 'openpyxl'),
    ]
    
    print("\n1. Checking required packages...")
    
    missing_packages = []
    for package, import_name in packages:
        if check_package(package, import_name):
            print(f"✓ {package} is installed")
        else:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n2. Installing missing packages: {missing_packages}")
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✓ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
                return False
    else:
        print("\n✅ All packages are already installed!")
    
    print("\n3. Testing Google Play Store scraper...")
    try:
        from google_play_scraper import app
        
        # Test with a simple app
        test_app = app('com.whatsapp', lang='en', country='us')
        print(f"✓ Test successful! Found app: {test_app['title']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    print("\n4. Setup verification...")
    
    # Check if files exist
    files_to_check = [
        'google_play_multi_scraper.py',
        'google_play_scraper_multi_apps.ipynb',
        'requirements.txt'
    ]
    
    for filename in files_to_check:
        if os.path.exists(filename):
            print(f"✓ {filename} found")
        else:
            print(f"❌ {filename} not found")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nYou can now run the scraper using:")
    print("1. Python script: python google_play_multi_scraper.py")
    print("2. Jupyter notebook: jupyter notebook google_play_scraper_multi_apps.ipynb")
    print("\nOr install all requirements with:")
    print("pip install -r requirements.txt")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
