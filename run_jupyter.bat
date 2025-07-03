@echo off
echo Instagram Reviews Sentiment Analysis - Jupyter Launcher
echo ============================================================

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Checking Jupyter installation...
jupyter --version >nul 2>&1
if errorlevel 1 (
    echo Installing Jupyter...
    pip install jupyter
    if errorlevel 1 (
        echo ERROR: Failed to install Jupyter
        pause
        exit /b 1
    )
)

echo Launching Jupyter notebook...
echo The notebook will open in your default browser
echo Press Ctrl+C to stop the Jupyter server when done

jupyter notebook instagram_sentiment_analysis.ipynb

echo Jupyter session ended
pause
