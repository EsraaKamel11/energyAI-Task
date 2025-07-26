#!/usr/bin/env python3
"""
Streamlit App Launcher
Handles setup and launches the Streamlit app
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import plotly
        import pandas
        import torch
        import transformers
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements_streamlit.txt")
        return False

def setup_environment():
    """Setup environment variables and paths"""
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Set environment variables
    os.environ.setdefault('STREAMLIT_SERVER_PORT', '8501')
    os.environ.setdefault('STREAMLIT_SERVER_ADDRESS', 'localhost')
    os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
    
    print("✅ Environment setup completed")

def main():
    """Main launcher function"""
    print("🚀 Starting EV Charging LLM Pipeline Streamlit App")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Setup environment
    setup_environment()
    
    # Check if streamlit app exists
    app_path = Path(__file__).parent / "streamlit_app.py"
    if not app_path.exists():
        print("❌ Streamlit app not found: streamlit_app.py")
        return
    
    print("✅ Streamlit app found")
    print("🌐 Launching Streamlit app...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the app")
    print("=" * 60)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error launching Streamlit app: {e}")

if __name__ == "__main__":
    main() 