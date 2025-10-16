"""
Launcher script for the Streamlit Time Series Analysis Chatbot
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app"""
    print("🚀 Starting Time Series Analysis Chatbot...")
    print("📊 Opening Streamlit interface...")
    print("🌐 The app will open in your default browser")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down the server...")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        print("💡 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
