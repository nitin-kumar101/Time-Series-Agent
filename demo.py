"""
Demo script showing how to use the Streamlit Time Series Analysis Chatbot
"""

import os
import subprocess
import sys

def main():
    print("🚀 Time Series Analysis Chatbot - Streamlit Demo")
    print("=" * 50)
    print()
    print("This demo will start the Streamlit web interface for your")
    print("hybrid RAG + Time Series Analysis chatbot.")
    print()
    print("Features:")
    print("✅ Chat interface for asking questions")
    print("✅ Upload CSV/PDF files for analysis")
    print("✅ Interactive time series visualizations")
    print("✅ Anomaly detection with Prophet")
    print("✅ RAG capabilities for document Q&A")
    print()
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  WARNING: GROQ_API_KEY not found!")
        print("   The chatbot will work but AI features may be limited.")
        print("   Set your API key: export GROQ_API_KEY='your-key-here'")
        print()
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit==1.39.0"])
        print("✅ Streamlit installed")
    
    print()
    print("🌐 Starting Streamlit app...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    print("=" * 50)
    
    try:
        # Start the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down the Streamlit server...")
        print("Thanks for using the Time Series Analysis Chatbot!")

if __name__ == "__main__":
    main()
