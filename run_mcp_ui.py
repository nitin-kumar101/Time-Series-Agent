#!/usr/bin/env python3
"""
Launcher script for MCP UI
Runs the Streamlit interface for the Time Series Analysis MCP Server
"""

import subprocess
import sys
import os

def main():
    """Launch the MCP UI"""
    print("🚀 Starting Time Series Analysis MCP UI...")
    print("This will open a web browser with the interactive interface.")
    print("Make sure the MCP server dependencies are installed.")
    print()
    print("💡 If the UI doesn't work, try running directly:")
    print("   streamlit run mcp_ui.py")
    print()

    # Path to the UI script
    ui_script = os.path.join(os.path.dirname(__file__), "mcp_ui.py")

    # Path to virtual environment Python
    venv_python = os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe")

    if not os.path.exists(venv_python):
        print("❌ Virtual environment not found. Please run from the Time-Series-Agent directory.")
        sys.exit(1)

    if not os.path.exists(ui_script):
        print("❌ MCP UI script not found.")
        sys.exit(1)

    try:
        # Launch Streamlit
        cmd = [venv_python, "-m", "streamlit", "run", ui_script]
        subprocess.run(cmd, cwd=os.path.dirname(__file__))
    except KeyboardInterrupt:
        print("\n👋 MCP UI closed.")
    except Exception as e:
        print(f"❌ Failed to start MCP UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
